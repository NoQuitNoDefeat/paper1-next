"""ns-3 backend: the same contract as the lightweight backend, executed by ns-3.

One episode = one supervised ns-3 run (fresh processes and shared-memory
segment).  INIT carries every private future input once; each cycle sends one
complete PLAN and receives one RESULT (no per-micro-step round trips).  ns-3
owns the authoritative packet state; Python only converts records.
"""

from __future__ import annotations

import hashlib
import math
import uuid

from ...contracts import EndType, ExecutionError, Plan, Report, StepOutcome
from ...scenario.base import Scenario
from ...scenario.channel import CHANNEL, IdealChannel
from ...scenario.routing import ROUTING, Routing
from ..base import BACKEND, Backend
from .convert import (RADIO_PROFILES, build_episode, facts_from, radio_settings, report_from,
                      result_capacity)
from .process import DEFAULT_NS3_ROOT, Supervisor, TransportConfig
from .wire import HEADER, Kind, ProtocolError, decode, encode, encode_value

PROFILES = ("full-sinr-v1", "controlled-budget-v1", *RADIO_PROFILES)


def _mib(nbytes: int) -> int:
    return int(math.ceil(nbytes / (1 << 20))) << 20


@BACKEND.register("ns3", role="variant")
class Ns3Backend(Backend):
    """ns-3.48 fanet-scheduler bridge (wire v2): frozen-CSI ledger or PHY radio (DATA/ACK, motion).

    Profiles: ``full-sinr-v1`` (cumulative SINR on observed gains, no PHY),
    ``ideal-spectrum-v1`` (spectrum PHY, ideal confirmation) and ``transaction-ack-v1``
    (spectrum PHY with DATA/ACK transactions; ``motion=True`` adds continuous motion
    with per-reception quasi-static gains).  Radio options go to ``radio``.
    """

    supports_state = False

    def __init__(self, channel: dict | str = "ideal", routing: dict | str = "min_hop",
                 stale_queue_policy: str = "rehome", waiting_restore: str = "keep",
                 execution_profile: str = "full-sinr-v1", motion: bool = False,
                 radio: dict | None = None, ns3_root: str | None = None,
                 timeout_seconds: float = 60.0):
        self.channel = CHANNEL.build(channel)
        if not isinstance(self.channel, IdealChannel):
            raise ValueError("ns3 full-sinr profile executes on the observed (ideal) channel")
        self.routing: Routing = ROUTING.build(routing)
        if stale_queue_policy not in {"rehome", "keep"} or waiting_restore not in {"keep", "drop"}:
            raise ValueError("stale_queue_policy in {rehome, keep}, waiting_restore in {keep, drop}")
        self.semantics = {"stale_queue_policy": stale_queue_policy,
                          "waiting_restore": waiting_restore}
        if execution_profile not in PROFILES:
            raise ValueError(f"execution_profile must be one of {PROFILES}")
        self.profile = execution_profile
        if motion and execution_profile != "transaction-ack-v1":
            raise ValueError("motion requires execution_profile = transaction-ack-v1")
        self.motion = bool(motion)
        self.radio_options = dict(radio or {})
        self.ns3_root = ns3_root
        self.timeout = float(timeout_seconds)
        self._sup: Supervisor | None = None

    # --------------------------------------------------------------- transport
    def _request(self, kind, expected, epoch, at, payload, *, next_epoch=False):
        data = encode(kind, self._run, self._seq, epoch, at, self._digest, payload,
                      self._transport.tx_capacity)
        resp = self._sup.exchange(data)
        actual, run, seq, r_epoch, r_at, digest, body = decode(resp, self._transport.rx_capacity)
        if run != self._run or seq != self._seq or digest != self._digest:
            raise ProtocolError("response run/sequence/config identity mismatch")
        if actual == Kind.ERROR:
            raise ProtocolError(f"ns-3 rejected {kind.name}: {body}")
        if (actual != expected or r_epoch != epoch + int(next_epoch)
                or r_at != at + (self._ep.period_ns if next_epoch else 0)):
            raise ProtocolError("response kind/epoch/time mismatch")
        self._seq += 1
        return body

    def close(self) -> None:
        if self._sup is not None:
            self._sup.abort()
            self._sup = None

    def _fail(self, error: Exception) -> ExecutionError:
        self.close()
        return ExecutionError(f"ns-3 backend: {error}")

    # ------------------------------------------------------------------ run
    def reset(self, scenario: Scenario, *, run_id: str = "run", episode: int = 0,
              seed: int = 0) -> Report:
        self.close()
        self.run_id, self.episode = run_id, episode
        wireless = (radio_settings(scenario, self.profile, motion=self.motion, **self.radio_options)
                    if self.profile in RADIO_PROFILES else None)
        self._ep = build_episode(scenario, self.routing, execution_profile=self.profile,
                                 wireless=wireless)
        self._ep.payload["config"].update(self.semantics)
        body = encode_value(self._ep.payload, limit=1 << 30)
        physical = max(len(encode_value(p)) for p in self._ep.payload["physical_inputs"])
        tx = _mib(len(body) + HEADER.size + 4096)
        rx = _mib(result_capacity(self._ep.payload, physical) + HEADER.size + 4096)
        shm = _mib(tx + rx + 8192)
        kwargs = {"ns3_root": self.ns3_root} if self.ns3_root else {}
        self._transport = TransportConfig(shm_bytes=shm, tx_capacity=tx, rx_capacity=rx,
                                          timeout_seconds=self.timeout, **kwargs)
        self._run = uuid.uuid4().hex
        self._digest = hashlib.sha256(body).digest()
        self._seq = 0
        try:
            self._sup = Supervisor(self._transport, self._run)
            hello = {"shm_bytes": shm, "tx_capacity": tx, "rx_capacity": rx}
            ready = self._request(Kind.HELLO, Kind.READY, 0, 0, hello)
            source, ns3, ai = self._sup.identity
            if (ready.get("source_sha256") != source or ready.get("ns3_commit") != ns3
                    or ready.get("ns3_ai_commit") != ai
                    or any(ready.get(k) != v for k, v in hello.items())):
                raise ProtocolError("ns-3 build identity or capacities do not match this project")
            self.build = {k: ready[k] for k in ("source_sha256", "ns3_commit", "ns3_ai_commit",
                                                 "compiler")}
            state = self._request(Kind.INIT, Kind.STATE, 0, 0, self._ep.payload)
        except Exception as error:  # noqa: BLE001
            raise self._fail(error) from error
        self.cycle = 0
        self._obs = state["observation"]
        self.report = report_from(self._obs, self._ep, run_id=run_id, episode=episode)
        return self.report

    def execute(self, plan: Plan) -> StepOutcome:
        if self._sup is None:
            raise ExecutionError("ns-3 backend: no active run")
        k = self.cycle
        if plan.cycle != k:
            raise ExecutionError(f"plan for cycle {plan.cycle} but backend is at {k}")
        ref = self._obs["reference"]
        at = ref["sampled_at_ns"]
        timing = dict.fromkeys(("sampled_at_ns", "report_received_at_ns", "plan_created_at_ns",
                                "command_received_at_ns", "execute_at_ns"), at)
        timing["deadline_ns"] = at + self._ep.period_ns
        action = {"reference": ref, "links": [[int(a), int(b)] for a, b in plan.links]}
        try:
            body = self._request(Kind.PLAN, Kind.RESULT, k, at,
                                 {"action": action, "timing": timing}, next_epoch=True)
            if body.get("timing") != timing:
                raise ProtocolError("result timing mismatch")
            cycle = body["cycle"]
            facts = facts_from(cycle, self._ep, k, body.get("radio"))
            self._obs = cycle["next_observation"]
            self.cycle = k + 1
            self.report = report_from(self._obs, self._ep, run_id=self.run_id,
                                      episode=self.episode)
            end = EndType.TRUNCATED if self.cycle >= self._ep.horizon else EndType.CONTINUE
            if end is not EndType.CONTINUE:
                self._finish()
        except ExecutionError:
            raise
        except Exception as error:  # noqa: BLE001
            raise self._fail(error) from error
        return StepOutcome(facts=facts, next_report=self.report, end=end)

    def _finish(self) -> None:
        ref = self._obs["reference"]
        final = self._request(Kind.STOP, Kind.FINAL, ref["period_index"], ref["sampled_at_ns"],
                              {"reason": "range_end"})
        if final.get("truncated") is not True or final.get("terminated") is not False:
            raise ProtocolError("invalid final lifecycle fields")
        self._request(Kind.ACK, Kind.CLOSED, ref["period_index"], ref["sampled_at_ns"], {})
        self._sup.finish()
        self._sup = None
