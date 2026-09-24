"""Conversion between this project's contracts and the ns-3 bridge (wire v2).

``build_episode`` turns a :class:`Scenario` plus the environment's routing into
one complete INIT payload: registered queues and waiting areas, per-cycle
births, service conditions and route updates, and the observed physical input
of every boundary.  Future inputs live only in ns-3, as in the lightweight
backend.  ``report_from`` / ``facts_from`` map STATE/RESULT back to
:class:`Report` / :class:`CycleFacts`.

Timing: the routes a frame carries take effect at that cycle's end-of-cycle
admission and in the next observation, which is exactly where the lightweight
backend recomputes routes (boundary k+1 when (k+1) % update_every == 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...contracts import CycleFacts, QueueSnapshot, Report
from ...physics import lin_to_db, path_gain, set_sinr
from ...scenario.base import Scenario
from ...scenario.routing import Routing

NS = 1_000_000_000
EPS = 1e-9
TERMINAL_REASON = {"route_wait_timeout": "waiting_timeout", "queue_capacity": "queue_overflow",
                   "waiting_capacity": "waiting_overflow"}


@dataclass
class Episode:
    """Everything precomputed for one episode (the backend's private inputs)."""

    payload: dict
    period_ns: int
    horizon: int
    packet_size: int
    qlinks: np.ndarray
    positions: list[np.ndarray]
    velocities: list[np.ndarray]
    gains: list[np.ndarray]  # observed gains per boundary (ideal channel: the truth)
    next_hops: list[np.ndarray]  # route table in effect at each boundary
    power: np.ndarray
    noise: float
    threshold: float
    service_bytes: int
    source_of: dict[int, int] = field(default_factory=dict)


def _routes(next_hop: np.ndarray) -> list[dict]:
    n = len(next_hop)
    return [{"node_id": u, "destination": d,
             "next_hop": int(next_hop[u, d]) if next_hop[u, d] >= 0 else None}
            for u in range(n) for d in range(n) if u != d]


def build_episode(scenario: Scenario, routing: Routing, *, execution_profile: str) -> Episode:
    sc, radio = scenario, scenario.radio
    n, horizon, ps = sc.num_nodes, sc.horizon, sc.packet_size
    period_ns = int(round(sc.cycle_length * NS))
    rate = int(radio.rate_bps // 8)
    if radio.service_bytes != rate * period_ns // NS:
        raise ValueError("ns-3 bridge requires a service window equal to the cycle "
                         f"({radio.service_bytes} B vs rate x period {rate * period_ns // NS} B)")
    power = np.full(n, radio.tx_power_w)
    noise, threshold = radio.noise_w, radio.threshold
    qlinks = np.asarray(sc.queue_links, dtype=np.int64)
    positions = [sc.positions(k) for k in range(horizon + 1)]
    velocities = [sc.velocities(k) for k in range(horizon + 1)]
    gains = [path_gain(p, radio) for p in positions]
    ratios = [power[:, None] * g / (threshold * noise) for g in gains]

    def table(k):
        adjacency = ratios[k] >= 1 - EPS
        np.fill_diagonal(adjacency, False)
        return routing.compute(adjacency, ratios[k])

    next_hops = [table(0)]
    for k in range(horizon):
        next_hops.append(table(k + 1) if (k + 1) % routing.update_every == 0 else next_hops[-1])

    config = {
        "node_ids": list(range(n)),
        "queues": [{"key": [int(u), int(v)], "capacity_bytes": int(c) * ps}
                   for (u, v), c in zip(qlinks, sc.queue_capacity)],
        "waiting_areas": [{"node_id": i, "capacity_bytes": int(sc.waiting_capacity[i]) * ps,
                           "max_wait_ns": int(round(sc.waiting_max_wait * NS))} for i in range(n)],
        "packet_size_bytes": ps, "start_ns": 0, "end_ns": horizon * period_ns,
        "period_ns": period_ns, "packet_deadlines_enabled": False,
        "retransmissions_enabled": False,
    }
    initial = {"packets": [],
               "queues": [{"key": [int(u), int(v)], "entries": []} for u, v in qlinks],
               "waiting_areas": [{"node_id": i, "entries": []} for i in range(n)],
               "routes": _routes(next_hops[0])}
    frames, source_of, pid = [], {}, 0
    for k in range(horizon):
        start, end = k * period_ns, (k + 1) * period_ns
        available = ratios[k] >= 1 - EPS
        services = [{"key": [int(u), int(v)],
                     "budget_bytes": radio.service_bytes if available[u, v] else 0,
                     "completed_at_ns": end, "available": bool(available[u, v]),
                     "rate_bytes_per_second": rate if available[u, v] else None}
                    for u, v in qlinks]
        births = []
        for b in sc.births(k):
            if b.size != ps:
                raise ValueError("ns-3 bridge requires a single packet size")
            born = min(max(int(round(b.time * NS)), start + 1), end)
            births.append({"packet_id": pid, "source": b.src, "destination": b.dst,
                           "born_at_ns": born, "deadline_ns": None})
            source_of[pid] = b.src
            pid += 1
        updates = _routes(next_hops[k + 1]) if (k + 1) % routing.update_every == 0 else []
        frames.append({"start_ns": start, "services": services, "births": births,
                       "route_updates": updates})

    def physical(k):
        snr = ratios[k] * threshold
        links = [{"key": [int(u), int(v)], "base_quality_db": float(lin_to_db(snr[u, v])),
                  "rate_bytes_per_second": rate, "sinr_threshold": float(threshold),
                  "success_probability": 1.0}
                 for u in range(n) for v in range(n) if u != v and ratios[k][u, v] >= 1 - EPS]
        return {"at_ns": k * period_ns,
                "nodes": [{"node_id": i, "position_m": [float(x) for x in positions[k][i]],
                           "velocity_m_s": [float(x) for x in velocities[k][i]],
                           "transmit_power_w": float(power[i]), "radio_count": 1,
                           "duplex_mode": "HD", "channels": [0]} for i in range(n)],
                "links": links,
                # every (tx, rx) pair, including a node with itself (zero gain): the bridge
                # requires the complete interference path between any two link endpoints
                "gains": [{"sender": s, "receiver": r, "gain": float(gains[k][s, r])}
                          for s in range(n) for r in range(n)],
                "noise_w": float(noise), "channel_id": 0}

    payload = {"config": config, "initial": initial, "trajectory": {"frames": frames},
               "physical_inputs": [physical(k) for k in range(horizon + 1)],
               "metadata": dict.fromkeys(("feature_schema_sha256", "history_sha256",
                                          "reward_sha256", "checkpoint_sha256"))}
    if execution_profile != "controlled-budget-v1":
        payload["execution_profile"] = execution_profile
    return Episode(payload=payload, period_ns=period_ns, horizon=horizon, packet_size=ps,
                   qlinks=qlinks, positions=positions, velocities=velocities, gains=gains,
                   next_hops=next_hops, power=power, noise=noise, threshold=threshold,
                   service_bytes=radio.service_bytes, source_of=source_of)


def result_capacity(payload: dict, physical_bytes: int) -> int:
    """Mirror of the C++ ``RequiredResultCapacity`` (no radio/motion/control terms)."""
    c, beyond = payload["config"], 100_001
    ps = c["packet_size_bytes"]
    frames = payload["trajectory"]["frames"]
    slots = sum(q["capacity_bytes"] // ps for q in c["queues"])
    slots += sum(w["capacity_bytes"] // ps for w in c["waiting_areas"])
    lifetime = min(beyond, len(payload["initial"]["packets"]) + sum(len(f["births"]) for f in frames))
    peak = max((len(f["births"]) for f in frames), default=0)
    packets = min(lifetime, min(beyond, slots + peak))
    routes = {(r["node_id"], r["destination"]) for r in payload["initial"]["routes"]}
    routes.update((r["node_id"], r["destination"]) for f in frames for r in f["route_updates"])
    return (16384 + physical_bytes + 4500 * packets + 1500 * len(c["queues"])
            + 1000 * len(c["waiting_areas"]) + 150 * len(routes) + 20 * len(c["node_ids"]))


# ------------------------------------------------------------------- inbound
def _queue_snapshot(queues: list[dict], ps: int, sampled_ns: int) -> QueueSnapshot:
    packets = np.array([len(q["entries"]) for q in queues], dtype=np.int64)
    nbytes = np.array([q["occupancy_bytes"] for q in queues], dtype=np.int64)
    cap = np.array([q["capacity_bytes"] // ps for q in queues], dtype=np.int64)
    hol = np.array([q["hol_ns"] / NS for q in queues], dtype=float)
    return QueueSnapshot(packets, nbytes, cap, hol)


def report_from(obs: dict, ep: Episode, *, run_id: str, episode: int) -> Report:
    k = obs["reference"]["period_index"]
    sampled = obs["reference"]["sampled_at_ns"]
    n = len(obs["node_ids"])
    next_hop = -np.ones((n, n), dtype=np.int64)
    for r in obs["routes"]:
        if r["next_hop"] is not None:
            next_hop[r["node_id"], r["destination"]] = r["next_hop"]
    u, v = ep.qlinks[:, 0], ep.qlinks[:, 1]
    waiting = obs["waiting_areas"]
    oldest = [(sampled - min(e["waiting_since_ns"] for e in w["entries"])) / NS if w["entries"] else 0.0
              for w in waiting]
    return Report(
        run_id=run_id, episode=episode, cycle=k, time=sampled / NS,
        cycle_length=ep.period_ns / NS, num_nodes=n, positions=ep.positions[k],
        velocities=ep.velocities[k], gain=ep.gains[k], tx_power=ep.power.copy(),
        noise=ep.noise, threshold=ep.threshold, service_bytes=ep.service_bytes,
        packet_size=ep.packet_size, queue_links=ep.qlinks.copy(),
        queues=_queue_snapshot(obs["queues"], ep.packet_size, sampled),
        route_next=(next_hop[u] == v[:, None]).any(axis=1), next_hop=next_hop,
        waiting_packets=np.array([len(w["entries"]) for w in waiting], dtype=np.int64),
        waiting_capacity=np.array([w["capacity_bytes"] // ep.packet_size for w in waiting],
                                  dtype=np.int64),
        waiting_oldest=np.maximum(np.array(oldest), 0.0),
        waiting_max_wait=waiting[0]["max_wait_ns"] / NS if waiting else 0.0)


def facts_from(cycle: dict, ep: Episode, k: int) -> CycleFacts:
    ps = ep.packet_size
    planned = tuple((int(a), int(b)) for a, b in cycle["action"]["links"])
    served_p = np.array([len(s["packet_ids"]) for s in cycle["services"]], dtype=np.int64)
    served_b = np.array([s["bytes_served"] for s in cycle["services"]], dtype=np.int64)
    served = {(int(s["key"][0]), int(s["key"][1])): len(s["packet_ids"]) for s in cycle["services"]}
    succeeded = tuple(l for l in planned if served.get(l, 0) > 0)
    failed = tuple(l for l in planned if served.get(l, 0) == 0)
    links = np.array(planned, dtype=np.int64).reshape(-1, 2)
    exec_sinr = set_sinr(links, ep.gains[k], ep.power, ep.noise)
    post = cycle["post_service"]
    events = cycle["events"]
    delivered = [e for e in events if e["kind"] == "delivery"]
    terminations: dict[str, list[int]] = {}
    relay_terminated = 0
    for e in events:
        if e["kind"] == "terminal":
            reason = TERMINAL_REASON.get(e["reason"], e["reason"])
            terminations.setdefault(reason, []).append(e["packet_id"])
            relay_terminated += int(e["node_id"] != ep.source_of.get(e["packet_id"], e["node_id"]))
    nxt = cycle["next_observation"]
    return CycleFacts(
        cycle=k, t_start=k * ep.period_ns / NS, t_end=cycle["ended_at_ns"] / NS,
        planned=planned, succeeded=succeeded, failed=failed, exec_sinr=exec_sinr,
        served_packets=served_p, served_bytes=served_b, service_ref_bytes=ep.service_bytes,
        post_service=_queue_snapshot(post["queues"], ps, post["sampled_at_ns"]),
        births=len(cycle["new_source_ids"]), risk_packets=len(cycle["risk_ids"]),
        delivered_ids=[e["packet_id"] for e in delivered],
        delivered_delays=[e["end_to_end_ns"] / NS for e in delivered],
        delivered_bytes=len(delivered) * ps, terminations=terminations,
        queued_end=sum(len(q["entries"]) for q in nxt["queues"]),
        waiting_end=sum(len(w["entries"]) for w in nxt["waiting_areas"]),
        waiting_post_service=sum(len(w["entries"]) for w in post["waiting_areas"]),
        moved_to_waiting=sum(e["kind"] == "wait_admit" for e in events),
        restored_from_waiting=sum(e["kind"] == "queue_admit" and e["origin"] == "waiting"
                                  for e in events),
        rehomed=0, relay_terminated=relay_terminated)
