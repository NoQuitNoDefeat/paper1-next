"""Rollout collection over lockstep environments, sliced at physical-cycle boundaries.

Environments keep running across rollouts.  Each rollout ends every stream
with a ``cut`` record (bootstrap with V_old of the current state); episodes
that hit their horizon end with ``truncated`` (bootstrap with the last valid
state).  Execution errors are counted and reported; the failed cycle is
dropped, the previous record becomes a ``cut`` and the environment restarts —
an error is never turned into a normal end or a zero reward.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..contracts import EndType, ExecutionError
from ..loop import SchedulingEnv
from ..policy.learned import LearnedPolicy
from .normalize import ReturnScaler
from .records import CUT, CycleRecord


@dataclass
class SeedStream:
    base: int
    counter: int = 0

    def next(self) -> int:
        s = self.base + self.counter
        self.counter += 1
        return s


@dataclass
class RolloutResult:
    streams: list[list[CycleRecord]]
    episodes: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def records(self) -> list[CycleRecord]:
        return [r for s in self.streams for r in s]


class RolloutCollector:
    def __init__(self, envs: list[SchedulingEnv], policy: LearnedPolicy, *, seeds: SeedStream,
                 scaler: ReturnScaler, rollout_cycles: int):
        self.envs = envs
        self.policy = policy
        self.seeds = seeds
        self.scaler = scaler
        self.rollout_cycles = rollout_cycles
        self.episode_ids = [-1] * len(envs)
        self.episode_counter = 0

    def _start(self, e: int) -> None:
        self.episode_ids[e] = self.episode_counter
        self.episode_counter += 1
        self.envs[e].reset(self.seeds.next(), episode=self.episode_ids[e])

    def collect(self) -> RolloutResult:
        t0 = time.perf_counter()
        for e, env in enumerate(self.envs):
            if env.current is None:
                self._start(e)
        streams: list[list[CycleRecord]] = [[] for _ in self.envs]
        result = RolloutResult(streams)
        for _ in range(self.rollout_cycles):
            inputs = [env.current for env in self.envs]
            outs = self.policy.act(inputs, mode="sample")
            truncated: list[tuple[CycleRecord, object]] = []
            restart: list[int] = []
            for e, (env, out) in enumerate(zip(self.envs, outs)):
                micro = out.record["micro"]
                cycle = env.current.report.cycle
                try:
                    tr = env.step(out.actions)
                except ExecutionError as exc:
                    result.errors.append(f"env {e} episode {self.episode_ids[e]} cycle {cycle}: {exc}")
                    if streams[e]:
                        prev = streams[e][-1]
                        prev.end, prev.bootstrap = CUT, float(micro.values[0])
                    restart.append(e)
                    continue
                ended = tr.end is not EndType.CONTINUE
                rec = CycleRecord(micro=micro, reward=self.scaler(e, tr.reward.total, ended),
                                  raw_reward=tr.reward.total, end=tr.end.value, env=e,
                                  episode=self.episode_ids[e], cycle=cycle)
                streams[e].append(rec)
                if tr.end is EndType.TRUNCATED:
                    truncated.append((rec, tr.next_input))
                if ended:
                    summary = env.metrics.summary()
                    summary.update(seed=env.seed, episode=self.episode_ids[e])
                    result.episodes.append(summary)
                    restart.append(e)
            if truncated:
                vals = self.policy.values_of([inp for _, inp in truncated])
                for (rec, _), v in zip(truncated, vals):
                    rec.bootstrap = float(v)
            for e in restart:
                self._start(e)
        # slice end: bootstrap from the current state, stop the recursion
        open_ = [e for e, s in enumerate(streams) if s and s[-1].end == "continue"]
        if open_:
            vals = self.policy.values_of([self.envs[e].current for e in open_])
            for e, v in zip(open_, vals):
                streams[e][-1].end, streams[e][-1].bootstrap = CUT, float(v)
        result.seconds = time.perf_counter() - t0
        return result

    # -------------------------------------------------------------- state
    def state_dict(self) -> dict:
        return {"envs": self.envs, "episode_ids": list(self.episode_ids),
                "episode_counter": self.episode_counter, "seeds": (self.seeds.base, self.seeds.counter),
                "scaler": self.scaler.state_dict()}

    def load_state_dict(self, s: dict, restore_envs: bool = True) -> None:
        if restore_envs:
            self.envs[:] = s["envs"]
            self.episode_ids = list(s["episode_ids"])
        self.episode_counter = s["episode_counter"]
        self.seeds = SeedStream(*s["seeds"])
        self.scaler.load_state_dict(s["scaler"])
