"""Trace-driven scenario: node positions replayed from recorded trajectories.

A trace file (``.npz``) holds ``t`` (T,) seconds on a uniform grid and ``pos``
(T, N, 3) metres (x east, y north, z above ground), e.g. a public multi-UAV flight
log or a third-party mobility generator's output (tools/data/).  Traffic, queues
and radio are as in the random source (one Poisson flow per node).

Splits are by whole trace: the seed's range (``fanet_next.seeds``) selects the
trace names allowed for training, development or test, so a test trace never
appears in training.  Per episode the seed picks one allowed trace, a start time
inside it and ``num_nodes`` of its nodes; positions at the cycle boundaries are
linearly interpolated, velocities are the boundary-to-boundary displacement.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from ..physics import RadioParams
from ..seeds import split_of
from .base import SCENARIO, Birth, Scenario, all_pairs
from .random_scenario import _pick, poisson_flows

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=64)
def load_trace(path: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as f:
        t, pos = np.asarray(f["t"], dtype=float), np.asarray(f["pos"], dtype=float)
    if t.ndim != 1 or pos.shape[:1] != t.shape or pos.ndim != 3 or pos.shape[2] != 3:
        raise ValueError(f"{path}: expected t (T,) and pos (T, N, 3)")
    if np.any(np.diff(t) <= 0):
        raise ValueError(f"{path}: time must increase")
    return t, pos


class TraceScenario(Scenario):
    def __init__(self, *, trace: str, start: float, nodes: np.ndarray, t: np.ndarray, pos: np.ndarray,
                 rng: np.random.Generator, flow_rate_pps: float, cycle_length: float, horizon: int,
                 packet_size: int, queue_capacity: int, waiting_capacity: int,
                 waiting_max_wait: float, radio: RadioParams, traffic_stop_cycle: int | None):
        self.trace, self.start, self.nodes = trace, float(start), nodes
        self.num_nodes = n = len(nodes)
        self.cycle_length, self.horizon, self.packet_size = cycle_length, horizon, packet_size
        self.radio = radio
        self.flow_rate_pps = flow_rate_pps
        self.queue_links = all_pairs(n)
        self.queue_capacity = np.full(len(self.queue_links), queue_capacity, dtype=np.int64)
        self.waiting_capacity = np.full(n, waiting_capacity, dtype=np.int64)
        self.waiting_max_wait = waiting_max_wait
        times = self.start + cycle_length * np.arange(horizon + 2)
        sel = pos[:, nodes, :]
        self._pos = np.stack([np.stack([np.interp(times, t, sel[:, i, j]) for j in range(3)], 1)
                              for i in range(n)], 1)  # (horizon + 2, n, 3)
        self._vel = np.diff(self._pos, axis=0) / cycle_length  # (horizon + 1, n, 3)
        self._pos = self._pos[:horizon + 1]
        self.flows, self._births, self.traffic_stop_cycle = poisson_flows(
            rng, n, flow_rate_pps, cycle_length, horizon, traffic_stop_cycle, packet_size)

    def positions(self, cycle: int) -> np.ndarray:
        return self._pos[cycle].copy()

    def velocities(self, cycle: int) -> np.ndarray:
        return self._vel[min(cycle, self.horizon)].copy()

    def births(self, cycle: int) -> list[Birth]:
        return list(self._births[cycle]) if cycle < self.horizon else []

    def traffic_done(self, cycle: int) -> bool:
        return cycle >= self.traffic_stop_cycle

    def describe(self) -> dict:
        d = super().describe()
        d.update(trace=self.trace, start_s=self.start, nodes=self.nodes.tolist(),
                 flow_rate_pps=self.flow_rate_pps, flows=self.flows)
        return d


@SCENARIO.register("trace", role="variant")
class TraceScenarioSource:
    """Recorded trajectories (public flight logs or a third-party mobility generator), split by trace."""

    def __init__(self, dir: str, train: list[str] = (), dev: list[str] = (), test: list[str] = (),
                 num_nodes=16, flow_rate_pps=(10.0, 30.0), cycle_length: float = 0.02,
                 horizon: int = 500, packet_size: int = 1024, queue_capacity: int = 64,
                 waiting_capacity: int = 16, waiting_max_wait: float = 1.0,
                 traffic_stop_cycle: int | None = None, radio: dict | None = None):
        root = Path(dir)
        self.dir = root if root.is_absolute() else PROJECT_ROOT / root
        self.splits = {"train": list(train), "dev": list(dev), "test": list(test)}
        overlap = set(self.splits["train"]) & (set(self.splits["dev"]) | set(self.splits["test"]))
        if overlap or set(self.splits["dev"]) & set(self.splits["test"]):
            raise ValueError("trace splits must be disjoint")
        self.params = dict(num_nodes=num_nodes, flow_rate_pps=flow_rate_pps, cycle_length=cycle_length,
                           horizon=horizon, packet_size=packet_size, queue_capacity=queue_capacity,
                           waiting_capacity=waiting_capacity, waiting_max_wait=waiting_max_wait,
                           traffic_stop_cycle=traffic_stop_cycle)
        self.radio = RadioParams(**(radio or {}))

    def make(self, seed: int) -> TraceScenario:
        split = split_of(seed)
        names = self.splits[split]
        if not names:
            raise ValueError(f"trace scenario: no traces for the {split} split (seed {seed})")
        rng = np.random.default_rng(seed)
        p = dict(self.params)
        name = names[int(rng.integers(len(names)))]
        t, pos = load_trace(str(self.dir / f"{name}.npz"))
        span = (p["horizon"] + 1) * p["cycle_length"]
        if t[-1] - t[0] < span:
            raise ValueError(f"trace {name} is shorter than an episode ({t[-1] - t[0]:.1f} s < {span:.1f} s)")
        start = float(rng.uniform(t[0], t[-1] - span))
        n = _pick(p.pop("num_nodes"), rng, integer=True)
        if n > pos.shape[1]:
            raise ValueError(f"trace {name} has {pos.shape[1]} nodes, {n} requested")
        nodes = np.sort(rng.choice(pos.shape[1], size=n, replace=False))
        p["flow_rate_pps"] = _pick(p["flow_rate_pps"], rng)
        return TraceScenario(trace=name, start=start, nodes=nodes, t=t, pos=pos, rng=rng,
                             radio=self.radio, **p)
