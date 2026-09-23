"""Random 3-D mobility with Poisson flows (training / evaluation generator)."""

from __future__ import annotations

import math

import numpy as np

from ..physics import RadioParams
from .base import SCENARIO, Birth, Scenario, all_pairs


def _pick(value, rng: np.random.Generator, integer: bool = False):
    """A scalar is fixed; a two-element list is a uniform range sampled per episode."""
    if isinstance(value, (list, tuple)):
        lo, hi = value
        return int(rng.integers(lo, hi + 1)) if integer else float(rng.uniform(lo, hi))
    return int(value) if integer else float(value)


class RandomScenario(Scenario):
    def __init__(self, *, rng: np.random.Generator, num_nodes: int, area_m: float,
                 altitude_m: tuple[float, float], speed_mps: float, vertical_speed_mps: float,
                 flow_rate_pps: float, cycle_length: float, horizon: int, packet_size: int,
                 queue_capacity: int, waiting_capacity: int, waiting_max_wait: float,
                 radio: RadioParams, traffic_stop_cycle: int | None):
        self.num_nodes = n = num_nodes
        self.cycle_length = cycle_length
        self.horizon = horizon
        self.packet_size = packet_size
        self.radio = radio
        self.queue_links = all_pairs(n)
        self.queue_capacity = np.full(len(self.queue_links), queue_capacity, dtype=np.int64)
        self.waiting_capacity = np.full(n, waiting_capacity, dtype=np.int64)
        self.waiting_max_wait = waiting_max_wait
        self.flow_rate_pps = flow_rate_pps
        self.speed_mps = speed_mps

        lo = np.array([0.0, 0.0, altitude_m[0]])
        hi = np.array([area_m, area_m, altitude_m[1]])
        pos = lo + rng.random((n, 3)) * (hi - lo)
        theta = rng.uniform(0, 2 * math.pi, n)
        vel = np.stack([speed_mps * np.cos(theta), speed_mps * np.sin(theta),
                        rng.uniform(-vertical_speed_mps, vertical_speed_mps, n)], axis=1)
        self._pos = np.empty((horizon + 1, n, 3))
        self._vel = np.empty((horizon + 1, n, 3))
        for k in range(horizon + 1):
            self._pos[k], self._vel[k] = pos, vel
            pos = pos + vel * cycle_length
            # specular reflection at the box walls
            below, above = pos < lo, pos > hi
            pos = np.where(below, 2 * lo - pos, pos)
            pos = np.where(above, 2 * hi - pos, pos)
            vel = np.where(below | above, -vel, vel)

        # one flow per node; destinations form a random derangement (cycle)
        perm = rng.permutation(n)
        self.flows = [(int(perm[i]), int(perm[(i + 1) % n])) for i in range(n)]
        stop = horizon if traffic_stop_cycle is None else min(horizon, traffic_stop_cycle)
        self.traffic_stop_cycle = stop
        t_end = stop * cycle_length
        births: list[Birth] = []
        for src, dst in self.flows:
            if flow_rate_pps <= 0:
                continue
            t = 0.0
            while True:
                t += rng.exponential(1.0 / flow_rate_pps)
                if t > t_end:
                    break
                births.append(Birth(t, src, dst, packet_size))
        births.sort(key=lambda b: (b.time, b.src))
        self._births: list[list[Birth]] = [[] for _ in range(horizon)]
        for b in births:
            # birth in (k*T, (k+1)*T] belongs to cycle k
            k = min(max(math.ceil(b.time / cycle_length) - 1, 0), horizon - 1)
            self._births[k].append(b)

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
        d.update(flow_rate_pps=self.flow_rate_pps, speed_mps=self.speed_mps, flows=self.flows)
        return d


@SCENARIO.register("random", role="primary")
class RandomScenarioSource:
    """Random-direction 3-D mobility in a box, one Poisson flow per node, min-hop routing."""

    def __init__(self, num_nodes=12, area_m: float = 400.0, altitude_m=(80.0, 120.0),
                 speed_mps=(5.0, 30.0), vertical_speed_mps: float = 1.0, flow_rate_pps=(30.0, 60.0),
                 cycle_length: float = 0.02, horizon: int = 500, packet_size: int = 1024,
                 queue_capacity: int = 64, waiting_capacity: int = 16, waiting_max_wait: float = 1.0,
                 traffic_stop_cycle: int | None = None, radio: dict | None = None):
        self.params = dict(num_nodes=num_nodes, area_m=area_m, altitude_m=tuple(altitude_m),
                           speed_mps=speed_mps, vertical_speed_mps=vertical_speed_mps,
                           flow_rate_pps=flow_rate_pps, cycle_length=cycle_length, horizon=horizon,
                           packet_size=packet_size, queue_capacity=queue_capacity,
                           waiting_capacity=waiting_capacity, waiting_max_wait=waiting_max_wait,
                           traffic_stop_cycle=traffic_stop_cycle)
        self.radio = RadioParams(**(radio or {}))

    def make(self, seed: int) -> RandomScenario:
        rng = np.random.default_rng(seed)
        p = dict(self.params)
        p["num_nodes"] = _pick(p["num_nodes"], rng, integer=True)
        p["speed_mps"] = _pick(p["speed_mps"], rng)
        p["flow_rate_pps"] = _pick(p["flow_rate_pps"], rng)
        return RandomScenario(rng=rng, radio=self.radio, **p)
