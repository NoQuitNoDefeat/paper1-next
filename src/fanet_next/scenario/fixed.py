"""Scripted scenario from explicit positions and births (hand-checkable tests)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..physics import RadioParams
from .base import SCENARIO, Birth, Scenario, all_pairs


class FixedScenario(Scenario):
    def __init__(self, *, positions, births, horizon: int, cycle_length: float,
                 packet_size: int, queue_capacity: int, waiting_capacity: int,
                 waiting_max_wait: float, radio: RadioParams, queue_links=None,
                 velocities=None):
        pos = np.asarray(positions, dtype=float)
        if pos.ndim == 2:  # static nodes
            pos = np.repeat(pos[None], horizon + 1, axis=0)
        if len(pos) < horizon + 1:
            raise ValueError("positions must cover cycles 0..horizon")
        self._pos = pos
        self._vel = (np.zeros_like(pos) if velocities is None
                     else np.broadcast_to(np.asarray(velocities, float), pos.shape).copy())
        self.num_nodes = pos.shape[1]
        self.horizon = int(horizon)
        self.cycle_length = float(cycle_length)
        self.packet_size = int(packet_size)
        self.radio = radio
        self.queue_links = (all_pairs(self.num_nodes) if queue_links is None
                            else np.asarray(queue_links, dtype=np.int64).reshape(-1, 2))
        self.queue_capacity = np.full(len(self.queue_links), int(queue_capacity), dtype=np.int64)
        self.waiting_capacity = np.full(self.num_nodes, int(waiting_capacity), dtype=np.int64)
        self.waiting_max_wait = float(waiting_max_wait)
        self._births: list[list[Birth]] = [[] for _ in range(self.horizon)]
        for b in births:
            t, src, dst = b[0], b[1], b[2]
            size = b[3] if len(b) > 3 else self.packet_size
            k = min(max(math.ceil(round(t / self.cycle_length, 9)) - 1, 0), self.horizon - 1)
            self._births[k].append(Birth(float(t), int(src), int(dst), int(size)))
        for lst in self._births:
            lst.sort(key=lambda b: (b.time, b.src))

    def positions(self, cycle: int) -> np.ndarray:
        return self._pos[cycle].copy()

    def velocities(self, cycle: int) -> np.ndarray:
        return self._vel[min(cycle, self.horizon)].copy()

    def births(self, cycle: int) -> list[Birth]:
        return list(self._births[cycle]) if cycle < self.horizon else []


@SCENARIO.register("fixed", role="variant")
class FixedScenarioSource:
    """Explicit positions (static or per cycle) and birth list; same scene for every seed."""

    def __init__(self, positions=None, births=(), horizon: int = 10, cycle_length: float = 0.02,
                 packet_size: int = 1024, queue_capacity: int = 64, waiting_capacity: int = 16,
                 waiting_max_wait: float = 1.0, queue_links=None, velocities=None,
                 radio: dict | None = None, path: str | None = None):
        if path is not None:
            spec = json.loads(Path(path).read_text())
            positions = spec["positions"]
            births = spec.get("births", births)
            horizon = spec.get("horizon", horizon)
        if positions is None:
            raise ValueError("fixed scenario needs positions or path")
        self.kw = dict(positions=positions, births=list(births), horizon=horizon,
                       cycle_length=cycle_length, packet_size=packet_size,
                       queue_capacity=queue_capacity, waiting_capacity=waiting_capacity,
                       waiting_max_wait=waiting_max_wait, queue_links=queue_links,
                       velocities=velocities, radio=RadioParams(**(radio or {})))

    def make(self, seed: int) -> FixedScenario:
        return FixedScenario(**self.kw)
