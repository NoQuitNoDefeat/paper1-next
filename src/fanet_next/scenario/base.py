"""Scenario contract: the exogenous processes of one episode.

A ``Scenario`` is created per episode by a scenario source (slot ``scenario``)
and is owned by the backend.  Future positions and births live only here; the
scheduler sees them only through backend reports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..physics import RadioParams
from ..registry import slot

SCENARIO = slot("scenario", "initial state, mobility and traffic of an episode")


@dataclass(frozen=True)
class Birth:
    time: float  # absolute birth time, s
    src: int
    dst: int
    size: int  # bytes


class Scenario(ABC):
    """Exogenous processes for one episode.

    Positions are defined at cycle boundaries ``k * cycle_length``; the backend
    may interpolate within a cycle with ``velocities``.
    """

    num_nodes: int
    cycle_length: float
    horizon: int  # number of cycles in the episode
    packet_size: int
    radio: RadioParams
    queue_links: np.ndarray  # (Q, 2) registered next-hop queues, fixed for the episode
    queue_capacity: np.ndarray  # (Q,) packets
    waiting_capacity: np.ndarray  # (N,) packets
    waiting_max_wait: float

    @abstractmethod
    def positions(self, cycle: int) -> np.ndarray:
        """(N, 3) positions at the start of ``cycle`` (cycle may equal horizon)."""

    @abstractmethod
    def velocities(self, cycle: int) -> np.ndarray:
        """(N, 3) velocities during ``cycle``."""

    @abstractmethod
    def births(self, cycle: int) -> list[Birth]:
        """Packets born in ``(start, end]`` of ``cycle``, sorted by time."""

    def traffic_done(self, cycle: int) -> bool:
        """True when no packet will be born at or after ``cycle``."""
        return False

    def describe(self) -> dict:
        return {"num_nodes": self.num_nodes, "horizon": self.horizon,
                "cycle_length": self.cycle_length, "queues": int(len(self.queue_links))}


def all_pairs(num_nodes: int) -> np.ndarray:
    """Every ordered pair (u, v), u != v — declared up front, not from future routes."""
    return np.array([(u, v) for u in range(num_nodes) for v in range(num_nodes) if u != v],
                    dtype=np.int64).reshape(-1, 2)
