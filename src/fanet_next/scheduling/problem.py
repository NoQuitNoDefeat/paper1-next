"""The per-cycle scheduling problem seen by constraints and the controller."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SchedulingProblem:
    """Candidate links of one decision snapshot plus decision-time channel estimates.

    ``links[i] = (s, r)`` is the stable directed physical link behind local
    candidate ``i``.  Gains are *estimates* available at decision time; the
    backend decides actual success with its own channel.
    """

    links: np.ndarray  # (C, 2) int
    gain: np.ndarray  # (N, N) linear, gain[s, r]
    power: np.ndarray  # (N,) W
    noise: float  # W
    threshold: float  # linear SINR threshold
    signal: np.ndarray = field(init=False)  # (C,)
    cross: np.ndarray = field(init=False)  # (C, C): tx of a received at rx of b

    def __post_init__(self) -> None:
        self.links = np.asarray(self.links, dtype=np.int64).reshape(-1, 2)
        tx, rx = self.links[:, 0], self.links[:, 1]
        self.cross = self.power[tx][:, None] * self.gain[tx][:, rx]
        self.signal = np.diag(self.cross).copy() if len(tx) else np.zeros(0)

    @property
    def num_candidates(self) -> int:
        return len(self.links)

    @property
    def num_nodes(self) -> int:
        return len(self.power)
