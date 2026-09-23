"""Routing processes provided by the environment (not learned by the MAC)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from ..registry import slot

ROUTING = slot("routing", "next-hop tables computed by the environment")


class Routing(ABC):
    update_every: int = 1  # cycles between route recomputations

    @abstractmethod
    def compute(self, adjacency: np.ndarray, snr_over_threshold: np.ndarray | None = None) -> np.ndarray:
        """next_hop[u, d] from a boolean directed adjacency; -1 = no route, diag -1.

        ``snr_over_threshold`` (interference-free SNR / threshold) lets a routing
        rule prefer links with margin.
        """


def _min_hop(adjacency: np.ndarray) -> np.ndarray:
    n = len(adjacency)
    next_hop = np.full((n, n), -1, dtype=np.int64)
    for d in range(n):
        # BFS on reversed edges: dist[u] = hops from u to d
        dist = np.full(n, -1)
        dist[d] = 0
        frontier = deque([d])
        while frontier:
            v = frontier.popleft()
            for u in np.nonzero(adjacency[:, v])[0]:
                if dist[u] < 0:
                    dist[u] = dist[v] + 1
                    frontier.append(u)
        for u in range(n):
            if u == d or dist[u] <= 0:
                continue
            nbrs = np.nonzero(adjacency[u] & (dist == dist[u] - 1) & (dist >= 0))[0]
            if len(nbrs):
                next_hop[u, d] = int(nbrs.min())
    return next_hop


@ROUTING.register("min_hop", role="primary")
class MinHop(Routing):
    """Min-hop routes over links with SNR margin; pairs unreachable that way fall back to all links.

    Ties are broken by the smallest next-hop id.  Nodes with a margin route only
    forward through nodes that also have one, so mixing the two tables is loop-free.
    """

    def __init__(self, update_every: int = 25, snr_margin_db: float = 0.0):
        self.update_every = int(update_every)
        self.margin = 10.0 ** (snr_margin_db / 10.0)

    def compute(self, adjacency: np.ndarray, snr_over_threshold: np.ndarray | None = None) -> np.ndarray:
        if snr_over_threshold is None or self.margin <= 1.0:
            return _min_hop(adjacency)
        strong = adjacency & (snr_over_threshold >= self.margin)
        table = _min_hop(strong)
        fallback = _min_hop(adjacency)
        return np.where(table >= 0, table, fallback)
