"""Candidate link rules (the action space of a cycle)."""

from __future__ import annotations

import numpy as np

from ..contracts import Report
from ..registry import slot
from .problem import SchedulingProblem

CANDIDATES = slot("candidates", "which links may be scheduled in a cycle")


def admissible_counts(report: Report, links: np.ndarray) -> np.ndarray:
    """For each link (u, v): queued packets of (u, v) the receiver could admit at decision time:
    destined to v, or their next queue at v has room (no route at v: room in v's waiting area).
    Arrivals within the cycle are ignored (decision-time view)."""
    links = np.asarray(links, dtype=np.int64).reshape(-1, 2)
    if len(links) == 0:
        return np.zeros(0)
    if report.queue_dst is None:
        raise ValueError("needs per-destination queue counts (Report.queue_dst)")
    n = report.num_nodes
    qmat = -np.ones((n, n), dtype=np.int64)
    qmat[report.queue_links[:, 0], report.queue_links[:, 1]] = np.arange(len(report.queue_links))
    free = report.queues.capacity - report.queues.packets
    wait_free = report.waiting_capacity - report.waiting_packets
    out = np.zeros(len(links))
    for i, (u, v) in enumerate(links):
        counts = report.queue_dst[qmat[u, v]]
        for d in np.nonzero(counts)[0]:
            nh = report.next_hop[v, d]
            ok = d == v or (free[qmat[v, nh]] > 0 if nh >= 0 else wait_free[v] > 0)
            out[i] += counts[d] if ok else 0
    return out


@CANDIDATES.register("standard", role="primary")
class StandardCandidates:
    """Physically available, route-allowed, non-empty queue, interference-free SNR >= margin*threshold.

    ``require_room`` (stage-0 diagnostic, default off): a link is a candidate only if its receiver
    could admit at least one of its packets now (``admissible_counts`` > 0), so every policy leaves
    links into full receiver queues idle."""

    def __init__(self, snr_margin_db: float = 0.0, require_room: bool = False):
        self.margin = 10.0 ** (snr_margin_db / 10.0)
        self.require_room = bool(require_room)

    def select(self, report: Report) -> np.ndarray:
        links = report.queue_links
        if len(links) == 0:
            return np.zeros((0, 2), dtype=np.int64)
        tx, rx = links[:, 0], links[:, 1]
        snr = report.tx_power[tx] * report.gain[tx, rx] / report.noise
        ok = ((report.queues.packets > 0) & report.route_next
              & (snr >= self.margin * report.threshold * (1 - 1e-9)))
        if self.require_room and ok.any():
            idx = np.nonzero(ok)[0]
            ok[idx] = admissible_counts(report, links[idx]) > 0
        # registered-queue order is (tx, rx) lexicographic → deterministic local order
        return links[ok].astype(np.int64)


def make_problem(report: Report, candidates: np.ndarray) -> SchedulingProblem:
    return SchedulingProblem(links=candidates, gain=report.gain, power=report.tx_power,
                             noise=report.noise, threshold=report.threshold)
