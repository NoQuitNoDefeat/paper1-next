"""Candidate link rules (the action space of a cycle)."""

from __future__ import annotations

import numpy as np

from ..contracts import Report
from ..registry import slot
from .problem import SchedulingProblem

CANDIDATES = slot("candidates", "which links may be scheduled in a cycle")


@CANDIDATES.register("standard", role="primary")
class StandardCandidates:
    """Physically available, route-allowed, non-empty queue, interference-free SNR >= margin*threshold."""

    def __init__(self, snr_margin_db: float = 0.0):
        self.margin = 10.0 ** (snr_margin_db / 10.0)

    def select(self, report: Report) -> np.ndarray:
        links = report.queue_links
        if len(links) == 0:
            return np.zeros((0, 2), dtype=np.int64)
        tx, rx = links[:, 0], links[:, 1]
        snr = report.tx_power[tx] * report.gain[tx, rx] / report.noise
        ok = ((report.queues.packets > 0) & report.route_next
              & (snr >= self.margin * report.threshold * (1 - 1e-9)))
        # registered-queue order is (tx, rx) lexicographic → deterministic local order
        return links[ok].astype(np.int64)


def make_problem(report: Report, candidates: np.ndarray) -> SchedulingProblem:
    return SchedulingProblem(links=candidates, gain=report.gain, power=report.tx_power,
                             noise=report.noise, threshold=report.threshold)
