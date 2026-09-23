"""Simple baselines: they use the same controller, so plans are maximal feasible sets."""

from __future__ import annotations

import numpy as np

from .base import POLICY, DecisionInput, DecisionOutput, Policy


def queue_view(inp: DecisionInput) -> tuple[np.ndarray, np.ndarray]:
    """Packets and HOL wait of each candidate's next-hop queue."""
    rep = inp.report
    n = rep.num_nodes
    qmat = -np.ones((n, n), dtype=np.int64)
    qmat[rep.queue_links[:, 0], rep.queue_links[:, 1]] = np.arange(len(rep.queue_links))
    links = inp.problem.links
    if len(links) == 0:
        return np.zeros(0), np.zeros(0)
    q = qmat[links[:, 0], links[:, 1]]
    return rep.queues.packets[q].astype(float), rep.queues.hol_wait[q]


class _ScoreGreedy(Policy):
    """Pick the feasible candidate with the highest static score until none is left."""

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def scores(self, inp: DecisionInput) -> np.ndarray:
        raise NotImplementedError

    def act(self, inputs, *, mode="sample"):
        outs = []
        for inp in inputs:
            score = self.scores(inp)
            ctl = inp.controller
            while not ctl.done:
                mask = ctl.mask
                idx = np.nonzero(mask)[0]
                ctl.step(int(idx[np.argmax(score[idx])]))  # argmax → lowest index on ties
            outs.append(DecisionOutput(actions=list(ctl.selected)))
        return outs


@POLICY.register("random", role="baseline")
class RandomFeasible(Policy):
    """Uniformly random feasible candidate at every micro step."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, inputs, *, mode="sample"):
        outs = []
        for inp in inputs:
            ctl = inp.controller
            while not ctl.done:
                ctl.step(int(self.rng.choice(np.nonzero(ctl.mask)[0])))
            outs.append(DecisionOutput(actions=list(ctl.selected)))
        return outs


@POLICY.register("longest_queue", role="baseline")
class LongestQueue(_ScoreGreedy):
    """Greedy by next-hop queue length (ties: older HOL). With a single rate this is max-weight greedy."""

    def scores(self, inp):
        packets, hol = queue_view(inp)
        return packets + 1e-3 * np.minimum(hol, 100.0)


@POLICY.register("oldest_hol", role="baseline")
class OldestHol(_ScoreGreedy):
    """Greedy by head-of-line waiting time (ties: longer queue)."""

    def scores(self, inp):
        packets, hol = queue_view(inp)
        return hol + 1e-6 * packets


@POLICY.register("hol_weighted", role="baseline")
class HolWeighted(_ScoreGreedy):
    """Greedy by queue length x (1 + HOL / ref): a delay-aware max-weight variant."""

    def __init__(self, hol_ref_s: float = 0.2):
        self.hol_ref = hol_ref_s

    def scores(self, inp):
        packets, hol = queue_view(inp)
        return packets * (1.0 + hol / self.hol_ref)
