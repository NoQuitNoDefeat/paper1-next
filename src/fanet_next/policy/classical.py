"""Classical scheduling baselines: exact max-weight / backpressure, local search, spatial TDMA.

All of them decide from the same decision-time report as every other policy and
replay their set through the controller, so every plan is feasible under the
configured constraints.  The optimisation itself uses half duplex + full
cumulative SINR (``scheduling.maxweight``), the primary constraint set.
"""

from __future__ import annotations

import numpy as np

from ..scheduling.maxweight import (_Set, greedy_complete, local_search, max_weight_set,
                                    pairwise_conflicts)
from ..scheduling.problem import SchedulingProblem
from .base import POLICY, DecisionOutput, Policy
from ..scheduling.candidates import admissible_counts
from .heuristics import _ScoreGreedy, queue_view

WEIGHTS = ("queue", "backpressure")


def queue_weights(inp) -> np.ndarray:
    """Max-weight with a single rate: next-hop queue length (ties: older HOL), as longest_queue."""
    packets, hol = queue_view(inp)
    return packets + 1e-3 * np.minimum(hol, 100.0)


def _queue_index(rep) -> np.ndarray:
    n = rep.num_nodes
    qmat = -np.ones((n, n), dtype=np.int64)
    qmat[rep.queue_links[:, 0], rep.queue_links[:, 1]] = np.arange(len(rep.queue_links))
    return qmat


def backpressure_differential(inp) -> np.ndarray:
    """For candidate (u, v): the largest commodity backlog differential Q_u^d - Q_v^d over the
    destinations d present in queue (u, v) (Q_d^d = 0), not floored.  Q_x^d counts x's queued
    packets destined to d."""
    rep, links = inp.report, inp.problem.links
    if len(links) == 0:
        return np.zeros(0)
    if rep.queue_dst is None:
        raise ValueError("backpressure needs per-destination queue counts (Report.queue_dst)")
    n = rep.num_nodes
    node_dst = np.zeros((n, n))
    np.add.at(node_dst, rep.queue_links[:, 0], rep.queue_dst)
    q = _queue_index(rep)[links[:, 0], links[:, 1]]
    present = rep.queue_dst[q] > 0  # (C, N)
    diff = node_dst[links[:, 0]] - node_dst[links[:, 1]]  # (C, N)
    return np.where(present, diff, -np.inf).max(axis=1)


def backpressure_weights(inp) -> np.ndarray:
    """Backpressure with fixed routes (Tassiulas-Ephremides): the commodity differential of
    ``backpressure_differential`` floored at 0, ties broken by the longer queue."""
    if len(inp.problem.links) == 0:
        return np.zeros(0)
    packets, _ = queue_view(inp)
    return np.maximum(backpressure_differential(inp), 0.0) + 1e-3 * packets  # ties: longer queue


def admissible_packets(inp) -> np.ndarray:
    """For candidate (u, v): queued packets the receiver could admit at decision time
    (``scheduling.candidates.admissible_counts``)."""
    return admissible_counts(inp.report, inp.problem.links)


def last_hop_share(inp) -> np.ndarray:
    """For candidate (u, v): share of the queue's packets destined to the receiver v."""
    rep, links = inp.report, inp.problem.links
    if len(links) == 0:
        return np.zeros(0)
    q = _queue_index(rep)[links[:, 0], links[:, 1]]
    counts = rep.queue_dst[q]
    return counts[np.arange(len(links)), links[:, 1]] / np.maximum(counts.sum(axis=1), 1)


def _greedy_allowed(inp, score: np.ndarray, allowed: np.ndarray) -> list[int]:
    """Highest score first among the allowed feasible candidates; stops when none is left, so
    allowed-out candidates stay idle even when feasible."""
    ctl = inp.controller
    while True:
        idx = np.nonzero(ctl.mask & allowed)[0]
        if len(idx) == 0:
            return list(ctl.selected)
        ctl.step(int(idx[np.argmax(score[idx])]))


def _weights(kind: str, inp) -> np.ndarray:
    if kind not in WEIGHTS:
        raise ValueError(f"weights must be one of {WEIGHTS}")
    return queue_weights(inp) if kind == "queue" else backpressure_weights(inp)


def _replay(inp, first: list[int], weights: np.ndarray, *, complete: bool = True) -> list[int]:
    """Feed a feasible set to the controller (heaviest first), then, if ``complete``, add the
    remaining feasible candidates by weight so the plan is maximal like every other policy's."""
    ctl = inp.controller
    for i in sorted(first, key=lambda i: -weights[i]):
        if ctl.mask[i]:
            ctl.step(i)
    while complete and not ctl.done:
        idx = np.nonzero(ctl.mask)[0]
        ctl.step(int(idx[np.argmax(weights[idx])]))
    return list(ctl.selected)


@POLICY.register("max_weight_opt", role="baseline")
class MaxWeightOptimal(Policy):
    """Per-cycle exact max-weight (MILP, HiGHS) under half duplex + full SINR; weights 'queue'
    (classical max-weight) or 'backpressure'.  Records solver status and time."""

    def __init__(self, weights: str = "queue", time_limit_s: float = 10.0):
        if weights not in WEIGHTS:
            raise ValueError(f"weights must be one of {WEIGHTS}")
        self.kind, self.time_limit = weights, float(time_limit_s)

    def act(self, inputs, *, mode="sample"):
        outs = []
        for inp in inputs:
            w = _weights(self.kind, inp)
            sel, info = max_weight_set(inp.problem, w, time_limit=self.time_limit)
            outs.append(DecisionOutput(actions=_replay(inp, sel, w),
                                       record={"solver": info.status, "solve_s": info.seconds}))
        return outs


@POLICY.register("backpressure_opt", role="baseline")
class BackpressureOptimal(MaxWeightOptimal):
    """Per-cycle exact backpressure (MILP): max_weight_opt with backpressure weights."""

    def __init__(self, time_limit_s: float = 10.0):
        super().__init__(weights="backpressure", time_limit_s=time_limit_s)


@POLICY.register("backpressure", role="baseline")
class Backpressure(_ScoreGreedy):
    """Greedy backpressure: feasible candidate with the largest commodity differential first."""

    def scores(self, inp):
        return backpressure_weights(inp)


@POLICY.register("backpressure_hold", role="control")
class BackpressureHold(Policy):
    """Diagnostic control (D4): greedy backpressure as in the classical algorithm, i.e. only
    links with a positive commodity differential transmit; no completion to a maximal set."""

    maximal_plans = False

    def act(self, inputs, *, mode="sample"):
        return [DecisionOutput(actions=_greedy_allowed(inp, backpressure_weights(inp),
                                                       backpressure_differential(inp) > 0))
                for inp in inputs]


@POLICY.register("lq_hold", role="control")
class LongestQueueHold(Policy):
    """Diagnostic control (D4): longest_queue that leaves a link idle when its receiver could
    admit none of the link's queued packets (every next queue at the receiver is full)."""

    maximal_plans = False

    def act(self, inputs, *, mode="sample"):
        return [DecisionOutput(actions=_greedy_allowed(inp, queue_weights(inp), admissible_packets(inp) > 0))
                for inp in inputs]


@POLICY.register("lq_lasthop", role="control")
class LongestQueueLastHop(_ScoreGreedy):
    """Stage-0 diagnostic baseline: longest_queue weighted up by the share of the queue's packets
    destined to the receiver, packets * (1 + beta * share) (ties: older HOL); maximal plans."""

    def __init__(self, beta: float = 1.0):
        self.beta = float(beta)

    def scores(self, inp):
        packets, hol = queue_view(inp)
        return packets * (1.0 + self.beta * last_hop_share(inp)) + 1e-3 * np.minimum(hol, 100.0)


@POLICY.register("lq_local_search", role="baseline")
class LongestQueueLocalSearch(Policy):
    """longest_queue greedy improved by swap local search on the same max-weight objective."""

    def __init__(self, rounds: int = 3, attempts: int = 20):
        self.rounds, self.attempts = int(rounds), int(attempts)

    def act(self, inputs, *, mode="sample"):
        outs = []
        for inp in inputs:
            w = queue_weights(inp)
            start = greedy_complete(inp.problem, w)
            sel = local_search(inp.problem, w, start, rounds=self.rounds, attempts=self.attempts)
            outs.append(DecisionOutput(actions=_replay(inp, sel, w)))
        return outs


@POLICY.register("spatial_tdma", role="baseline")
class SpatialTdma(Policy):
    """Spatial TDMA with demand-based slot reservation: the links that had traffic during the
    last frame period (``links="active"``: any cycle with a non-empty queue on a current
    route; ``"routed"``: every link on a current route) are colored into compatible
    slots (half duplex + full SINR, most-constrained first, first fit); with ``fill_slots``
    every slot is then filled maximally with further compatible routed links (a link may
    own several slots, the usual STDMA refinement).  Cycle k serves slot k mod K.  The frame
    is rebuilt every ``rebuild_every`` cycles from that cycle's report (topology changes in
    between are only caught by the controller's mask).  Only slot members with packets
    transmit: no queue-aware, work-conserving fill."""

    maximal_plans = False

    def __init__(self, rebuild_every: int = 25, fill_slots: bool = True, links: str = "active"):
        if links not in ("active", "routed"):
            raise ValueError("links must be 'active' or 'routed'")
        self.rebuild_every = int(rebuild_every)
        self.fill_slots = bool(fill_slots)
        self.links = links
        self._active: dict[str, set] = {}  # run_id -> links seen with traffic since the rebuild
        self._frames: dict[str, tuple[int, int, list[set]]] = {}  # run_id -> (episode, start, frame)

    def _frame(self, rep, active: set | None = None) -> list[set]:
        n = rep.num_nodes
        routed = {(u, int(rep.next_hop[u, d])) for u in range(n) for d in range(n)
                  if u != d and rep.next_hop[u, d] >= 0}
        if self.links == "active":
            routed &= active or set()
        links = np.array(sorted(routed), dtype=np.int64).reshape(-1, 2)
        if len(links) == 0:
            return [set()]
        prob = SchedulingProblem(links=links, gain=rep.gain, power=rep.tx_power,
                                 noise=rep.noise, threshold=rep.threshold)
        degree = pairwise_conflicts(prob).sum(axis=1)  # most constrained first
        slots: list[_Set] = []
        for i in np.argsort(-degree, kind="stable"):
            for st in slots:
                if st.can_add(int(i)):
                    st.add(int(i))
                    break
            else:
                slots.append(_Set(prob, [int(i)]))
        if self.fill_slots:
            for st in slots:
                for i in np.argsort(degree, kind="stable"):  # least constrained first
                    if st.can_add(int(i)):
                        st.add(int(i))
        return [{tuple(int(x) for x in links[m]) for m in st.members} for st in slots]

    def act(self, inputs, *, mode="sample"):
        outs = []
        for inp in inputs:
            rep = inp.report
            built = self._frames.get(rep.run_id)
            fresh = built is None or built[0] != rep.episode or rep.cycle < built[1]
            active = set() if fresh else self._active.get(rep.run_id, set())
            active |= {(int(a), int(b)) for a, b in inp.problem.links}  # non-empty, routed
            idle = not fresh and not any(built[2]) and inp.problem.num_candidates > 0
            if fresh or idle or rep.cycle - built[1] >= self.rebuild_every:
                built = (rep.episode, rep.cycle, self._frame(rep, active))
                self._frames[rep.run_id] = built
                active = {(int(a), int(b)) for a, b in inp.problem.links}
            self._active[rep.run_id] = active
            _, start, frame = built
            slot = frame[(rep.cycle - start) % len(frame)]
            ctl = inp.controller
            for i, (s, r) in enumerate(inp.problem.links):
                if (int(s), int(r)) in slot and ctl.mask[i]:
                    ctl.step(i)
            outs.append(DecisionOutput(actions=list(ctl.selected)))
        return outs
