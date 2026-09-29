"""Per-cycle weighted scheduling under half duplex + full cumulative SINR (baseline tools).

These routines work on a :class:`SchedulingProblem` directly (no micro steps):
an exact MILP (``max_weight_set``), a feasibility check with the same tolerance
as the controller, greedy completion and a swap local search.  A set feasible
here is feasible under every configured interference rule (pairwise / none are
looser), so policies can replay it through any controller.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from .constraints import REL_TOL
from .problem import SchedulingProblem

# The MILP keeps a small relative margin under the SINR threshold so that solver
# tolerances never produce a set the controller (REL_TOL = 1e-9) would reject.
MILP_MARGIN = 1e-6


@dataclass
class SolveInfo:
    status: str  # "optimal" | "time_limit" | "empty"
    objective: float
    bound: float  # best dual bound (== objective when optimal)
    seconds: float


def node_conflicts(problem: SchedulingProblem) -> np.ndarray:
    """(C, C) bool: two candidates share an endpoint (half duplex, single radio)."""
    s, r = problem.links[:, 0], problem.links[:, 1]
    share = ((s[:, None] == s[None, :]) | (s[:, None] == r[None, :])
             | (r[:, None] == s[None, :]) | (r[:, None] == r[None, :]))
    np.fill_diagonal(share, False)
    return share


def pairwise_conflicts(problem: SchedulingProblem) -> np.ndarray:
    """(C, C) bool conflict graph: shared endpoint, or the pair fails its SINR on its own
    (the binary conflict model of graph-based schedulers; cumulative effects are not in it)."""
    own = problem.signal[:, None] >= problem.threshold * (problem.noise + problem.cross.T) * (1 - REL_TOL)
    out = node_conflicts(problem) | ~(own & own.T)
    np.fill_diagonal(out, False)
    return out


def feasible(problem: SchedulingProblem, idx) -> bool:
    """Half duplex and every member's cumulative SINR, with the controller's tolerance."""
    idx = np.asarray(idx, dtype=np.int64)
    if len(idx) == 0:
        return True
    nodes = problem.links[idx].ravel()
    if len(np.unique(nodes)) != len(nodes):
        return False
    interf = problem.cross[np.ix_(idx, idx)].sum(axis=0) - problem.signal[idx]
    return bool(np.all(problem.signal[idx]
                       >= problem.threshold * (problem.noise + interf) * (1 - REL_TOL)))


class _Set:
    """Incremental feasible set: interference at every candidate's receiver and the slack
    of every member, so a candidate can be tested in O(|S|)."""

    def __init__(self, problem: SchedulingProblem, members=()):
        self.p = problem
        self.members: list[int] = []
        self.used = np.zeros(problem.num_nodes, dtype=bool)
        self.interf = np.zeros(problem.num_candidates)
        for i in members:
            self.add(int(i))

    def can_add(self, c: int) -> bool:
        p = self.p
        s, r = p.links[c]
        if self.used[s] or self.used[r] or c in self.members:
            return False
        if p.signal[c] < p.threshold * (p.noise + self.interf[c]) * (1 - REL_TOL):
            return False
        if self.members:
            m = np.asarray(self.members)
            tot = self.interf[m] + p.cross[c, m]
            if np.any(p.signal[m] < p.threshold * (p.noise + tot) * (1 - REL_TOL)):
                return False
        return True

    def add(self, c: int) -> None:
        s, r = self.p.links[c]
        self.used[s] = self.used[r] = True
        self.interf += self.p.cross[c]
        self.interf[c] -= self.p.cross[c, c]
        self.members.append(c)


def greedy_complete(problem: SchedulingProblem, weights: np.ndarray, start=()) -> list[int]:
    """Add candidates in descending weight (ties: lower index) while feasible; ``start`` first."""
    st = _Set(problem, start)
    for c in np.argsort(-weights, kind="stable"):
        if st.can_add(int(c)):
            st.add(int(c))
    return st.members


def max_weight_set(problem: SchedulingProblem, weights: np.ndarray, *, time_limit: float = 10.0,
                   rel_gap: float = 1e-6) -> tuple[list[int], SolveInfo]:
    """Exact max sum(weights) over half-duplex, full-SINR feasible candidate sets (MILP).

    SINR of l as a big-M row in interference-to-noise units:
        sum_{k != l} (I_kl / N0) x_k + M_l x_l <= (S_l / (theta N0) - 1)(1 - margin) + M_l,
    with M_l = sum_{k != l} I_kl / N0 (the most interference l could see).
    """
    t0 = time.perf_counter()
    p, c = problem, problem.num_candidates
    if c == 0:
        return [], SolveInfo("empty", 0.0, 0.0, 0.0)
    w = np.asarray(weights, dtype=float)
    inr = p.cross.T / p.noise  # [l, k]: k's transmitter at l's receiver
    np.fill_diagonal(inr, 0.0)
    rhs0 = np.maximum(p.signal / (p.threshold * p.noise) - 1.0, 0.0) * (1 - MILP_MARGIN)
    big_m = inr.sum(axis=1)
    rows, ub = [], []
    for l in np.nonzero(big_m > rhs0)[0]:  # other rows can never bind
        row = inr[l].copy()
        row[l] = big_m[l]
        rows.append(row)
        ub.append(rhs0[l] + big_m[l])
    n = p.num_nodes
    inc = np.zeros((n, c))
    inc[p.links[:, 0], np.arange(c)] = 1.0
    inc[p.links[:, 1], np.arange(c)] = 1.0
    inc = inc[inc.sum(axis=1) > 1]  # nodes shared by at least two candidates
    a = np.vstack([np.asarray(rows).reshape(-1, c), inc])
    b = np.concatenate([np.asarray(ub, dtype=float), np.ones(len(inc))])
    res = milp(c=-w, constraints=[LinearConstraint(csr_matrix(a), -np.inf, b)],
               integrality=np.ones(c), bounds=Bounds(0, 1),
               options={"time_limit": float(time_limit), "mip_rel_gap": rel_gap, "disp": False})
    if res.x is None:  # no incumbent within the time limit: fall back to greedy
        sel = greedy_complete(p, w)
        return sel, SolveInfo("time_limit", float(w[sel].sum()), float("inf"),
                              time.perf_counter() - t0)
    sel = [int(i) for i in np.nonzero(res.x > 0.5)[0]]
    # replay with the controller's exact check; a solver-tolerance violator is dropped
    st = _Set(p)
    for i in sorted(sel, key=lambda i: -w[i]):
        if st.can_add(i):
            st.add(i)
    status = "optimal" if res.status == 0 else "time_limit"
    bound = -getattr(res, "mip_dual_bound", -res.fun) if res.status != 0 else -res.fun
    return st.members, SolveInfo(status, float(w[st.members].sum()), float(bound),
                                 time.perf_counter() - t0)


def local_search(problem: SchedulingProblem, weights: np.ndarray, start: list[int], *,
                 rounds: int = 3, attempts: int = 20) -> list[int]:
    """Swap local search from ``start``: force in a heavier blocked candidate, drop what blocks
    it (endpoint conflicts, then the strongest interferers / the lightest violated members),
    refill greedily, keep the change if the total weight grows."""
    p, w = problem, np.asarray(weights, dtype=float)
    cur = list(start)
    cur_w = float(w[cur].sum()) if cur else 0.0
    conflict = node_conflicts(p)
    for _ in range(rounds):
        improved = False
        outside = [i for i in np.argsort(-w, kind="stable") if i not in set(cur) and w[i] > 0]
        for cand in outside[:attempts]:
            cand = int(cand)
            keep = [m for m in cur if not conflict[cand, m]]
            trial = keep + [cand]
            while not feasible(p, trial):
                others = [m for m in trial if m != cand]
                if not others:
                    break
                own_i = p.cross[others, cand].sum()
                if p.signal[cand] < p.threshold * (p.noise + own_i) * (1 - REL_TOL):
                    drop = others[int(np.argmax(p.cross[others, cand]))]
                else:
                    idx = np.asarray(trial)
                    interf = p.cross[np.ix_(idx, idx)].sum(axis=0) - p.signal[idx]
                    bad = [int(m) for m, ok in zip(idx, p.signal[idx] >= p.threshold * (
                        p.noise + interf) * (1 - REL_TOL)) if not ok and m != cand]
                    drop = min(bad, key=lambda m: w[m]) if bad else others[0]
                trial.remove(drop)
            if not feasible(p, trial):
                continue
            trial = greedy_complete(p, w, trial)
            tw = float(w[trial].sum())
            if tw > cur_w + 1e-9:
                cur, cur_w, improved = trial, tw, True
        if not improved:
            break
    return cur
