"""Resource and interference constraints for the micro-step controller.

A constraint is a stateless component; ``tracker(problem)`` returns a per-cycle
object that keeps the selected set and answers "which candidates can still be
added so that the whole set stays feasible under this constraint".

The primary method uses ``half_duplex`` resources and ``full_sinr`` cumulative
interference.  ``pairwise``, ``new_link_only`` and ``none`` are explicitly
labelled controls: they are *different* constraints, not approximations of the
primary one, and executed sets may violate the full SINR condition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..registry import slot
from .problem import SchedulingProblem

RESOURCE = slot("resource", "per-node resource condition (radio / duplex)")
INTERFERENCE = slot("interference", "interference feasibility of the concurrent set")

# Relative tolerance so that a set sitting exactly on the threshold is feasible
# despite floating point noise.
REL_TOL = 1e-9


class Tracker(ABC):
    """Per-cycle incremental feasibility state."""

    @abstractmethod
    def mask(self) -> np.ndarray:
        """Boolean (C,) — candidates that may be added (selected ones may be True)."""

    @abstractmethod
    def add(self, idx: int) -> None:
        """Record that candidate ``idx`` joined the set."""


class Constraint(ABC):
    name: str = ""

    @abstractmethod
    def tracker(self, problem: SchedulingProblem) -> Tracker: ...


# ----------------------------------------------------------------- resources
class _HalfDuplexTracker(Tracker):
    def __init__(self, problem: SchedulingProblem):
        self.links = problem.links
        self.busy = np.zeros(problem.num_nodes, dtype=bool)

    def mask(self) -> np.ndarray:
        if len(self.links) == 0:
            return np.zeros(0, dtype=bool)
        return ~(self.busy[self.links[:, 0]] | self.busy[self.links[:, 1]])

    def add(self, idx: int) -> None:
        s, r = self.links[idx]
        self.busy[s] = True
        self.busy[r] = True


@RESOURCE.register("half_duplex", role="primary")
class HalfDuplex(Constraint):
    """Single radio, half duplex: a node joins at most one transmission (tx or rx)."""

    name = "half_duplex"

    def tracker(self, problem: SchedulingProblem) -> Tracker:
        return _HalfDuplexTracker(problem)


# -------------------------------------------------------------- interference
class _FullSinrTracker(Tracker):
    """Exact cumulative-SINR feasibility with O(C) incremental updates.

    ``interf[c]`` is the interference the selected set puts on candidate c's
    receiver.  A candidate c can join set S iff its own SINR holds and, for every
    selected m, the extra interference ``cross[c, m]`` fits in m's slack.
    """

    def __init__(self, problem: SchedulingProblem):
        self.p = problem
        c = problem.num_candidates
        self.interf = np.zeros(c)
        # max extra interference each *selected* link can still absorb; +inf means
        # "not selected yet", so it never constrains anyone.
        self.slack = np.full(c, np.inf)
        self.selected: list[int] = []

    def _own_ok(self) -> np.ndarray:
        p = self.p
        return p.signal >= p.threshold * (p.noise + self.interf) * (1 - REL_TOL)

    def mask(self) -> np.ndarray:
        p = self.p
        if p.num_candidates == 0:
            return np.zeros(0, dtype=bool)
        ok = self._own_ok()
        if self.selected:
            sel = np.asarray(self.selected)
            # cross[:, sel] — interference every candidate would add at each selected rx
            tol = REL_TOL * p.signal[sel] / p.threshold
            ok &= np.all(p.cross[:, sel] <= self.slack[sel] + tol, axis=1)
        return ok

    def add(self, idx: int) -> None:
        p = self.p
        self.interf += p.cross[idx]
        self.interf[idx] -= p.cross[idx, idx]  # a link does not interfere with itself
        self.selected.append(idx)
        sel = np.asarray(self.selected)
        self.slack[sel] = p.signal[sel] / p.threshold - p.noise - self.interf[sel]


@INTERFERENCE.register("full_sinr", role="primary")
class FullSinr(Constraint):
    """Every link of the set meets its SINR threshold under cumulative interference."""

    name = "full_sinr"

    def tracker(self, problem: SchedulingProblem) -> Tracker:
        return _FullSinrTracker(problem)


class _BidirectionalTracker(Tracker):
    """Full cumulative SINR in both phases of a DATA/ACK transaction.

    DATA phase: senders transmit, each receiver sees the other senders (the primary
    constraint).  ACK phase: with equal frame sizes and rates the transactions are
    aligned, so each sender receives its ACK while the other *receivers* send theirs.
    """

    def __init__(self, problem: SchedulingProblem):
        from .problem import SchedulingProblem as SP

        self.data = _FullSinrTracker(problem)
        s, r = problem.links[:, 0], problem.links[:, 1]
        reverse = SP.__new__(SP)  # same candidates, ACK signal r->s, interference r_a->s_b
        reverse.links, reverse.gain, reverse.power = problem.links, problem.gain, problem.power
        reverse.noise, reverse.threshold = problem.noise, problem.threshold
        reverse.cross = problem.power[r][:, None] * problem.gain[r][:, s]
        reverse.signal = (problem.power[r] * problem.gain[r, s]) if len(s) else np.zeros(0)
        np.fill_diagonal(reverse.cross, 0.0)
        reverse.cross[np.arange(len(s)), np.arange(len(s))] = reverse.signal
        self.ack = _FullSinrTracker(reverse)

    def mask(self) -> np.ndarray:
        return self.data.mask() & self.ack.mask()

    def add(self, idx: int) -> None:
        self.data.add(idx)
        self.ack.add(idx)


@INTERFERENCE.register("full_sinr_bidirectional", role="variant")
class FullSinrBidirectional(Constraint):
    """Variant: full cumulative SINR in the DATA phase (at receivers) and ACK phase (at senders)."""

    name = "full_sinr_bidirectional"

    def tracker(self, problem: SchedulingProblem) -> Tracker:
        return _BidirectionalTracker(problem)


class _NewLinkOnlyTracker(_FullSinrTracker):
    def mask(self) -> np.ndarray:
        if self.p.num_candidates == 0:
            return np.zeros(0, dtype=bool)
        return self._own_ok()


@INTERFERENCE.register("new_link_only", role="control")
class NewLinkOnly(Constraint):
    """Control: only the added link's cumulative SINR is checked, not the existing links'."""

    name = "new_link_only"

    def tracker(self, problem: SchedulingProblem) -> Tracker:
        return _NewLinkOnlyTracker(problem)


class _PairwiseTracker(Tracker):
    def __init__(self, problem: SchedulingProblem):
        p = problem
        noise_th = p.threshold * p.noise
        alone = p.signal >= noise_th * (1 - REL_TOL)
        # pair_ok[a, b]: a and b both meet the threshold when only the two transmit
        a_ok = p.signal[:, None] >= p.threshold * (p.noise + p.cross.T) * (1 - REL_TOL)
        self.pair_ok = a_ok & a_ok.T & alone[:, None] & alone[None, :]
        self.alone = alone
        self.selected: list[int] = []

    def mask(self) -> np.ndarray:
        if len(self.alone) == 0:
            return np.zeros(0, dtype=bool)
        ok = self.alone.copy()
        if self.selected:
            ok &= np.all(self.pair_ok[:, self.selected], axis=1)
        return ok

    def add(self, idx: int) -> None:
        self.selected.append(idx)


@INTERFERENCE.register("pairwise", role="control")
class Pairwise(Constraint):
    """Control: every pair is compatible in isolation; cumulative interference ignored."""

    name = "pairwise"

    def tracker(self, problem: SchedulingProblem) -> Tracker:
        return _PairwiseTracker(problem)


class _NoInterferenceTracker(Tracker):
    def __init__(self, problem: SchedulingProblem):
        self.n = problem.num_candidates

    def mask(self) -> np.ndarray:
        return np.ones(self.n, dtype=bool)

    def add(self, idx: int) -> None:
        pass


@INTERFERENCE.register("none", role="control")
class NoInterference(Constraint):
    """Control: resource conflicts only; same channel is never a conflict by itself."""

    name = "none"

    def tracker(self, problem: SchedulingProblem) -> Tracker:
        return _NoInterferenceTracker(problem)


# ------------------------------------------------------------------ assembly
class ConstraintSet:
    """Resource + interference constraints used together by the controller."""

    def __init__(self, resource: Constraint, interference: Constraint):
        self.resource = resource
        self.interference = interference

    @classmethod
    def from_config(cls, cfg: dict) -> "ConstraintSet":
        return cls(RESOURCE.build(cfg.get("resource", "half_duplex")),
                   INTERFERENCE.build(cfg.get("interference", "full_sinr")))

    @property
    def label(self) -> str:
        return f"{self.resource.name}+{self.interference.name}"

    @property
    def is_primary(self) -> bool:
        return (getattr(self.resource, "registered_role", "") == "primary"
                and getattr(self.interference, "registered_role", "") == "primary")

    def start(self, problem: SchedulingProblem):
        from .controller import MicroStepController

        return MicroStepController(problem, [self.resource.tracker(problem),
                                             self.interference.tracker(problem)])


def full_sinr_violations(links: np.ndarray, gain: np.ndarray, power: np.ndarray,
                         noise: float, threshold: float) -> int:
    """Number of links of a set that miss the threshold under cumulative interference."""
    from ..physics import set_sinr

    if len(links) == 0:
        return 0
    return int(np.sum(set_sinr(links, gain, power, noise) < threshold * (1 - REL_TOL)))
