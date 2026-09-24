"""Micro-step controller: owns the selected set, resources and feasibility mask.

It is independent of any model — policies (learned or heuristic) only read
``mask`` and call ``step``.  There is no STOP action: the episode of micro steps
ends when no candidate can be added, so the final set is maximal under the
configured constraints.
"""

from __future__ import annotations

import numpy as np

from .constraints import Tracker
from .problem import SchedulingProblem


class MicroStepController:
    def __init__(self, problem: SchedulingProblem, trackers: list[Tracker]):
        self.problem = problem
        self.trackers = trackers
        self.selected: list[int] = []
        self._chosen = np.zeros(problem.num_candidates, dtype=bool)
        self._mask = self._compute_mask()

    def _compute_mask(self) -> np.ndarray:
        mask = ~self._chosen
        for t in self.trackers:
            mask &= t.mask()
        return mask

    @property
    def mask(self) -> np.ndarray:
        return self._mask.copy()

    @property
    def done(self) -> bool:
        return not self._mask.any()

    def step(self, idx: int) -> None:
        idx = int(idx)
        if not (0 <= idx < self.problem.num_candidates) or not self._mask[idx]:
            raise ValueError(f"candidate {idx} is not feasible in the current micro state")
        for t in self.trackers:
            t.add(idx)
        self.selected.append(idx)
        self._chosen[idx] = True
        self._mask = self._compute_mask()

    def selected_links(self) -> list[tuple[int, int]]:
        return [tuple(int(x) for x in self.problem.links[i]) for i in self.selected]

    def micro_features(self) -> np.ndarray:
        """Controller-side state for the critic: selected / remaining counts."""
        c = max(self.problem.num_candidates, 1)
        n_sel = len(self.selected)
        n_feas = int(self._mask.sum())
        return np.array([n_sel / c, n_feas / c, float(n_feas == 0), np.log1p(n_sel)],
                        dtype=np.float32)


    def candidate_features(self) -> np.ndarray:
        """(C, 4) per-candidate micro-state features from decision-time estimates.

        0. own SINR margin under the selected set's interference, dB/10, clipped to [-1, 3];
        1. largest share of a selected link's remaining interference budget that the
           candidate would consume (0 when nothing is selected), clipped to [0, 2];
        2. share of the other currently feasible candidates it would exclude (shared
           node, or their own SINR would drop below threshold);
        3. mean interference it puts on the other feasible candidates, relative to
           their remaining budget, clipped to [0, 2].
        Computed from the problem and the selected set only (independent of the
        constraint implementation), so every policy sees the same definition.
        """
        p = self.problem
        c = p.num_candidates
        out = np.zeros((c, CANDIDATE_FEATURE_DIM), dtype=np.float32)
        if c == 0:
            return out
        sel = np.asarray(self.selected, dtype=np.int64)
        interf = p.cross[sel].sum(axis=0) if len(sel) else np.zeros(c)
        if len(sel):
            interf = interf - np.where(np.isin(np.arange(c), sel), np.diag(p.cross), 0.0)
        sinr = p.signal / (p.noise + interf)
        out[:, 0] = np.clip(10 * np.log10(np.maximum(sinr / p.threshold, 1e-6)) / 10, -1, 3)
        budget = np.maximum(p.signal / p.threshold - p.noise - interf, 1e-30)
        if len(sel):
            out[:, 1] = np.clip((p.cross[:, sel] / budget[sel]).max(axis=1), 0, 2)
        feas = self._mask
        others = feas.copy()
        n_other = max(int(feas.sum()) - 1, 1)
        links = p.links
        share = ((links[:, 0][:, None] == links[:, 0][None, :]) | (links[:, 0][:, None] == links[:, 1][None, :])
                 | (links[:, 1][:, None] == links[:, 0][None, :]) | (links[:, 1][:, None] == links[:, 1][None, :]))
        hurts = p.cross > budget[None, :]
        excl = (share | hurts) & others[None, :]
        np.fill_diagonal(excl, False)
        out[:, 2] = excl.sum(axis=1) / n_other
        rel = np.clip(p.cross / budget[None, :], 0, 2) * others[None, :]
        np.fill_diagonal(rel, 0.0)
        out[:, 3] = rel.sum(axis=1) / n_other
        out[~feas] = 0.0  # only selectable candidates carry dynamic information
        return out


MICRO_FEATURE_DIM = 4
CANDIDATE_FEATURE_DIM = 4
