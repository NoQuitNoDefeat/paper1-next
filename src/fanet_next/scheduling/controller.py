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


MICRO_FEATURE_DIM = 4
