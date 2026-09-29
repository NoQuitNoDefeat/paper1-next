"""Policy contract.

A policy receives, per environment, the decision snapshot (report, scheduling
problem, a fresh micro-step controller and optionally the dual graph) and
drives the controller until no candidate is feasible.  The resulting ordered
selection is the complete cycle plan.  Policies act on a *batch* of
environments so learned policies can share one forward pass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..contracts import Report
from ..observation.graph import DualGraph
from ..registry import slot
from ..scheduling.controller import MicroStepController
from ..scheduling.problem import SchedulingProblem

POLICY = slot("policy", "decision snapshot -> complete cycle plan (via micro steps)")


@dataclass
class DecisionInput:
    report: Report
    problem: SchedulingProblem
    controller: MicroStepController
    graph: DualGraph | None = None


@dataclass
class DecisionOutput:
    actions: list[int]  # local candidate indices in selection order
    record: dict[str, Any] = field(default_factory=dict)  # learner data (masks, logp, values)


class Policy(ABC):
    needs_graph: bool = False
    learnable: bool = False
    # True: the policy drives the controller until no candidate is feasible (maximal plans,
    # the method's design).  A baseline that deliberately leaves capacity idle sets False.
    maximal_plans: bool = True
    # True: evaluation ("greedy" mode) still draws from the policy's randomness, so evaluation
    # reseeds it for reproducible results (baselines whose papers evaluate stochastically).
    stochastic_eval: bool = False

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def act(self, inputs: list[DecisionInput], *, mode: str = "sample") -> list[DecisionOutput]:
        """Drive every controller to completion.  ``mode``: 'sample' or 'greedy'."""
