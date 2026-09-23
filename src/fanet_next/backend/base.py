"""Backend contract.

A backend owns the authoritative packet state of one run: it executes a
complete cycle plan and returns execution facts plus the next report.  Python
and ns-3 backends implement the same contract; exactly one of them advances
the packets of a given run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..contracts import Plan, Report, StepOutcome
from ..registry import slot
from ..scenario.base import Scenario

BACKEND = slot("backend", "executes complete plans and advances packet state")


class Backend(ABC):
    #: backend can snapshot/restore its full state (exact resume of running episodes)
    supports_state: bool = False

    @abstractmethod
    def reset(self, scenario: Scenario, *, run_id: str, episode: int, seed: int) -> Report:
        """Start an episode and return the first decision report."""

    @abstractmethod
    def execute(self, plan: Plan) -> StepOutcome:
        """Execute one full cycle plan; raise ``ExecutionError`` on failure."""

    def close(self) -> None:
        pass
