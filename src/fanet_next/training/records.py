"""Per-cycle training record: the micro record plus the boundary transition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model.runner import MicroRecord

# Stream end markers of the boundary transition
CONTINUE = "continue"  # next record of the same stream is the next state
TRUNCATED = "truncated"  # episode time limit: bootstrap with V(last valid state)
CUT = "cut"  # rollout slice ends here: bootstrap, stop recursion
TERMINATED = "terminated"  # task end: no future value
ENDS = (CONTINUE, TRUNCATED, CUT, TERMINATED)


@dataclass
class CycleRecord:
    micro: MicroRecord
    reward: float  # reward used for learning (possibly scaled)
    raw_reward: float
    end: str
    bootstrap: float = 0.0  # V_old of the state after this cycle when end in {truncated, cut}
    env: int = 0
    episode: int = 0
    cycle: int = 0
    advantages: np.ndarray | None = None  # (k+1,)
    returns: np.ndarray | None = None  # (k+1,)

    @property
    def num_actions(self) -> int:
        return self.micro.num_actions
