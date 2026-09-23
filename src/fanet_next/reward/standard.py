"""Cycle reward from actual execution facts (research-method §6).

``r = alpha * U - beta * Q - eta * D - lambda_viol * V`` with, over the fixed
set of registered next-hop queues (including empty, unselected and
unschedulable ones; the waiting area is accounted separately):

* ``U`` service: mean of actually dequeued bytes / reference service bytes;
* ``Q`` remaining queue: mean post-service occupancy / capacity;
* ``D`` delay pressure: mean of occupancy/capacity * min(HOL / hol_ref, 1);
* ``V`` termination violation: unique terminated packets / risk packets, where
  risk = queued + waiting at cycle start + packets born in the cycle.

With no registered queues U = Q = D = 0; V can still be non-zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..contracts import CycleFacts
from ..registry import slot

REWARD = slot("reward", "cycle reward computed from execution facts")


@dataclass
class RewardBreakdown:
    total: float
    parts: dict[str, float] = field(default_factory=dict)  # unweighted components


def reward_components(facts: CycleFacts, hol_ref_s: float,
                      violation_ref: float | None = None) -> dict[str, float]:
    """Unweighted components; ``violation_ref=None`` normalises terminations by risk packets."""
    post = facts.post_service
    n_q = len(post.packets)
    if n_q:
        service = float(np.mean(facts.served_bytes / max(facts.service_ref_bytes, 1)))
        occ = post.packets / np.maximum(post.capacity, 1)
        queue = float(np.mean(occ))
        delay = float(np.mean(occ * np.minimum(post.hol_wait / hol_ref_s, 1.0)))
    else:
        service = queue = delay = 0.0
    terminated = len(set(facts.terminated_ids))
    denom = max(facts.risk_packets, 1) if violation_ref is None else violation_ref
    violation = terminated / denom
    return {"service": service, "queue": queue, "delay": delay, "violation": violation}


@REWARD.register("standard", role="primary")
class StandardReward:
    """alpha*service - beta*queue - eta*delay - lambda_viol*violation (all from facts)."""

    def __init__(self, alpha: float = 1.0, beta: float = 0.5, eta: float = 0.5,
                 lambda_viol: float = 1.0, hol_ref_s: float = 0.2, scale: float = 1.0):
        self.alpha, self.beta, self.eta, self.lambda_viol = alpha, beta, eta, lambda_viol
        self.hol_ref_s = hol_ref_s
        self.scale = scale

    def weights(self) -> dict[str, float]:
        return {"service": self.alpha, "queue": -self.beta, "delay": -self.eta,
                "violation": -self.lambda_viol}

    def __call__(self, facts: CycleFacts) -> RewardBreakdown:
        parts = reward_components(facts, self.hol_ref_s)
        total = self.scale * sum(w * parts[k] for k, w in self.weights().items())
        return RewardBreakdown(total=float(total), parts=parts)


@REWARD.register("fixed_violation_ref", role="control")
class FixedViolationRefReward(StandardReward):
    """Control (plan B): violation = terminated / fixed reference instead of / risk packets.

    Under congestion the risk count grows, which dilutes the per-drop penalty of
    the primary reward; a fixed reference keeps it constant.  This changes the
    meaning of the violation term and is only a labelled control.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5, eta: float = 0.5,
                 lambda_viol: float = 1.0, hol_ref_s: float = 0.2, scale: float = 1.0,
                 violation_ref_packets: float = 300.0):
        super().__init__(alpha, beta, eta, lambda_viol, hol_ref_s, scale)
        self.violation_ref = float(violation_ref_packets)

    def __call__(self, facts: CycleFacts) -> RewardBreakdown:
        parts = reward_components(facts, self.hol_ref_s, self.violation_ref)
        total = self.scale * sum(w * parts[k] for k, w in self.weights().items())
        return RewardBreakdown(total=float(total), parts=parts)
