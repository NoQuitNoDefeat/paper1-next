"""Evaluation metrics, kept separate from the reward.

One-hop service counts relay hops and HOL is only a delay proxy, so final
delivery, end-to-end delay, backlog and termination causes are recorded here
independently.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from ..contracts import CycleFacts
from .standard import RewardBreakdown


ONTIME_DEADLINES_S = (0.5, 1.0, 2.0)


class MetricsAccumulator:
    def __init__(self) -> None:
        self.cycles = 0
        self.duration = 0.0
        self.born = 0
        self.delivered = 0
        self.delivered_bytes = 0
        self.delays: list[float] = []
        self.terminations: Counter = Counter()
        self.served_packets = 0
        self.planned_links = 0
        self.failed_links = 0
        self.sinr_violation_cycles = 0
        self.sinr_violation_links = 0
        self.queue_backlog = 0.0
        self.waiting_backlog = 0.0
        self.candidates = 0
        self.rewards: list[float] = []
        self.reward_parts: Counter = Counter()
        self.decision_seconds = 0.0
        self.empty_candidate_cycles = 0
        self.rehomed = 0
        self.relay_terminated = 0
        self.in_system_end = 0

    def add(self, facts: CycleFacts, reward: RewardBreakdown, *, n_candidates: int,
            decision_seconds: float, planned_sinr_violations: int) -> None:
        self.cycles += 1
        self.duration += facts.t_end - facts.t_start
        self.born += facts.births
        self.delivered += len(facts.delivered_ids)
        self.delivered_bytes += facts.delivered_bytes
        self.delays.extend(facts.delivered_delays)
        for reason, ids in facts.terminations.items():
            self.terminations[reason] += len(ids)
        self.served_packets += int(facts.served_packets.sum())
        self.planned_links += len(facts.planned)
        self.failed_links += len(facts.failed)
        self.sinr_violation_links += planned_sinr_violations
        self.sinr_violation_cycles += int(planned_sinr_violations > 0)
        self.queue_backlog += int(facts.post_service.packets.sum())
        self.waiting_backlog += facts.waiting_post_service
        self.candidates += n_candidates
        self.empty_candidate_cycles += int(n_candidates == 0)
        self.rewards.append(reward.total)
        self.reward_parts.update(reward.parts)
        self.decision_seconds += decision_seconds
        self.rehomed += facts.rehomed
        self.relay_terminated += facts.relay_terminated
        self.in_system_end = facts.queued_end + facts.waiting_end

    def summary(self) -> dict[str, float]:
        c = max(self.cycles, 1)
        d = np.asarray(self.delays) if self.delays else np.zeros(0)
        terminated = sum(self.terminations.values())
        out = {
            "cycles": self.cycles,
            "born": self.born,
            "delivered": self.delivered,
            "delivery_ratio": self.delivered / max(self.born, 1),
            "terminated": terminated,
            "termination_ratio": terminated / max(self.born, 1),
            "in_system_end": self.in_system_end,
            "throughput_bps": 8.0 * self.delivered_bytes / max(self.duration, 1e-12),
            "e2e_delay_mean_s": float(d.mean()) if len(d) else float("nan"),
            "e2e_delay_p95_s": float(np.percentile(d, 95)) if len(d) else float("nan"),
            "queue_backlog_mean": self.queue_backlog / c,
            "waiting_backlog_mean": self.waiting_backlog / c,
            "one_hop_served_per_cycle": self.served_packets / c,
            "plan_size_mean": self.planned_links / c,
            "candidates_mean": self.candidates / c,
            "empty_candidate_cycle_frac": self.empty_candidate_cycles / c,
            "exec_failed_link_frac": self.failed_links / max(self.planned_links, 1),
            "sinr_violation_cycle_frac": self.sinr_violation_cycles / c,
            "sinr_violation_links": self.sinr_violation_links,
            "reward_mean": float(np.mean(self.rewards)) if self.rewards else 0.0,
            "decision_ms_mean": 1e3 * self.decision_seconds / c,
            "rehomed": self.rehomed,
            "relay_terminated_frac": self.relay_terminated / max(terminated, 1),
        }
        # share of *all* born packets delivered within D: drops and stragglers count as late,
        # so a lower delay among survivors cannot hide more drops
        for dl in ONTIME_DEADLINES_S:
            out[f"ontime_{dl:g}s"] = float((d <= dl).sum()) / max(self.born, 1)
        for k, v in self.reward_parts.items():
            out[f"reward_{k}_mean"] = v / c
        for k, v in self.terminations.items():
            out[f"term_{k}"] = v
        return out


def merge_summaries(summaries: list[dict]) -> dict[str, float]:
    """Mean over episodes of every numeric metric (keys missing count as 0)."""
    keys = sorted({k for s in summaries for k in s})
    out = {}
    for k in keys:
        vals = [s.get(k, 0.0) for s in summaries]
        vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out
