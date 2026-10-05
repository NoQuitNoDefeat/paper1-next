"""Data contracts exchanged between backend, observation, policy and reward.

Units: seconds, bytes, watts, linear gains.  Links are directed ``(tx, rx)``
node-id pairs; node ids are stable for a whole episode, so a link id is a
stable physical identity across cycles.  Arrays indexed by ``q`` follow the
backend's fixed registered-queue order ``Report.queue_links``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

Link = tuple[int, int]


class EndType(str, Enum):
    """How a physical-cycle transition ends (for TD bootstrapping)."""

    CONTINUE = "continue"  # next observation belongs to the same episode
    TRUNCATED = "truncated"  # external time limit: keep V(next), stop recursion
    TERMINATED = "terminated"  # genuine task end: no future value


class ExecutionError(RuntimeError):
    """Backend could not execute a plan (invalid plan, IPC failure, ...).

    Never folded into a normal end or a zero reward; collectors report it.
    """


@dataclass(frozen=True)
class QueueSnapshot:
    packets: np.ndarray  # (Q,) int
    bytes: np.ndarray  # (Q,) int
    capacity: np.ndarray  # (Q,) int, packets
    hol_wait: np.ndarray  # (Q,) s, sample time - enqueue time of head; 0 if empty


@dataclass(frozen=True)
class Report:
    """Everything the scheduler may use at a decision boundary (causal only)."""

    run_id: str
    episode: int
    cycle: int
    time: float  # sample time (cycle start)
    cycle_length: float
    num_nodes: int
    positions: np.ndarray  # (N, 3) reported
    velocities: np.ndarray  # (N, 3) reported
    gain: np.ndarray  # (N, N) decision-time channel estimate Ghat[tx, rx]
    tx_power: np.ndarray  # (N,) W
    noise: float  # W
    # linear SINR threshold to plan against: the decoding threshold times the channel
    # model's fade margin (1 for a channel without fading)
    threshold: float
    service_bytes: int  # bytes a scheduled link can move in one cycle
    packet_size: int
    queue_links: np.ndarray  # (Q, 2) registered next-hop queues (fixed per episode)
    queues: QueueSnapshot
    route_next: np.ndarray  # (Q,) bool: (u, v) is a current next hop of u
    next_hop: np.ndarray  # (N, N) int next_hop[u, dst], -1 = no route
    waiting_packets: np.ndarray  # (N,) int
    waiting_capacity: np.ndarray  # (N,) int
    waiting_oldest: np.ndarray  # (N,) s since the oldest waiting packet entered; 0 if empty
    waiting_max_wait: float
    # (Q, N) packets per destination in each queue: the commodity view backpressure needs;
    # None when a backend does not report it
    queue_dst: np.ndarray | None = None


@dataclass(frozen=True)
class Plan:
    """A complete one-cycle transmission plan answering ``Report.cycle``."""

    cycle: int
    links: tuple[Link, ...]


@dataclass
class CycleFacts:
    """Actual execution facts of one cycle, in the backend's authority."""

    cycle: int
    t_start: float
    t_end: float
    planned: tuple[Link, ...]
    succeeded: tuple[Link, ...]
    failed: tuple[Link, ...]
    exec_sinr: np.ndarray  # (len(planned),) actual SINR at execution
    served_packets: np.ndarray  # (Q,) packets actually removed from each queue
    served_bytes: np.ndarray  # (Q,)
    service_ref_bytes: int
    post_service: QueueSnapshot  # after service, before terminations and admission
    births: int
    risk_packets: int  # queued + waiting at cycle start + births in this cycle
    delivered_ids: list[int] = field(default_factory=list)
    delivered_delays: list[float] = field(default_factory=list)  # end-to-end, s
    delivered_bytes: int = 0
    terminations: dict[str, list[int]] = field(default_factory=dict)  # reason -> ids
    queued_end: int = 0  # after admission
    waiting_end: int = 0
    waiting_post_service: int = 0
    moved_to_waiting: int = 0
    restored_from_waiting: int = 0
    rehomed: int = 0
    relay_terminated: int = 0  # terminated packets that were at a relay (not their source)
    radio_events: dict[str, int] = field(default_factory=dict)  # PHY/MAC trace counts (ns-3)
    terminated_nodes: dict[int, int] = field(default_factory=dict)  # node -> packets terminated there

    @property
    def terminated_ids(self) -> list[int]:
        return [pid for ids in self.terminations.values() for pid in ids]


@dataclass
class StepOutcome:
    facts: CycleFacts
    next_report: Report  # last valid state even when the episode ends
    end: EndType
