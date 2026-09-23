"""Dual-graph observation builders.

``observe(report)`` must be called once per decision boundary, in order, even
when the policy does not need a graph: it updates causal history (link age,
survival, quality trend, distance rate, neighbour stability) from the current
and past reports only.  ``build`` then turns the current report and the
candidate list into a :class:`DualGraph`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..contracts import Report
from ..physics import lin_to_db
from ..registry import slot
from ..scheduling.problem import SchedulingProblem
from .graph import DualGraph, FeatureSchema

OBSERVATION = slot("observation", "report -> dual graph (features, candidates mapping)")


class ObservationBuilder(ABC):
    schema: FeatureSchema

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def observe(self, report: Report) -> None: ...

    @abstractmethod
    def build(self, report: Report, problem: SchedulingProblem) -> DualGraph: ...


class LinkHistory:
    """Per directed pair causal statistics, updated once per boundary."""

    def __init__(self, survival_beta: float, quality_alpha: float, age_cap: int):
        self.beta, self.alpha, self.age_cap = survival_beta, quality_alpha, age_cap
        self.n = -1

    def reset(self) -> None:
        self.n = -1

    def update(self, present: np.ndarray, snr_db: np.ndarray, dist: np.ndarray) -> None:
        if self.n != len(present):
            n = self.n = len(present)
            self.age = np.zeros((n, n))
            self.survival = np.zeros((n, n))
            self.q_ema = snr_db.copy()
            self.trend = np.zeros((n, n))
            self.prev_present = np.zeros((n, n), dtype=bool)
            self.prev_dist = dist.copy()
            self.dist_rate = np.zeros((n, n))
            self.stability = np.zeros(n)
            self.first = True
        else:
            self.first = False
        self.age = np.where(present, self.age + 1, 0)
        self.survival = self.beta * self.survival + (1 - self.beta) * present
        new_ema = self.alpha * self.q_ema + (1 - self.alpha) * snr_db
        both = present & self.prev_present
        self.trend = np.where(both, np.clip((new_ema - self.q_ema) / 1.0, -1, 1), 0.0)
        self.q_ema = np.where(present, new_ema, snr_db)
        self.dist_rate = np.zeros_like(dist) if self.first else dist - self.prev_dist
        # Jaccard similarity of out-neighbour sets now vs. previous boundary
        inter = (present & self.prev_present).sum(axis=1)
        union = (present | self.prev_present).sum(axis=1)
        self.stability = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
        if self.first:
            self.stability = np.zeros(len(present))
        self.prev_present = present.copy()
        self.prev_dist = dist.copy()


NODE_FEATURES = ("out_queue", "max_out_queue", "max_out_hol", "waiting_occ", "waiting_urgency",
                 "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z", "degree",
                 "neighbor_stability", "out_candidates", "reachable")
EDGE_FEATURES = ("queue", "hol", "route_next", "snr_margin", "distance", "distance_rate",
                 "age", "survival", "quality_trend", "is_candidate", "reverse_queue")
INTER_FEATURES = ("shared_tx", "shared_rx", "tx_is_rx", "rx_is_tx", "conflict",
                  "budget_frac", "budget_log", "incompatible")
GLOBAL_FEATURES = ("candidates", "mean_queue", "total_queue", "wait_occ_mean",
                   "wait_pressure_mean", "num_nodes", "reachable_mean")


@OBSERVATION.register("standard", role="primary")
class StandardObservation(ObservationBuilder):
    """Comm graph over physically available links; interaction graph with conflict + interference edges."""

    def __init__(self, hol_ref_s: float = 0.2, pos_scale_m: float = 400.0,
                 speed_scale_mps: float = 30.0, dist_scale_m: float = 200.0,
                 survival_beta: float = 0.9, quality_alpha: float = 0.8, age_cap: int = 50,
                 inter_min_frac: float = 0.02, inter_top_k: int = 8, use_history: bool = True):
        self.hol_ref = hol_ref_s
        self.pos_scale, self.speed_scale, self.dist_scale = pos_scale_m, speed_scale_mps, dist_scale_m
        self.inter_min_frac, self.inter_top_k = inter_min_frac, inter_top_k
        self.use_history = use_history
        self.history = LinkHistory(survival_beta, quality_alpha, age_cap)
        self.schema = FeatureSchema(NODE_FEATURES, EDGE_FEATURES, EDGE_FEATURES, INTER_FEATURES,
                                    GLOBAL_FEATURES)
        self._last_cycle: tuple | None = None

    def reset(self) -> None:
        self.history.reset()
        self._last_cycle = None

    # -------------------------------------------------------------- history
    def _physical(self, report: Report):
        snr = report.tx_power[:, None] * report.gain / report.noise
        present = snr >= report.threshold * (1 - 1e-9)
        np.fill_diagonal(present, False)
        dist = np.linalg.norm(report.positions[:, None] - report.positions[None], axis=-1)
        return snr, present, dist

    def observe(self, report: Report) -> None:
        key = (report.run_id, report.episode, report.cycle)
        if key == self._last_cycle:
            return
        snr, present, dist = self._physical(report)
        self.history.update(present, lin_to_db(snr), dist)
        self._last_cycle = key

    # ---------------------------------------------------------------- build
    def build(self, report: Report, problem: SchedulingProblem) -> DualGraph:
        if self._last_cycle != (report.run_id, report.episode, report.cycle):
            raise RuntimeError("observe(report) must be called before build(report)")
        n = report.num_nodes
        h = self.history
        snr, present, dist = self._physical(report)
        snr_db = lin_to_db(snr)
        th_db = float(lin_to_db(report.threshold))

        # queue tables as (N, N) matrices
        q = report.queues
        occ = np.zeros((n, n))
        hol = np.zeros((n, n))
        route = np.zeros((n, n), dtype=bool)
        u, v = report.queue_links[:, 0], report.queue_links[:, 1]
        occ[u, v] = q.packets / np.maximum(q.capacity, 1)
        hol[u, v] = np.minimum(q.hol_wait / self.hol_ref, 2.0)
        route[u, v] = report.route_next

        cand = problem.links
        is_cand = np.zeros((n, n), dtype=bool)
        if len(cand):
            is_cand[cand[:, 0], cand[:, 1]] = True
        edge_mask = present | is_cand  # candidates are always physical, keep the guard
        tx, rx = np.nonzero(edge_mask)
        eid = -np.ones((n, n), dtype=np.int64)
        eid[tx, rx] = np.arange(len(tx))

        hist = self.use_history
        edge_x = np.stack([
            occ[tx, rx], hol[tx, rx], route[tx, rx].astype(float),
            np.clip((snr_db[tx, rx] - th_db) / 10.0, -1, 4),
            dist[tx, rx] / self.dist_scale,
            (h.dist_rate[tx, rx] / (self.speed_scale * report.cycle_length)) if hist else np.zeros(len(tx)),
            (np.minimum(h.age[tx, rx], h.age_cap) / h.age_cap) if hist else np.zeros(len(tx)),
            h.survival[tx, rx] if hist else np.zeros(len(tx)),
            h.trend[tx, rx] if hist else np.zeros(len(tx)),
            is_cand[tx, rx].astype(float), occ[rx, tx],
        ], axis=1).astype(np.float32) if len(tx) else np.zeros((0, len(EDGE_FEATURES)), np.float32)

        # node features
        cap = max(float(np.max(q.capacity)) if len(q.capacity) else 1.0, 1.0)
        out_pk = np.zeros(n)
        np.add.at(out_pk, u, q.packets)
        wait_occ = report.waiting_packets / np.maximum(report.waiting_capacity, 1)
        urgency = np.minimum(report.waiting_oldest / max(report.waiting_max_wait, 1e-9), 1.0)
        urgency = np.where(report.waiting_packets > 0, urgency, 0.0)
        reach = (report.next_hop >= 0).sum(axis=1) / max(n - 1, 1)
        node_x = np.column_stack([
            np.minimum(out_pk / cap, 4.0), occ.max(axis=1), hol.max(axis=1), wait_occ, urgency,
            report.positions / self.pos_scale, report.velocities / self.speed_scale,
            present.sum(axis=1) / max(n - 1, 1), h.stability if hist else np.zeros(n),
            is_cand.sum(axis=1) / max(n - 1, 1), reach,
        ]).astype(np.float32)

        # candidates and interaction graph
        c = len(cand)
        cand_edge = eid[cand[:, 0], cand[:, 1]] if c else np.zeros(0, np.int64)
        cand_x = edge_x[cand_edge] if c else np.zeros((0, len(EDGE_FEATURES)), np.float32)
        inter_index, inter_x = self._interaction(problem)

        glob = np.array([
            c / max(n, 1), float(np.mean(occ[u, v])) if len(u) else 0.0,
            float(q.packets.sum()) / max(cap * n, 1.0), float(np.mean(wait_occ)),
            float(np.mean(wait_occ * urgency)), n / 16.0, float(np.mean(reach)),
        ], dtype=np.float32)
        return DualGraph(node_x=node_x, edge_index=np.stack([tx, rx]).astype(np.int64),
                         edge_x=edge_x, cand_link=cand.astype(np.int64),
                         cand_edge=cand_edge.astype(np.int64), cand_x=cand_x,
                         inter_index=inter_index, inter_x=inter_x, global_x=glob)

    def _interaction(self, p: SchedulingProblem):
        c = p.num_candidates
        if c == 0:
            return np.zeros((2, 0), np.int64), np.zeros((0, len(INTER_FEATURES)), np.float32)
        s, r = p.links[:, 0], p.links[:, 1]
        shared_tx = s[:, None] == s[None, :]
        shared_rx = r[:, None] == r[None, :]
        tx_is_rx = s[:, None] == r[None, :]
        rx_is_tx = r[:, None] == s[None, :]
        conflict = shared_tx | shared_rx | tx_is_rx | rx_is_tx
        # fraction of b's interference budget (alone) consumed by a's transmitter
        budget = np.maximum(p.signal / p.threshold - p.noise, 1e-30)
        frac = p.cross / budget[None, :]
        np.fill_diagonal(frac, 0.0)
        keep = frac >= self.inter_min_frac
        if self.inter_top_k and c > self.inter_top_k:
            # per target b keep the k strongest interferers
            order = np.argsort(-frac, axis=0)[: self.inter_top_k]
            top = np.zeros_like(keep)
            top[order, np.arange(c)[None, :]] = True
            keep &= top
        keep |= conflict
        np.fill_diagonal(keep, False)
        a_idx, b_idx = np.nonzero(keep)
        incompatible = (frac >= 1.0) | (frac.T >= 1.0)
        feats = np.stack([
            shared_tx[a_idx, b_idx], shared_rx[a_idx, b_idx], tx_is_rx[a_idx, b_idx],
            rx_is_tx[a_idx, b_idx], conflict[a_idx, b_idx],
            np.minimum(frac[a_idx, b_idx], 2.0) / 2.0,
            (np.clip(np.log10(np.maximum(frac[a_idx, b_idx], 1e-3)), -3, 1) + 3) / 4,
            incompatible[a_idx, b_idx],
        ], axis=1).astype(np.float32)
        return np.stack([a_idx, b_idx]).astype(np.int64), feats
