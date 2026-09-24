"""Micro-step execution of a :class:`SchedulingModel`, shared by sampling and PPO.

Per cycle the graph is encoded once; then, for micro state j = 0..k:
value V(s_j) is computed from the current summary, remaining mask and
controller features; while candidates remain, an action is chosen and the
summary updated.  ``s_k`` (no feasible candidate) is the cycle-boundary state.

``sample`` drives live controllers; ``replay`` recomputes log-probabilities,
entropies and values from stored graph inputs, masks and action prefixes with
the *current* parameters, so gradients reach every model component.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..observation.graph import DualGraph, batch_graphs
from ..scheduling.controller import CANDIDATE_FEATURE_DIM, MICRO_FEATURE_DIM
from .dual_graph import SchedulingModel


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """log-probabilities over ``mask``; rows without any valid entry return zeros."""
    has_any = mask.any(-1, keepdim=True)
    safe = torch.where(mask, logits, torch.full_like(logits, -1e9))
    logp = torch.log_softmax(safe, -1)
    logp = torch.where(mask, logp, torch.full_like(logp, float("-inf")))
    return torch.where(has_any, logp, torch.zeros_like(logp))


def entropy_from_logp(logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    p = logp.exp() * mask
    return -(p * torch.where(mask, logp, torch.zeros_like(logp))).sum(-1)


@dataclass
class MicroRecord:
    """What sampling stores for one cycle (numpy; replayable later)."""

    graph: DualGraph
    masks: np.ndarray  # (k+1, C) bool, mask before each action; last row: all False
    actions: np.ndarray  # (k,) int64
    logp: np.ndarray  # (k,) float32
    values: np.ndarray  # (k+1,) float32
    micro: np.ndarray  # (k+1, F) float32
    cand_dyn: np.ndarray | None = None  # (k+1, C, F_dyn) controller per-candidate features

    @property
    def num_actions(self) -> int:
        return len(self.actions)


@torch.no_grad()
def sample(model: SchedulingModel, inputs, *, mode: str = "sample",
           generator: torch.Generator | None = None, device: str = "cpu") -> list[MicroRecord]:
    if mode not in {"sample", "greedy"}:
        raise ValueError(f"unknown mode {mode!r}")
    graphs = [inp.graph for inp in inputs]
    batch = batch_graphs(graphs, device)
    enc = model.encode(batch)
    state = model.init_state(enc)
    b_n, c_max = enc.valid.shape
    n_cand = [g.num_candidates for g in graphs]
    rec = [dict(masks=[], actions=[], logp=[], values=[], micro=[], dyn=[]) for _ in inputs]
    finished = np.zeros(b_n, dtype=bool)
    while True:
        mask = np.zeros((b_n, c_max), dtype=bool)
        micro = np.zeros((b_n, MICRO_FEATURE_DIM), dtype=np.float32)
        dyn = np.zeros((b_n, c_max, CANDIDATE_FEATURE_DIM), dtype=np.float32)
        for b, inp in enumerate(inputs):
            if not finished[b]:
                mask[b, : n_cand[b]] = inp.controller.mask
                micro[b] = inp.controller.micro_features()
                dyn[b, : n_cand[b]] = inp.controller.candidate_features()
        mask_t = torch.as_tensor(mask, device=device)
        dyn_t = torch.as_tensor(dyn, device=device)
        v = model.value(enc, state, mask_t, torch.as_tensor(micro, device=device), dyn_t).cpu().numpy()
        active = mask.any(1) & ~finished
        for b in np.nonzero(~finished)[0]:
            rec[b]["masks"].append(mask[b, : n_cand[b]].copy())
            rec[b]["values"].append(v[b])
            rec[b]["micro"].append(micro[b])
            rec[b]["dyn"].append(dyn[b, : n_cand[b]].copy())
        finished |= ~active
        if not active.any():
            break
        logp_all = masked_log_softmax(model.logits(enc, state, dyn_t), mask_t)
        rows = torch.as_tensor(np.nonzero(active)[0], device=device)
        if mode == "greedy":
            a = logp_all[rows].argmax(-1)
        else:
            a = torch.multinomial(logp_all[rows].exp(), 1, generator=generator).squeeze(1)
        chosen = torch.full((b_n,), -1, dtype=torch.long, device=device)
        chosen[rows] = a
        lp = logp_all[rows, a].cpu().numpy()
        for i, b in enumerate(rows.tolist()):
            inputs[b].controller.step(int(a[i]))
            rec[b]["actions"].append(int(a[i]))
            rec[b]["logp"].append(lp[i])
        state = model.update_state(enc, state, chosen, torch.as_tensor(active, device=device))
    out = []
    for b, r in enumerate(rec):
        c = n_cand[b]
        out.append(MicroRecord(
            graph=graphs[b], masks=np.array(r["masks"], dtype=bool).reshape(len(r["masks"]), c),
            actions=np.array(r["actions"], dtype=np.int64),
            logp=np.array(r["logp"], dtype=np.float32),
            values=np.array(r["values"], dtype=np.float32),
            micro=np.array(r["micro"], dtype=np.float32).reshape(-1, MICRO_FEATURE_DIM),
            cand_dyn=np.array(r["dyn"], dtype=np.float32).reshape(len(r["dyn"]), c,
                                                                   CANDIDATE_FEATURE_DIM)))
    return out


@torch.no_grad()
def initial_values(model: SchedulingModel, inputs, device: str = "cpu") -> np.ndarray:
    """V(s_0) of fresh decision inputs (bootstrap at truncation / slice ends)."""
    batch = batch_graphs([inp.graph for inp in inputs], device)
    enc = model.encode(batch)
    state = model.init_state(enc)
    b_n, c_max = enc.valid.shape
    mask = np.zeros((b_n, c_max), dtype=bool)
    micro = np.zeros((b_n, MICRO_FEATURE_DIM), dtype=np.float32)
    dyn = np.zeros((b_n, c_max, CANDIDATE_FEATURE_DIM), dtype=np.float32)
    for b, inp in enumerate(inputs):
        m = inp.controller.mask
        mask[b, : len(m)] = m
        micro[b] = inp.controller.micro_features()
        dyn[b, : len(m)] = inp.controller.candidate_features()
    return model.value(enc, state, torch.as_tensor(mask, device=device),
                       torch.as_tensor(micro, device=device),
                       torch.as_tensor(dyn, device=device)).cpu().numpy()


@dataclass
class Replay:
    logp: torch.Tensor  # (B, K)
    entropy: torch.Tensor  # (B, K)
    values: torch.Tensor  # (B, K+1)
    act_mask: torch.Tensor  # (B, K) real actions
    state_mask: torch.Tensor  # (B, K+1) real micro states


def replay(model: SchedulingModel, records: list[MicroRecord], device: str = "cpu") -> Replay:
    batch = batch_graphs([r.graph for r in records], device)
    enc = model.encode(batch)
    state = model.init_state(enc)
    b_n, c_max = enc.valid.shape
    k = np.array([r.num_actions for r in records])
    k_max = int(k.max()) if len(k) else 0
    masks = np.zeros((b_n, k_max + 1, c_max), dtype=bool)
    micro = np.zeros((b_n, k_max + 1, MICRO_FEATURE_DIM), dtype=np.float32)
    actions = np.zeros((b_n, max(k_max, 1)), dtype=np.int64)
    dyn = np.zeros((b_n, k_max + 1, c_max, CANDIDATE_FEATURE_DIM), dtype=np.float32)
    for b, r in enumerate(records):
        c = r.graph.num_candidates
        masks[b, : k[b] + 1, :c] = r.masks
        micro[b, : k[b] + 1] = r.micro
        actions[b, : k[b]] = r.actions
        if r.cand_dyn is not None:
            dyn[b, : k[b] + 1, :c] = r.cand_dyn
    masks_t = torch.as_tensor(masks, device=device)
    micro_t = torch.as_tensor(micro, device=device)
    actions_t = torch.as_tensor(actions, device=device)
    dyn_t = torch.as_tensor(dyn, device=device)
    k_t = torch.as_tensor(k, device=device)
    values, logps, ents = [], [], []
    for j in range(k_max + 1):
        values.append(model.value(enc, state, masks_t[:, j], micro_t[:, j], dyn_t[:, j]))
        if j == k_max:
            break
        act = j < k_t
        logp_all = masked_log_softmax(model.logits(enc, state, dyn_t[:, j]), masks_t[:, j])
        a = actions_t[:, j]
        logps.append(torch.where(act, logp_all.gather(1, a[:, None]).squeeze(1),
                                 torch.zeros_like(a, dtype=logp_all.dtype)))
        ents.append(torch.where(act, entropy_from_logp(logp_all, masks_t[:, j]),
                                torch.zeros(b_n, device=device)))
        state = model.update_state(enc, state, torch.where(act, a, torch.full_like(a, -1)), act)
    ar = torch.arange(k_max + 1, device=device)
    empty = torch.zeros(b_n, 0, device=device)
    return Replay(
        logp=torch.stack(logps, 1) if logps else empty,
        entropy=torch.stack(ents, 1) if ents else empty,
        values=torch.stack(values, 1),
        act_mask=(ar[None, :k_max] < k_t[:, None]),
        state_mask=(ar[None, :] <= k_t[:, None]))
