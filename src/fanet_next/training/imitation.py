"""Imitation warm start / diagnostic: fit the actor to a teacher policy's micro actions.

The teacher (any registered policy) drives the environments; its ordered
selection is replayed on a fresh controller to recover the mask and micro
features before every action, producing ordinary :class:`MicroRecord`s.  The
actor is fitted by maximum likelihood through the same ``replay`` used by PPO,
and the critic regresses discounted returns of the teacher's rewards.  This
changes only how training starts; the scheduling method is unchanged.
"""

from __future__ import annotations

import numpy as np
import torch

from ..contracts import EndType
from ..loop import SchedulingEnv
from ..model.runner import MicroRecord, replay
from ..policy.base import Policy
from ..scheduling.controller import MICRO_FEATURE_DIM
from .advantages import compute_stream_advantages
from .records import CUT, CycleRecord


def teacher_records(envs: list[SchedulingEnv], teacher: Policy, seeds: list[int], cycles: int,
                    gamma: float) -> list[CycleRecord]:
    """Run the teacher for ``cycles`` lockstep cycles; return records with MC-style returns."""
    for env, s in zip(envs, seeds):
        env.reset(s)
    streams: list[list[CycleRecord]] = [[] for _ in envs]
    next_seed = max(seeds) + 1
    for _ in range(cycles):
        inputs = [env.current for env in envs]
        outs = teacher.act(inputs, mode="greedy")
        for e, (env, inp, out) in enumerate(zip(envs, inputs, outs)):
            ctl = env.constraints.start(inp.problem)
            masks, micro = [], []
            for a in out.actions:
                masks.append(ctl.mask)
                micro.append(ctl.micro_features())
                ctl.step(a)
            masks.append(ctl.mask)
            micro.append(ctl.micro_features())
            k = len(out.actions)
            rec = MicroRecord(graph=inp.graph, masks=np.array(masks, bool).reshape(k + 1, -1),
                              actions=np.array(out.actions, np.int64),
                              logp=np.zeros(k, np.float32), values=np.zeros(k + 1, np.float32),
                              micro=np.array(micro, np.float32).reshape(k + 1, MICRO_FEATURE_DIM))
            tr = env.step(out.actions)
            streams[e].append(CycleRecord(micro=rec, reward=tr.reward.total,
                                          raw_reward=tr.reward.total, end=tr.end.value))
            if tr.end is not EndType.CONTINUE:
                env.reset(next_seed)
                next_seed += 1
    for s in streams:
        if s[-1].end == "continue":
            s[-1].end = CUT
        # lambda = 1 with zero values: returns are discounted reward sums within the slice
        compute_stream_advantages(s, gamma, 1.0)
    return [r for s in streams for r in s]


def fit(model, records: list[CycleRecord], *, epochs: int = 4, minibatch: int = 256,
        lr: float = 1e-3, vf_coef: float = 0.5, reward_scale: float = 1.0, seed: int = 0,
        device: str = "cpu") -> dict[str, float]:
    """Maximum likelihood on teacher actions (+ value regression); returns final stats."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    rng = np.random.default_rng(seed)
    stats = {}
    for ep in range(epochs):
        order = rng.permutation(len(records))
        nll, vl, acc, n_act = 0.0, 0.0, 0.0, 0
        for start in range(0, len(records), minibatch):
            mb = [records[i] for i in order[start:start + minibatch]]
            rp = replay(model, [r.micro for r in mb], device=device)
            k1 = rp.values.shape[1]
            ret = torch.zeros(len(mb), k1)
            for b, r in enumerate(mb):
                ret[b, : len(r.returns)] = torch.as_tensor(r.returns) / reward_scale
            am, sm = rp.act_mask, rp.state_mask
            n = am.sum().clamp(min=1)
            loss_pi = -(rp.logp * am).sum() / n
            loss_v = 0.5 * (((rp.values - ret) ** 2) * sm).sum() / sm.sum().clamp(min=1)
            opt.zero_grad()
            (loss_pi + vf_coef * loss_v).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            nll += loss_pi.item() * n.item()
            vl += loss_v.item() * n.item()
            acc += ((rp.logp > np.log(0.5)) & am).sum().item()
            n_act += int(n.item())
        stats = {"epoch": ep + 1, "nll": nll / max(n_act, 1), "value_loss": vl / max(n_act, 1),
                 "p_teacher_gt_half": acc / max(n_act, 1)}
    return stats
