"""Scheduling model contract and the dual-graph composite model.

The contract is all the micro-step runner (sampling, PPO recomputation,
evaluation) relies on; it never touches layer names or hidden sizes:

* ``encode(batch)`` once per cycle -> :class:`Encoding`
* ``init_state(enc)`` / ``update_state(enc, state, chosen, active)`` — opaque
  selected-set state, recomputed from the action prefix during PPO updates
* ``logits(enc, state, dyn)`` -> (B, Cmax) unmasked candidate scores
* ``value(enc, state, remaining, micro, dyn)`` -> (B,) micro-state value

``dyn`` (B, Cmax, F) holds the controller's per-candidate micro-state features
(``MicroStepController.candidate_features``); models may ignore it.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

import torch
from torch import nn

from ..observation.graph import FeatureSchema, GraphBatch
from ..registry import slot
from ..scheduling.controller import CANDIDATE_FEATURE_DIM, MICRO_FEATURE_DIM
from .components import (ACTOR_HEAD, COMM_ENCODER, CRITIC_HEAD, INTERACTION_ENCODER, LIFT,
                         SET_SUMMARY, Readout)

MODEL = slot("model", "scheduling model (encoder + summary + actor/critic)")


@dataclass
class Encoding:
    z_actor: torch.Tensor  # (B, Cmax, d)
    z_critic: torch.Tensor
    ctx_actor: torch.Tensor  # (B, d)
    ctx_critic: torch.Tensor
    valid: torch.Tensor  # (B, Cmax) bool


class SchedulingModel(nn.Module):
    @abstractmethod
    def encode(self, batch: GraphBatch) -> Encoding: ...

    @abstractmethod
    def init_state(self, enc: Encoding): ...

    @abstractmethod
    def update_state(self, enc: Encoding, state, chosen: torch.Tensor, active: torch.Tensor): ...

    @abstractmethod
    def logits(self, enc: Encoding, state, dyn: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def value(self, enc: Encoding, state, remaining: torch.Tensor, micro: torch.Tensor,
              dyn: torch.Tensor) -> torch.Tensor: ...


class Trunk(nn.Module):
    """comm encoder -> lift -> interaction encoder, plus graph readout."""

    def __init__(self, schema: FeatureSchema, hidden: int, comm: dict, lift: dict, inter: dict):
        super().__init__()
        d = schema.dims()
        self.comm = COMM_ENCODER.build(comm, node_dim=d["node"], edge_dim=d["edge"], hidden=hidden)
        self.lift = LIFT.build(lift, node_hidden=self.comm.out_dim, link_dim=d["cand"], hidden=hidden)
        self.inter = INTERACTION_ENCODER.build(inter, inter_dim=d["inter"], hidden=hidden)
        self.readout = Readout(self.comm.out_dim, d["glob"])

    def forward(self, b: GraphBatch):
        h = self.comm(b.node_x, b.edge_index, b.edge_x)
        z = self.lift(h[b.cand_tx], h[b.cand_rx], b.cand_x)
        z = self.inter(z, b.inter_index, b.inter_x)
        ctx = self.readout(h, b.node_graph, b.num_graphs, b.global_x)
        return b.pad_candidates(z), ctx


@MODEL.register("dual_graph", role="primary")
class DualGraphModel(SchedulingModel):
    """Dual-graph encoder, gated-sum set summary, MLP actor/critic; trunk optionally shared.

    ``candidate_dynamics=True`` adds a projection of the controller's per-candidate
    micro-state features to every candidate embedding at every micro step.
    """

    def __init__(self, schema: FeatureSchema, hidden: int = 64, share_trunk: bool = True,
                 candidate_dynamics: bool = False,
                 comm_encoder: dict | str = "mpnn", lift: dict | str = "concat",
                 interaction_encoder: dict | str = "mpnn", set_summary: dict | str = "gated_sum",
                 actor_head: dict | str = "mlp", critic_head: dict | str = "mlp"):
        super().__init__()
        self.schema = schema
        self.share_trunk = share_trunk
        self.actor_trunk = Trunk(schema, hidden, comm_encoder, lift, interaction_encoder)
        self.critic_trunk = (self.actor_trunk if share_trunk
                             else Trunk(schema, hidden, comm_encoder, lift, interaction_encoder))
        self.summary_a = SET_SUMMARY.build(set_summary, hidden=hidden)
        self.summary_c = (self.summary_a if share_trunk
                          else SET_SUMMARY.build(set_summary, hidden=hidden))
        sd = self.summary_a.out_dim
        self.actor = ACTOR_HEAD.build(actor_head, link_dim=hidden, state_dim=sd, ctx_dim=hidden)
        self.critic = CRITIC_HEAD.build(critic_head, link_dim=hidden, state_dim=sd, ctx_dim=hidden,
                                        micro_dim=MICRO_FEATURE_DIM)
        self.dyn_actor = nn.Linear(CANDIDATE_FEATURE_DIM, hidden) if candidate_dynamics else None
        self.dyn_critic = (None if not candidate_dynamics else
                           self.dyn_actor if share_trunk else nn.Linear(CANDIDATE_FEATURE_DIM, hidden))

    def encode(self, batch: GraphBatch) -> Encoding:
        za, ca = self.actor_trunk(batch)
        zc, cc = (za, ca) if self.share_trunk else self.critic_trunk(batch)
        return Encoding(za, zc, ca, cc, batch.cand_valid())

    def init_state(self, enc: Encoding):
        b = enc.ctx_actor.shape[0]
        sa = self.summary_a.init(b, enc.ctx_actor)
        return (sa, sa) if self.share_trunk else (sa, self.summary_c.init(b, enc.ctx_critic))

    def update_state(self, enc: Encoding, state, chosen, active):
        rows = torch.arange(len(chosen), device=chosen.device)
        idx = chosen.clamp(min=0)
        sa = self.summary_a.update(state[0], enc.z_actor[rows, idx], active)
        if self.share_trunk:
            return (sa, sa)
        return (sa, self.summary_c.update(state[1], enc.z_critic[rows, idx], active))

    def logits(self, enc: Encoding, state, dyn) -> torch.Tensor:
        z = enc.z_actor if self.dyn_actor is None else enc.z_actor + self.dyn_actor(dyn)
        return self.actor(z, self.summary_a.read(state[0]), enc.ctx_actor)

    def value(self, enc: Encoding, state, remaining, micro, dyn) -> torch.Tensor:
        z = enc.z_critic if self.dyn_critic is None else enc.z_critic + self.dyn_critic(dyn)
        return self.critic(z, enc.valid, remaining, self.summary_c.read(state[1]), enc.ctx_critic,
                           micro)
