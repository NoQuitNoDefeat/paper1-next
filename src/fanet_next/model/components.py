"""Replaceable model parts.  Each sub-slot takes dimensions by injection.

* ``comm_encoder``: communication graph -> node embeddings
* ``lift``: (tx context, rx context, direct link features) -> link embeddings
* ``interaction_encoder``: link interaction graph -> refined link embeddings
* ``set_summary``: selected-set summary state and its per-step update
* ``actor_head`` / ``critic_head``: candidate scores / micro-state value
"""

from __future__ import annotations

import torch
from torch import nn

from ..registry import slot
from .layers import DirectedMessagePassing, mlp, scatter_max, scatter_mean

COMM_ENCODER = slot("comm_encoder", "communication graph -> node embeddings")
LIFT = slot("lift", "node context + direct link features -> link embeddings")
INTERACTION_ENCODER = slot("interaction_encoder", "link interaction graph -> link embeddings")
SET_SUMMARY = slot("set_summary", "selected-set summary for micro steps")
ACTOR_HEAD = slot("actor_head", "candidate scores")
CRITIC_HEAD = slot("critic_head", "micro-state value")


# ---------------------------------------------------------------- encoders
@COMM_ENCODER.register("mpnn", role="primary")
class CommMPNN(nn.Module):
    """Directed edge-conditioned message passing over the communication graph."""

    def __init__(self, node_dim: int, edge_dim: int, hidden: int, layers: int = 2,
                 edge_hidden: int | None = None, aggr: str = "mean"):
        super().__init__()
        eh = edge_hidden or hidden // 2
        self.node_proj = mlp([node_dim, hidden, hidden])
        self.edge_proj = mlp([edge_dim, eh, eh])
        self.layers = nn.ModuleList(DirectedMessagePassing(hidden, eh, aggr) for _ in range(layers))
        self.out_dim = hidden

    def forward(self, node_x, edge_index, edge_x):
        h = self.node_proj(node_x)
        e = self.edge_proj(edge_x)
        for layer in self.layers:
            h = layer(h, edge_index, e)
        return h


@COMM_ENCODER.register("node_mlp", role="control")
class CommNodeMLP(nn.Module):
    """Ablation: per-node MLP, no communication-graph message passing."""

    def __init__(self, node_dim: int, edge_dim: int, hidden: int):
        super().__init__()
        self.net = mlp([node_dim, hidden, hidden])
        self.norm = nn.LayerNorm(hidden)
        self.out_dim = hidden

    def forward(self, node_x, edge_index, edge_x):
        return self.norm(self.net(node_x))


@LIFT.register("concat", role="primary")
class LiftConcat(nn.Module):
    """z = LN(MLP[h_tx, h_rx, x_link] + W x_link): direction-aware, with a direct link path."""

    def __init__(self, node_hidden: int, link_dim: int, hidden: int, direct_path: bool = True):
        super().__init__()
        self.net = mlp([2 * node_hidden + link_dim, hidden, hidden])
        self.direct = nn.Linear(link_dim, hidden, bias=False) if direct_path else None
        self.norm = nn.LayerNorm(hidden)
        self.out_dim = hidden

    def forward(self, h_tx, h_rx, x_link):
        z = self.net(torch.cat([h_tx, h_rx, x_link], -1))
        if self.direct is not None:
            z = z + self.direct(x_link)
        return self.norm(z)


@LIFT.register("context_only", role="control")
class LiftContextOnly(nn.Module):
    """Ablation: link embedding from endpoint contexts only (no direct link features)."""

    def __init__(self, node_hidden: int, link_dim: int, hidden: int):
        super().__init__()
        self.net = mlp([2 * node_hidden, hidden, hidden])
        self.norm = nn.LayerNorm(hidden)
        self.out_dim = hidden

    def forward(self, h_tx, h_rx, x_link):
        return self.norm(self.net(torch.cat([h_tx, h_rx], -1)))


@INTERACTION_ENCODER.register("mpnn", role="primary")
class InteractionMPNN(nn.Module):
    """Directed message passing over conflict / interference edges between candidates."""

    def __init__(self, inter_dim: int, hidden: int, layers: int = 1, edge_hidden: int | None = None,
                 aggr: str = "mean"):
        super().__init__()
        eh = edge_hidden or hidden // 2
        self.edge_proj = mlp([inter_dim, eh, eh])
        self.layers = nn.ModuleList(DirectedMessagePassing(hidden, eh, aggr) for _ in range(layers))

    def forward(self, z, inter_index, inter_x):
        e = self.edge_proj(inter_x)
        for layer in self.layers:
            z = layer(z, inter_index, e)
        return z


@INTERACTION_ENCODER.register("none", role="control")
class InteractionNone(nn.Module):
    """Ablation: no link interaction graph (links only see their own endpoints)."""

    def __init__(self, inter_dim: int, hidden: int):
        super().__init__()

    def forward(self, z, inter_index, inter_x):
        return z


class Readout(nn.Module):
    """Graph context from pooled node embeddings and global features."""

    def __init__(self, hidden: int, global_dim: int):
        super().__init__()
        self.net = mlp([2 * hidden + global_dim, hidden, hidden])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, node_graph, num_graphs, global_x):
        pooled = torch.cat([scatter_mean(h, node_graph, num_graphs),
                            scatter_max(h, node_graph, num_graphs), global_x], -1)
        return self.norm(self.net(pooled))


# ------------------------------------------------------------- set summary
@SET_SUMMARY.register("gated_sum", role="primary")
class GatedSum(nn.Module):
    """s <- s + sigmoid(W z + b) * z, starting from zero."""

    def __init__(self, hidden: int):
        super().__init__()
        self.gate = nn.Linear(hidden, hidden)
        self.out_dim = hidden

    def init(self, batch: int, like: torch.Tensor) -> torch.Tensor:
        return like.new_zeros(batch, self.out_dim)

    def update(self, s, z_chosen, active):
        new = s + torch.sigmoid(self.gate(z_chosen)) * z_chosen
        return torch.where(active.unsqueeze(-1), new, s)


@SET_SUMMARY.register("none", role="control")
class NoSummary(nn.Module):
    """Ablation: no selected-set summary (actor sees only the feasibility mask)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.out_dim = 1

    def init(self, batch, like):
        return like.new_zeros(batch, 1)

    def update(self, s, z_chosen, active):
        return s


# ------------------------------------------------------------------ heads
@ACTOR_HEAD.register("mlp", role="primary")
class ActorMLP(nn.Module):
    """score_i = MLP[z_i, s, ctx]."""

    def __init__(self, link_dim: int, state_dim: int, ctx_dim: int, hidden: int = 128):
        super().__init__()
        self.net = mlp([link_dim + state_dim + ctx_dim, hidden, hidden, 1])
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, z, s, ctx):  # z (B, C, d)
        b, c, _ = z.shape
        inp = torch.cat([z, s.unsqueeze(1).expand(b, c, -1), ctx.unsqueeze(1).expand(b, c, -1)], -1)
        return self.net(inp).squeeze(-1)


@CRITIC_HEAD.register("mlp", role="primary")
class CriticMLP(nn.Module):
    """V = MLP[ctx, s, mean(remaining feasible z), mean(all z), micro features]."""

    def __init__(self, link_dim: int, state_dim: int, ctx_dim: int, micro_dim: int,
                 hidden: int = 128):
        super().__init__()
        self.net = mlp([ctx_dim + state_dim + 2 * link_dim + micro_dim, hidden, hidden, 1])

    def forward(self, z, valid, remaining, s, ctx, micro):
        def masked_mean(mask):
            w = mask.to(z.dtype).unsqueeze(-1)
            return (z * w).sum(1) / w.sum(1).clamp(min=1.0)

        inp = torch.cat([ctx, s, masked_mean(remaining), masked_mean(valid), micro], -1)
        return self.net(inp).squeeze(-1)
