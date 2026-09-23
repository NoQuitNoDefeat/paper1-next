"""Small graph building blocks (pure torch, no PyG)."""

from __future__ import annotations

import torch
from torch import nn


def mlp(dims: list[int], act: type[nn.Module] = nn.SiLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


def scatter_sum(src: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    out = src.new_zeros((size,) + tuple(src.shape[1:]))
    return out.index_add_(0, index, src)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    total = scatter_sum(src, index, size)
    count = torch.zeros(size, dtype=src.dtype, device=src.device).index_add_(
        0, index, torch.ones(len(index), dtype=src.dtype, device=src.device))
    return total / count.clamp(min=1.0).unsqueeze(-1)


def scatter_max(src: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    """Per-segment max; empty segments give 0."""
    out = src.new_full((size,) + tuple(src.shape[1:]), float("-inf"))
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out = out.scatter_reduce(0, idx, src, reduce="amax", include_self=True)
    return torch.where(torch.isinf(out), torch.zeros_like(out), out)


AGGREGATORS = {"sum": scatter_sum, "mean": scatter_mean, "max": scatter_max}


class DirectedMessagePassing(nn.Module):
    """Edge-conditioned layer that keeps direction: separate incoming / outgoing messages."""

    def __init__(self, dim: int, edge_dim: int, aggr: str = "mean"):
        super().__init__()
        self.aggr = AGGREGATORS[aggr]
        self.msg_in = mlp([2 * dim + edge_dim, dim, dim])
        self.msg_out = mlp([2 * dim + edge_dim, dim, dim])
        self.update = mlp([3 * dim, dim, dim])
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        n = h.shape[0]
        if edge_index.shape[1]:
            agg_in = self.aggr(self.msg_in(torch.cat([h[src], h[dst], e], -1)), dst, n)
            agg_out = self.aggr(self.msg_out(torch.cat([h[dst], h[src], e], -1)), src, n)
        else:
            agg_in = agg_out = torch.zeros_like(h)
        return self.norm(h + self.update(torch.cat([h, agg_in, agg_out], -1)))
