"""Dual-graph container, feature schema and batching.

Index conventions (kept stable through batching and PPO recomputation):

* communication-graph node ``i`` is UAV ``i`` (stable within an episode);
* candidate ``c`` is local action index ``c`` and physical link ``cand_link[c]``,
  which is also communication edge ``cand_edge[c]``;
* interaction edges index candidates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class FeatureSchema:
    node: tuple[str, ...]
    edge: tuple[str, ...]
    cand: tuple[str, ...]
    inter: tuple[str, ...]
    glob: tuple[str, ...]

    def dims(self) -> dict[str, int]:
        return {k: len(getattr(self, k)) for k in ("node", "edge", "cand", "inter", "glob")}


@dataclass
class DualGraph:
    node_x: np.ndarray  # (N, Fn)
    edge_index: np.ndarray  # (2, E) tx -> rx node indices
    edge_x: np.ndarray  # (E, Fe)
    cand_link: np.ndarray  # (C, 2) stable (tx, rx) node ids
    cand_edge: np.ndarray  # (C,) comm-edge index of each candidate
    cand_x: np.ndarray  # (C, Fl) direct link features
    inter_index: np.ndarray  # (2, M) candidate a -> candidate b
    inter_x: np.ndarray  # (M, Fi)
    global_x: np.ndarray  # (Fg,)

    @property
    def num_nodes(self) -> int:
        return len(self.node_x)

    @property
    def num_candidates(self) -> int:
        return len(self.cand_link)


@dataclass
class GraphBatch:
    node_x: torch.Tensor  # (sumN, Fn)
    node_graph: torch.Tensor  # (sumN,)
    edge_index: torch.Tensor  # (2, sumE) global node indices
    edge_x: torch.Tensor
    cand_tx: torch.Tensor  # (sumC,) global node index of the transmitter
    cand_rx: torch.Tensor
    cand_x: torch.Tensor
    cand_graph: torch.Tensor  # (sumC,)
    cand_local: torch.Tensor  # (sumC,) local candidate index
    inter_index: torch.Tensor  # (2, sumM) global candidate indices
    inter_x: torch.Tensor
    global_x: torch.Tensor  # (B, Fg)
    num_graphs: int
    num_nodes: torch.Tensor  # (B,)
    num_cand: torch.Tensor  # (B,)
    max_cand: int
    cand_offset: torch.Tensor  # (B,) first global candidate index of each graph

    def pad_candidates(self, flat: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
        """(sumC, ...) -> (B, max_cand, ...), padding with ``fill``."""
        shape = (self.num_graphs, max(self.max_cand, 1)) + tuple(flat.shape[1:])
        out = flat.new_full(shape, fill)
        out[self.cand_graph, self.cand_local] = flat
        return out

    def cand_valid(self) -> torch.Tensor:
        """(B, max_cand) True where a real candidate exists."""
        ar = torch.arange(max(self.max_cand, 1), device=self.num_cand.device)
        return ar[None, :] < self.num_cand[:, None]


def batch_graphs(graphs: list[DualGraph], device: torch.device | str = "cpu") -> GraphBatch:
    n_nodes = np.array([g.num_nodes for g in graphs], dtype=np.int64)
    n_cand = np.array([g.num_candidates for g in graphs], dtype=np.int64)
    node_off = np.concatenate([[0], np.cumsum(n_nodes)[:-1]]).astype(np.int64)
    cand_off = np.concatenate([[0], np.cumsum(n_cand)[:-1]]).astype(np.int64)

    def cat(arrs, dim=0):
        return np.concatenate(arrs, axis=dim) if arrs else np.zeros(0)

    edge_index = cat([g.edge_index + node_off[i] for i, g in enumerate(graphs)], dim=1)
    inter_index = cat([g.inter_index + cand_off[i] for i, g in enumerate(graphs)], dim=1)
    cand_link = cat([g.cand_link.reshape(-1, 2) + node_off[i] for i, g in enumerate(graphs)])

    def t(x, dtype=torch.float32):
        return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype, device=device)

    g0 = graphs[0]
    return GraphBatch(
        node_x=t(cat([g.node_x for g in graphs])),
        node_graph=t(np.repeat(np.arange(len(graphs)), n_nodes), torch.long),
        edge_index=t(edge_index.reshape(2, -1), torch.long),
        edge_x=t(cat([g.edge_x for g in graphs]).reshape(-1, g0.edge_x.shape[1])),
        cand_tx=t(cand_link.reshape(-1, 2)[:, 0], torch.long),
        cand_rx=t(cand_link.reshape(-1, 2)[:, 1], torch.long),
        cand_x=t(cat([g.cand_x for g in graphs]).reshape(-1, g0.cand_x.shape[1])),
        cand_graph=t(np.repeat(np.arange(len(graphs)), n_cand), torch.long),
        cand_local=t(cat([np.arange(c) for c in n_cand]).astype(np.int64), torch.long),
        inter_index=t(inter_index.reshape(2, -1), torch.long),
        inter_x=t(cat([g.inter_x for g in graphs]).reshape(-1, g0.inter_x.shape[1])),
        global_x=t(np.stack([g.global_x for g in graphs])),
        num_graphs=len(graphs),
        num_nodes=t(n_nodes, torch.long),
        num_cand=t(n_cand, torch.long),
        max_cand=int(n_cand.max()) if len(n_cand) else 0,
        cand_offset=t(cand_off, torch.long),
    )
