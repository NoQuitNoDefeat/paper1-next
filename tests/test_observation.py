"""Dual graph: candidate ↔ edge ↔ physical link mapping, batching offsets, causality."""

import numpy as np
import torch

from fanet_next.observation import batch_graphs

from helpers import fixed_env, line_positions


def _graphs(env, cycles):
    inp = env.reset(0)
    out = [inp]
    for _ in range(cycles):
        inp = env.step(list(_greedy(inp))).next_input
        out.append(inp)
    return out


def _greedy(inp):
    c = inp.controller
    while not c.done:
        c.step(int(np.nonzero(c.mask)[0][0]))
    return c.selected


def _scene(tail_positions=None):
    base = [[0, 0, 100], [100, 0, 100], [200, 0, 100], [100, 100, 100], [0, 100, 100]]
    positions = [base] * 6 + [tail_positions or base] * 5
    births = [(0.001 + 0.004 * i, i % 5, (i + 2) % 5) for i in range(20)]
    return fixed_env(positions, births, horizon=10)


def test_candidate_mapping_is_consistent():
    for inp in _graphs(_scene(), 5):
        g, p = inp.graph, inp.problem
        assert np.array_equal(g.cand_link, p.links)
        tx, rx = g.edge_index
        for c, e in enumerate(g.cand_edge):
            assert (tx[e], rx[e]) == tuple(p.links[c])
        assert g.cand_x.shape == (p.num_candidates, len(inp.graph.edge_x[0]) if len(g.edge_x) else 11)
        assert g.inter_index.max(initial=-1) < p.num_candidates


def test_batching_keeps_graphs_apart():
    inputs = [i for i in _graphs(_scene(), 5) if i.problem.num_candidates > 0][:3]
    graphs = [i.graph for i in inputs]
    b = batch_graphs(graphs)
    off_n, off_c = 0, 0
    for i, g in enumerate(graphs):
        sl = slice(off_c, off_c + g.num_candidates)
        assert torch.equal(b.cand_tx[sl], torch.as_tensor(g.cand_link[:, 0]) + off_n)
        assert torch.equal(b.cand_graph[sl], torch.full((g.num_candidates,), i))
        assert torch.equal(b.cand_local[sl], torch.arange(g.num_candidates))
        m = (b.inter_index[0] >= off_c) & (b.inter_index[0] < off_c + g.num_candidates)
        assert ((b.inter_index[1][m] >= off_c) & (b.inter_index[1][m] < off_c + g.num_candidates)).all()
        off_n += g.num_nodes
        off_c += g.num_candidates
    padded = b.pad_candidates(b.cand_x)
    assert torch.equal(padded[1, : graphs[1].num_candidates], torch.as_tensor(graphs[1].cand_x))


def test_future_positions_do_not_change_current_observation():
    far = [[0, 0, 100], [900, 0, 100], [200, 0, 100], [100, 900, 100], [0, 100, 100]]
    a = _graphs(_scene(), 5)
    b = _graphs(_scene(tail_positions=far), 5)
    for x, y in zip(a, b):  # cycles 0..5 identical; the scenes differ from cycle 6 on
        for field in ("node_x", "edge_x", "cand_x", "inter_x", "global_x"):
            assert np.array_equal(getattr(x.graph, field), getattr(y.graph, field)), field


def test_history_features_age_and_waiting_summary():
    env = fixed_env(line_positions(3), [(0.005, 0, 2)], horizon=6)
    inp = env.reset(0)
    names = env.observation.schema.edge
    ages = []
    for _ in range(3):
        g = inp.graph
        e = [i for i in range(g.edge_index.shape[1]) if tuple(g.edge_index[:, i]) == (0, 1)][0]
        ages.append(g.edge_x[e, names.index("age")])
        inp = env.step(list(_greedy(inp))).next_input
    assert ages[0] < ages[1] < ages[2]
    glob = env.observation.schema.glob
    assert inp.graph.global_x[glob.index("wait_occ_mean")] == 0.0
