"""Learned baselines: pieces checked by hand, training smoke runs, checkpoint round trip."""

import numpy as np
import pytest
import torch

from fanet_next.baselines.grlinq import objective, slot_graph
from fanet_next.baselines.oneshot import _batch, _forward, bernoulli_logp
from fanet_next.baselines.zhao_gcn import laplacian
from fanet_next.config import load_config
from fanet_next.experiment.assemble import build_env, build_policy, policy_from_checkpoint
from fanet_next.experiment.train_baseline import BaselineRun
from fanet_next.training.checkpoint import load_checkpoint

from helpers import fixed_env, line_positions


def test_normalized_laplacian():
    a = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=bool)
    lap = laplacian(a)
    assert np.allclose(lap, [[1, -1, 0], [-1, 1, 0], [0, 0, 1]])


def test_grlinq_objective_counts_only_successful_links():
    # 0 -> 1 and 2 -> 3 far apart; 1 -> 2 shares endpoints with both
    births = [(0.001, 0, 1), (0.002, 2, 3), (0.003, 1, 2), (0.004, 1, 2)]
    env = fixed_env(line_positions(4, spacing=150.0), births, horizon=4)
    env.reset(0)
    inp = env.step([]).next_input
    links = [tuple(l) for l in inp.problem.links]
    g = slot_graph(inp)
    w = g["w"]
    x = np.zeros(len(links), dtype=bool)
    i01, i12 = links.index((0, 1)), links.index((1, 2))
    x[i01] = True
    assert objective(inp, w, x) == pytest.approx(w[i01] / w.sum())
    x[i12] = True  # shares node 1 with (0, 1): both fail under half duplex
    assert objective(inp, w, x) == 0.0
    assert set(map(tuple, g["edges"])) >= {(i12, i01), (i01, i12)}  # endpoint-sharing edges


def _smoke_cfg(name):
    return load_config(f"configs/baselines/{name}.toml", [
        "training.iterations=1", "training.num_envs=2", "training.rollout_cycles=6",
        "training.eval_every=0", "training.checkpoint_every=1", "training.keep_every=1",
        "scenario.horizon=30", "imitation.cycles=8", "imitation.epochs=1"])


@pytest.mark.parametrize("name", ["zhao_gcn", "oneshot_ppo", "grlinq"])
def test_baseline_trains_and_reloads_from_checkpoint(name, tmp_path):
    cfg = _smoke_cfg(name)
    run = BaselineRun(cfg, tmp_path)
    run.train()
    ck = load_checkpoint(tmp_path / "checkpoints" / "iter_00001.pt")
    assert ck["policy_spec"]["type"] == name
    pol = policy_from_checkpoint(cfg, ck)
    for (k, v), (k2, v2) in zip(run.policy.state_dict().items(), pol.state_dict().items()):
        assert k == k2 and torch.equal(v, v2)
    env = build_env(cfg, run_id="t", build_graph=pol.needs_graph)
    inp = env.reset(10_000_005)
    for _ in range(10):
        out = pol.act([inp], mode="greedy")[0]
        inp = env.step(out.actions).next_input
    assert env.metrics.summary()["sinr_violation_cycle_frac"] == 0


def test_oneshot_logp_is_reproduced_by_the_update_forward():
    cfg = _smoke_cfg("oneshot_ppo")
    pol = build_policy(cfg, {"type": "oneshot_ppo"}, seed=3)
    env = build_env(cfg, run_id="t", build_graph=True)
    inp = env.reset(10_000_006)
    recs = []
    for _ in range(12):
        out = pol.act([inp], mode="sample")[0]
        if len(out.record["mask"]):
            recs.append(out.record)
        inp = env.step(out.actions).next_input
    graphs, mask, micro, dyn, act = _batch(recs)
    with torch.no_grad():
        logits, value, mask_t = _forward(pol.model, graphs, mask, micro, dyn)
        logp = bernoulli_logp(logits, act, mask_t.float())
    assert np.allclose(logp.numpy(), [r["logp"] for r in recs], atol=1e-5)
    assert np.allclose(value.numpy(), [r["value"] for r in recs], atol=1e-5)
