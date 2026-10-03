"""PPO recomputation: identical probabilities, gradients into the encoder, prefix dependence."""

import numpy as np
import pytest
import torch

from fanet_next.config import load_config
from fanet_next.experiment.assemble import build_env, build_model, build_policy, feature_schema
from fanet_next.model.runner import replay
from fanet_next.training.advantages import compute_stream_advantages
from fanet_next.training.collector import RolloutCollector, SeedStream
from fanet_next.training.normalize import ReturnScaler
from fanet_next.training.ppo import PPO

CFG = load_config("configs/smoke.toml", ["scenario.horizon=30", "scenario.flow_rate_pps=60.0"])


def collect(n_envs=3, cycles=12, seed=0):
    torch.manual_seed(seed)
    model = build_model(CFG, feature_schema(CFG))
    policy = build_policy(CFG, "ppo", model=model, seed=seed)
    envs = [build_env(CFG, run_id=f"t{e}") for e in range(n_envs)]
    col = RolloutCollector(envs, policy, seeds=SeedStream(123), scaler=ReturnScaler(n_envs, 0.99),
                           rollout_cycles=cycles)
    return model, policy, col, col.collect()


def test_replay_reproduces_sampled_logp_and_values_exactly():
    model, _, _, res = collect()
    recs = res.records
    assert any(r.num_actions == 0 for r in recs), "want an empty-candidate cycle in the sample"
    assert max(r.num_actions for r in recs) >= 2, "want multi-step cycles"
    rp = replay(model, [r.micro for r in recs])
    old = np.concatenate([r.micro.logp for r in recs])
    assert np.allclose(rp.logp[rp.act_mask].detach().numpy(), old, atol=1e-5)
    oldv = np.concatenate([r.micro.values for r in recs])
    assert np.allclose(rp.values[rp.state_mask].detach().numpy(), oldv, atol=1e-5)
    # empty-candidate cycles: no actor row, one boundary value row
    for b, r in enumerate(recs):
        assert int(rp.act_mask[b].sum()) == r.num_actions
        assert int(rp.state_mask[b].sum()) == r.num_actions + 1


def test_gradients_reach_every_model_component():
    model, _, _, res = collect()
    rp = replay(model, [r.micro for r in res.records])
    loss = -rp.logp[rp.act_mask].sum() + rp.values[rp.state_mask].pow(2).sum()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_summary_depends_on_action_prefix():
    model, _, _, res = collect()
    rec = next(r for r in res.records if r.num_actions >= 2 and
               r.micro.masks[0][r.micro.actions[1]])  # second action was feasible first
    a = rec.micro.actions
    base = replay(model, [rec.micro]).logp[0, 1].item()
    # same second action, but reached without the first selection in the summary
    from dataclasses import replace
    alt = replace(rec.micro, actions=a[1:2], masks=rec.micro.masks[[0, -1]],
                  micro=rec.micro.micro[[0, -1]], logp=rec.micro.logp[1:2],
                  values=rec.micro.values[[0, -1]], cand_dyn=rec.micro.cand_dyn[[0, -1]])
    other = replay(model, [alt]).logp[0, 0].item()
    assert base != pytest.approx(other)


def test_ppo_update_first_ratio_is_one_and_parameters_move():
    model, _, _, res = collect()
    for s in res.streams:
        compute_stream_advantages(s, 0.99, 0.95)
    ppo = PPO(model, epochs=1, minibatch_cycles=10_000, seed=0)
    before = [p.detach().clone() for p in model.parameters()]
    stats = ppo.update(res.records)
    assert stats["approx_kl"] == pytest.approx(0.0, abs=1e-6)  # single full-batch step: ratio 1
    assert stats["clip_frac"] == 0.0
    assert any(not torch.equal(b, p) for b, p in zip(before, model.parameters()))
    assert stats["actions"] == sum(r.num_actions for r in res.records)


def test_rollout_streams_end_with_bootstrap_markers_and_scaler_is_finite():
    _, _, col, res = collect(cycles=40)  # horizon 30 → every env truncates once
    for s in res.streams:
        assert s[-1].end in {"cut", "truncated"}
        assert all(r.end == "continue" for r in s[:-1] if r.end != "truncated")
    truncs = [r for r in res.records if r.end == "truncated"]
    assert len(truncs) == 3 and all(np.isfinite(r.bootstrap) for r in truncs)
    assert all(np.isfinite(r.reward) for r in res.records)
    assert len(res.episodes) == 3 and not res.errors


def test_imitation_records_replay_and_fit_raises_teacher_likelihood():
    from fanet_next.training.imitation import fit, teacher_records

    torch.manual_seed(0)
    model = build_model(CFG, feature_schema(CFG))
    teacher = build_policy(CFG, "longest_queue")
    envs = [build_env(CFG, run_id=f"i{e}") for e in range(2)]
    recs = teacher_records(envs, teacher, [7, 8], cycles=15, gamma=0.99)
    for r in recs:  # masks reproduce a legal teacher trajectory ending in a maximal set
        for j, a in enumerate(r.micro.actions):
            assert r.micro.masks[j][a]
        assert not r.micro.masks[-1].any()
    before = replay(model, [r.micro for r in recs]).logp.sum().item()
    fit(model, recs, epochs=15, minibatch=64, lr=3e-3)
    after = replay(model, [r.micro for r in recs]).logp.sum().item()
    assert after > before


def test_sequence_ratio_update_runs_and_starts_at_ratio_one():
    model, _, _, res = collect()
    for s in res.streams:
        compute_stream_advantages(s, 0.99, 0.95, credit="cycle")
    ppo = PPO(model, epochs=1, minibatch_cycles=10_000, seed=0, ratio="sequence")
    stats = ppo.update(res.records)
    assert stats["approx_kl"] == pytest.approx(0.0, abs=1e-6) and stats["clip_frac"] == 0.0
    with pytest.raises(ValueError):
        PPO(model, ratio="joint")


def test_critic_without_a_plan_summary_cannot_tell_equal_size_complete_plans_apart():
    """Final method (set_summary none): at a complete plan the critic sees only the cycle
    encoding and the number of chosen links; with critic_set_summary it sees the links."""
    from dataclasses import replace
    from fanet_next.config import deep_merge
    from fanet_next.model.runner import replay as rp_
    blind = deep_merge(CFG, {"model": {"set_summary": {"type": "none"}}})
    sees = deep_merge(blind, {"model": {"critic_set_summary": {"type": "gated_mean"}}})
    for cfg, expect_equal in ((blind, True), (sees, False)):
        torch.manual_seed(1)
        model = build_model(cfg, feature_schema(cfg))
        policy = build_policy(cfg, "ppo", model=model, seed=0)
        envs = [build_env(cfg, run_id="t0")]
        col = RolloutCollector(envs, policy, seeds=SeedStream(7), scaler=ReturnScaler(1, 0.99),
                               rollout_cycles=30)
        res = col.collect()
        # a cycle where a different single first choice also completes a one-link plan is rare;
        # compare two complete one-link plans built from the first record with >= 2 candidates
        rec = next(r for r in res.records if r.num_actions >= 1 and r.micro.masks[0].sum() >= 2)
        first = int(rec.micro.actions[0])
        other = int(np.flatnonzero(rec.micro.masks[0])[np.flatnonzero(rec.micro.masks[0]) != first][0])
        empty = np.zeros_like(rec.micro.masks[0])
        def plan(a):
            m = rec.micro
            return replace(m, actions=np.array([a]), masks=np.stack([m.masks[0], empty]),
                           micro=np.stack([m.micro[0], m.micro[-1]]), logp=m.logp[:1],
                           values=m.values[[0, -1]],
                           cand_dyn=None if m.cand_dyn is None else m.cand_dyn[[0, -1]])
        v = rp_(model, [plan(first), plan(other)]).values
        same = torch.isclose(v[0, 1], v[1, 1], atol=1e-6).item()
        assert same == expect_equal
