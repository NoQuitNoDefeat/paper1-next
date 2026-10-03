"""Contract tests every registered implementation must pass.

Adding an implementation to a slot automatically adds it to these tests.
"""

import numpy as np
import pytest
import torch

from fanet_next.config import deep_merge, load_config
from fanet_next.experiment.assemble import build_env, build_model, build_policy, feature_schema
from fanet_next.model.components import (COMM_ENCODER, INTERACTION_ENCODER, LIFT, SET_SUMMARY)
from fanet_next.model.runner import replay
from fanet_next.physics import set_sinr
from fanet_next.policy import POLICY
from fanet_next.registry import ConfigError
from fanet_next.scenario import CHANNEL, SCENARIO
from fanet_next.training.collector import RolloutCollector, SeedStream
from fanet_next.training.normalize import ReturnScaler

CFG = load_config("configs/smoke.toml", ["scenario.horizon=20", "scenario.flow_rate_pps=60.0"])


def run_policy(cfg, policy, cycles=15, seed=3):
    env = build_env(cfg, run_id="c", build_graph=policy.needs_graph)
    inp = env.reset(seed)
    for _ in range(cycles):
        out = policy.act([inp])[0]
        links = inp.problem.links[out.actions] if out.actions else np.zeros((0, 2), int)
        nodes = links.flatten()
        assert len(nodes) == len(set(nodes)), "half duplex violated"
        if policy.maximal_plans:
            assert inp.controller.done, "plan must be maximal under the controller"
        if cfg.get("constraints", {}).get("interference", "full_sinr") == "full_sinr" and len(links):
            p = inp.problem
            assert np.all(set_sinr(links, p.gain, p.power, p.noise) >= p.threshold * (1 - 1e-9))
        inp = env.step(out.actions).next_input
    b = env.backend
    assert b.totals["born"] == b.totals["delivered"] + b.totals["terminated"] + sum(b.in_system())
    return env


@pytest.mark.parametrize("name", [n for n in POLICY.names()])
def test_every_policy_produces_valid_maximal_plans(name):
    policy = build_policy(CFG, name, seed=0)
    run_policy(CFG, policy)


@pytest.mark.parametrize("interference", ["full_sinr", "pairwise", "new_link_only", "none"])
def test_policies_run_under_every_constraint(interference):
    cfg = deep_merge(CFG, {"constraints": {"interference": interference}})
    run_policy(cfg, build_policy(cfg, "longest_queue"))


@pytest.mark.parametrize("channel", CHANNEL.names())
def test_every_channel_model_conserves_packets(channel):
    cfg = deep_merge(CFG, {"backend": {"channel": {"type": channel}}})
    env = run_policy(cfg, build_policy(cfg, "longest_queue"), cycles=19)
    if channel == "ideal":
        assert env.metrics.failed_links == 0  # feasible under exact CSI ⇒ succeeds


@pytest.mark.parametrize("name", SCENARIO.names())
def test_every_scenario_source_is_well_formed(name, tmp_path):
    spec = {"type": name}
    if name == "fixed":
        spec.update(positions=[[0, 0, 100], [100, 0, 100]], births=[(0.01, 0, 1)], horizon=3)
    if name == "mixture":
        spec.update(base_type="random", num_nodes=5, horizon=20,
                    components=[{}, {"speed_mps": [0.0, 5.0], "area_m": [150.0, 300.0]}])
    if name == "trace":
        t = np.arange(0.0, 30.0, 0.2)
        np.savez(tmp_path / "toy.npz", t=t, pos=np.stack([np.stack([t, 50.0 * i + t * 0, 100 + 0 * t], 1)
                                                         for i in range(5)], 1))
        spec.update(dir=str(tmp_path), train=["toy"], num_nodes=4, horizon=40)
    sc = SCENARIO.build(spec).make(0)
    assert sc.positions(0).shape == (sc.num_nodes, 3)
    assert sc.positions(sc.horizon).shape == (sc.num_nodes, 3)
    ql = sc.queue_links
    assert len(ql) == len({tuple(x) for x in ql}) and np.all(ql[:, 0] != ql[:, 1])
    for k in range(sc.horizon):
        for b in sc.births(k):
            assert k * sc.cycle_length < b.time <= (k + 1) * sc.cycle_length + 1e-12
            assert b.src != b.dst


VARIANTS = ([("comm_encoder", n) for n in COMM_ENCODER.names()]
            + [("lift", n) for n in LIFT.names()]
            + [("interaction_encoder", n) for n in INTERACTION_ENCODER.names()]
            + [("set_summary", n) for n in SET_SUMMARY.names()]
            + [("share_trunk", False), ("candidate_dynamics", True), ("critic_set_summary", "gated_mean")])


@pytest.mark.parametrize("key,value", VARIANTS)
def test_every_model_variant_samples_replays_and_trains(key, value):
    model_cfg = dict(CFG["model"])
    model_cfg[key] = value if key in ("share_trunk", "candidate_dynamics") else {"type": value}
    cfg = {**CFG, "model": model_cfg}
    torch.manual_seed(0)
    model = build_model(cfg, feature_schema(cfg))
    policy = build_policy(cfg, "ppo", model=model)
    envs = [build_env(cfg, run_id=f"v{e}") for e in range(2)]
    col = RolloutCollector(envs, policy, seeds=SeedStream(5), scaler=ReturnScaler(2, 0.99),
                           rollout_cycles=10)
    recs = col.collect().records
    rp = replay(model, [r.micro for r in recs])
    assert np.allclose(rp.logp[rp.act_mask].detach().numpy(),
                       np.concatenate([r.micro.logp for r in recs]), atol=1e-5)
    (-rp.logp[rp.act_mask].sum() + rp.values[rp.state_mask].pow(2).sum()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_config_errors_are_explicit():
    with pytest.raises(ConfigError, match="available"):
        build_policy(CFG, "no_such_policy")
    with pytest.raises(ConfigError, match="unknown parameters"):
        POLICY.build({"type": "hol_weighted", "typo": 1})
    cfg = load_config("configs/smoke.toml", ["model.lift.type=context_only", "training.lr=1e-3"])
    assert cfg["model"]["lift"] == {"type": "context_only"}  # type change drops stale params
    assert cfg["training"]["lr"] == 1e-3 and cfg["scenario"]["num_nodes"] == 8


def test_model_and_observation_mismatch_is_rejected():
    from fanet_next.experiment.assemble import check_compatibility
    from fanet_next.observation.graph import FeatureSchema

    schema = feature_schema(CFG)
    small = FeatureSchema(schema.node[:-1], schema.edge, schema.cand, schema.inter, schema.glob)
    policy = build_policy(CFG, "ppo", model=build_model(CFG, small))
    with pytest.raises(ConfigError, match="features"):
        check_compatibility(build_env(CFG), policy)
    with pytest.raises(ConfigError, match="dual graph"):
        check_compatibility(build_env(CFG, build_graph=False), build_policy(CFG, "ppo"))


def test_drain_evaluation_keeps_the_same_scene_then_stops_traffic():
    from fanet_next.experiment.evaluate import eval_scenario

    cfg = load_config("configs/base.toml", ["scenario.horizon=40"])
    base = SCENARIO.build(cfg["scenario"]).make(10_000_003)
    drained = SCENARIO.build(eval_scenario(cfg, drain_cycles=20)).make(10_000_003)
    assert drained.horizon == 60 and drained.traffic_done(40)
    for k in range(41):
        assert np.array_equal(base.positions(k), drained.positions(k))
    for k in range(40):
        assert base.births(k) == drained.births(k)
    assert all(not drained.births(k) for k in range(40, 60))


@pytest.mark.parametrize("name", __import__("fanet_next.reward.standard", fromlist=["REWARD"]).REWARD.names())
def test_every_reward_returns_the_four_components(name):
    from fanet_next.reward.standard import REWARD

    cfg = deep_merge(CFG, {"reward": {"type": name}})
    env = build_env(cfg, build_graph=False)
    env.reset(1)
    for _ in range(5):
        tr = env.step([])
        assert set(tr.reward.parts) == {"service", "queue", "delay", "violation"}
        assert np.isfinite(tr.reward.total)


def test_select_command_applies_reliability_first_rule(tmp_path):
    from fanet_next.experiment.select import select_checkpoint
    from fanet_next.experiment.train import TrainingRun

    cfg = load_config("configs/smoke.toml", ["training.keep_every=1", "training.checkpoint_every=1"])
    TrainingRun(cfg, tmp_path).train(2, verbose=False)
    out = select_checkpoint(tmp_path, episodes=2, drain=5, num_envs=2)
    assert len(out["candidates"]) == 2 and out["selected"]["iteration"] in (1, 2)
    if out["any_eligible"]:
        assert out["selected"]["eligible"]
    assert (tmp_path / "selection.json").exists()


def test_switch_backend_keeps_environment_parameters():
    """``--set backend.type=`` clears the section (default routing!); ``switch_backend`` keeps it."""
    from fanet_next.config import apply_override, load_config
    from fanet_next.experiment.assemble import switch_backend

    cfg = load_config("configs/protocol_final.toml")
    switched = switch_backend(cfg, "ns3")
    assert switched["backend"] == {**cfg["backend"], "type": "ns3"}
    assert cfg["backend"]["type"] == "lightweight"  # input untouched
    apply_override(switched, "backend.motion=false")
    assert switched["backend"]["routing"]["snr_margin_db"] == 6.0
    assert switched["backend"]["motion"] is False
    cleared = load_config("configs/protocol_final.toml", ["backend.type=\"ns3\""])
    assert "routing" not in cleared["backend"]
