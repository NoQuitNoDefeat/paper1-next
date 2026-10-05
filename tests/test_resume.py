"""Training resume within this version is exact at iteration boundaries (CPU)."""

import json

import torch

from fanet_next.config import load_config
from fanet_next.experiment.train import TrainingRun


def test_resume_matches_uninterrupted_training(tmp_path):
    cfg = load_config("configs/smoke.toml", ["scenario.horizon=10", "training.rollout_cycles=6"])
    straight = TrainingRun(cfg, tmp_path / "a")
    straight.train(3, verbose=False)

    first = TrainingRun(cfg, tmp_path / "b")
    first.train(2, verbose=False)
    resumed = TrainingRun.resume(tmp_path / "b")
    assert resumed.progress["iteration"] == 2
    resumed.train(3, verbose=False)

    for (name, a), (_, b) in zip(straight.model.state_dict().items(),
                                 resumed.model.state_dict().items()):
        assert torch.equal(a, b), name
    assert straight.progress["cycles"] == resumed.progress["cycles"]
    logs = [json.loads(line) for line in (tmp_path / "b" / "train_log.jsonl").read_text().splitlines()]
    assert any(e.get("event") == "resume" for e in logs)
    meta = json.loads((tmp_path / "a" / "meta.json").read_text())
    assert meta["manifest"]["interference"] == "full_sinr(primary)"


def test_resume_with_fresh_envs_keeps_learning_state(tmp_path):
    cfg = load_config("configs/smoke.toml", ["scenario.horizon=10", "training.rollout_cycles=6"])
    run = TrainingRun(cfg, tmp_path / "c")
    run.train(1, verbose=False)
    again = TrainingRun.resume(tmp_path / "c", fresh_envs=True)
    for (_, a), (_, b) in zip(run.model.state_dict().items(), again.model.state_dict().items()):
        assert torch.equal(a, b)
    again.train(2, verbose=False)
    assert again.progress["iteration"] == 2


def test_imitation_env_count_keeps_the_imitation_start(tmp_path):
    """imitation.num_envs (E19): more PPO environments, same imitation data and start."""
    import torch

    from fanet_next.config import load_config
    from fanet_next.experiment.train import TrainingRun

    sets = ["imitation.cycles=12", "imitation.epochs=1", "scenario.horizon=30"]
    base = load_config("configs/protocol_final.toml", sets + ["training.num_envs=2"])
    more = load_config("configs/protocol_final.toml", sets + ["training.num_envs=4", "imitation.num_envs=2"])
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a, b = TrainingRun(base, tmp_path / "a"), TrainingRun(more, tmp_path / "b")
    ea, eb = a.imitate(), b.imitate()
    assert ea["records"] == eb["records"] and ea["return_scale"] == eb["return_scale"]
    for (ka, va), (kb, vb) in zip(a.model.state_dict().items(), b.model.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)
