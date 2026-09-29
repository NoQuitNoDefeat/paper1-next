"""Learned baselines from prior work, adapted to this problem (E11); see docs/decisions.md §9.

Each is a registered policy that owns its network plus a learner implementing the
original paper's training; ``make_learner`` wires the learner for a training run.
"""

from __future__ import annotations

from ..training.collector import SeedStream
from . import grlinq, oneshot, zhao_gcn  # noqa: F401  (register the policies)
from .common import EnvStream, LearnedBaseline, Learner
from .grlinq import GRLinQLearner
from .oneshot import OneShotLearner
from .zhao_gcn import ZhaoGcnLearner

__all__ = ["EnvStream", "LearnedBaseline", "Learner", "make_learner"]


def make_learner(kind: str, policy, envs, cfg: dict, seed_base: int, seed: int) -> Learner:
    """Learner for ``kind``; the one-shot baseline takes the primary method's trainer,
    training and imitation settings, the others their paper's settings (``baseline.learner``)."""
    stream = EnvStream(envs, SeedStream(seed_base))
    t = cfg.get("training", {})
    params = dict(cfg.get("baseline", {}).get("learner", {}))
    common = {"rollout_cycles": int(t.get("rollout_cycles", 128)), "seed": seed}
    if kind == "zhao_gcn":
        return ZhaoGcnLearner(policy, stream, **common, **params)
    if kind == "grlinq":
        return GRLinQLearner(policy, stream, **common, **params)
    if kind == "oneshot_ppo":
        tr = {k: v for k, v in cfg.get("trainer", {}).items() if k != "type"}
        return OneShotLearner(policy, stream, **common, gamma=float(t.get("gamma", 0.99)),
                              gae_lambda=float(t.get("gae_lambda", 0.95)), lr=float(t.get("lr", 1e-4)),
                              reward_norm=bool(t.get("reward_norm", True)),
                              imitation=cfg.get("imitation"), **tr, **params)
    raise ValueError(f"no learner for baseline {kind!r}")
