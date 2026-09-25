"""Evaluation on fixed, split-separated seed sets."""

from __future__ import annotations

import numpy as np

from ..config import deep_merge
from ..loop import run_episodes
from ..policy.base import Policy
from ..reward.metrics import merge_summaries
from .assemble import DEV_SEED_BASE, TEST_SEED_BASE, build_env, check_compatibility

SPLITS = {"dev": DEV_SEED_BASE, "test": TEST_SEED_BASE}


def split_seeds(split: str, episodes: int, offset: int = 0) -> list[int]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}")
    return [SPLITS[split] + offset + i for i in range(episodes)]


def eval_scenario(cfg: dict, drain_cycles: int = 0, overrides: dict | None = None) -> dict:
    """Training scenario with evaluation-only changes.

    ``drain_cycles > 0`` stops traffic at the training horizon and keeps running
    for that many cycles, so (almost) every packet's final fate is known.
    """
    sc = deep_merge(cfg["scenario"], cfg.get("evaluation", {}).get("scenario_overrides", {}))
    sc = deep_merge(sc, overrides or {})
    if drain_cycles:
        stop = int(sc.get("horizon", 500))
        sc = deep_merge(sc, {"traffic_stop_cycle": stop, "horizon": stop + int(drain_cycles)})
    return sc


def paired(rows_a: list[dict], rows_b: list[dict], key: str) -> dict:
    """Per-seed paired difference a - b: mean, 95% CI (t), wins of a."""
    a = np.array([r[key] for r in rows_a], dtype=float)
    b = np.array([r[key] for r in rows_b], dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    n = len(d)
    if n < 2:
        return {"n": n, "mean": float(d.mean()) if n else float("nan")}
    from math import sqrt

    t = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 8: 2.36, 10: 2.26, 16: 2.13, 32: 2.04}
    tval = t.get(n, 2.0 if n > 32 else 2.3)
    half = tval * d.std(ddof=1) / sqrt(n)
    return {"n": n, "mean": float(d.mean()), "ci95": [float(d.mean() - half), float(d.mean() + half)],
            "wins": int((d > 0).sum())}


def evaluate(cfg: dict, policy: Policy, *, split: str = "dev", episodes: int = 8,
             mode: str = "greedy", num_envs: int = 8, scenario: dict | None = None,
             drain_cycles: int = 0, seed_offset: int = 0, on_episode=None) -> dict:
    """Mean metrics plus per-episode rows; the same seeds give the same scenes for every policy."""
    scenario = scenario or eval_scenario(cfg, drain_cycles)
    envs = [build_env(cfg, run_id=f"eval-{split}-{i}", build_graph=policy.needs_graph,
                      scenario_override=scenario) for i in range(min(num_envs, episodes))]
    manifest = check_compatibility(envs[0], policy)
    if hasattr(policy, "seed") and not policy.learnable:
        policy.seed(12345)
    seeds = split_seeds(split, episodes, seed_offset)
    rows = run_episodes(policy, envs, seeds, mode=mode, on_episode=on_episode)
    mean = merge_summaries(rows)
    for key in ("delivery_ratio", "e2e_delay_mean_s", "reward_mean"):
        vals = np.array([r[key] for r in rows], dtype=float)
        mean[f"{key}_std"] = float(np.nanstd(vals))
    return {"split": split, "mode": mode, "episodes": episodes, "seeds": seeds,
            "drain_cycles": drain_cycles, "scenario": scenario, "manifest": manifest,
            "mean": mean, "rows": rows}
