"""Diagnostic D2 (no method change): at one network state, do different complete plans of the
same size lead to different average outcomes, and does the critic tell them apart?

For a checkpoint (selected one of a run) and dev scenes far from the selection seeds:

1. run the policy (sampling, as in training) to a random cycle t0 of an episode; that state is
   kept (deep copy of the environment);
2. sample K plans there from the policy; keep the plans of the most common size (complete
   plans: no feasible candidate left), up to M distinct ones, with the critic's value at the
   complete plan V_k(p) and at the cycle start V_0;
3. for every kept plan p and repetition r = 1..R: copy the state, execute p, then let the policy
   decide normally (sampling) until the episode ends; the policy's random generator is seeded
   with r in every branch (paired random numbers; arrivals and mobility are fixed per scene);
   G(p, r) = sum_h gamma^h clip(raw reward_h / s, +-10) with the reward scale s frozen at the
   checkpoint's value (no running update, no carried-over accumulator), i.e. the critic's units;
4. R more branches start with a policy-sampled plan: their mean estimates V_0's target.

Episodes are cut at their horizon (no bootstrap, so the targets do not contain the critic);
t0 is drawn from [60, 150] so the cut tail weighs at most gamma^350 ~ 0.03.

Reported: how often the policy produces several distinct plans of one size at a state, how far
their mean outcomes Q(p) lie apart (with paired standard errors), whether the critic's V_k
separates them (rank correlation), and the critic's bias and RMSE against the targets.

    .venv/bin/python tools/critic_counterfactual.py results/e6/no_set_summary-s0 --states 40 --workers 6
"""

from __future__ import annotations

import argparse
import copy
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from confirm import ROOT
from fanet_next.experiment.assemble import build_env, policy_from_checkpoint
from fanet_next.loop import EndType
from fanet_next.training.checkpoint import load_checkpoint

DEV_BASE = 10_000_000 + 500  # dev scenes far from the selection (0-15) and ablation (16-47) seeds


def _setup(run: str):
    sel = json.loads((ROOT / run / "selection.json").read_text())["selected"]["checkpoint"]
    data = load_checkpoint(ROOT / sel)
    cfg = data["config"]
    rms = data["collector"]["scaler"]["rms"]
    scale = float(np.sqrt(rms["var"] + 1e-8)) if cfg.get("training", {}).get("reward_norm", True) else 1.0
    policy = policy_from_checkpoint(cfg, data)
    gamma = float(cfg.get("training", {}).get("gamma", 0.99))
    return cfg, policy, scale, gamma


def _rollout(env, policy, first_actions, seed: int, scale: float, gamma: float) -> float:
    """Execute ``first_actions`` (or a policy sample when None), then the policy to episode end."""
    policy.generator.manual_seed(seed)
    g, disc = 0.0, 1.0
    actions = first_actions
    while True:
        if actions is None:
            actions = policy.act([env.current], mode="sample")[0].actions
        tr = env.step(list(actions))
        g += disc * float(np.clip(tr.reward.total / scale, -10.0, 10.0))
        disc *= gamma
        if tr.end is not EndType.CONTINUE:
            return g
        actions = None


def probe_state(args) -> dict | None:
    run, idx, k_samples, m_plans, reps = args
    cfg, policy, scale, gamma = _setup(run)
    rng = np.random.default_rng(idx)
    env = build_env(cfg, run_id=f"d2-{idx}")
    env.reset(DEV_BASE + idx, episode=idx)
    t0 = int(rng.integers(60, 151))
    policy.generator.manual_seed(900_000 + idx)
    for _ in range(t0):  # reach the state with the policy itself
        tr = env.step(list(policy.act([env.current], mode="sample")[0].actions))
        if tr.end is not EndType.CONTINUE:
            return None
    plans: dict[tuple, dict] = {}
    v0 = None
    for i in range(k_samples):
        clone = copy.deepcopy(env)
        policy.generator.manual_seed(700_000 + 1000 * idx + i)
        out = policy.act([clone.current], mode="sample")[0]
        vals = out.record["micro"].values
        v0 = float(vals[0])
        key = tuple(sorted(int(a) for a in out.actions))
        p = plans.setdefault(key, {"order": list(map(int, out.actions)), "count": 0, "vk": float(vals[-1])})
        p["count"] += 1
    sizes = [len(k) for k, p in plans.items() for _ in range(p["count"])]
    k_star = max(set(sizes), key=sizes.count)
    same = sorted(((k, p) for k, p in plans.items() if len(k) == k_star), key=lambda kp: -kp[1]["count"])[:m_plans]
    res = {"index": idx, "t0": t0, "k_star": k_star, "distinct_plans": len(plans),
           "distinct_same_size": sum(len(k) == k_star for k in plans), "v0": v0, "plans": []}
    for key, p in same:
        g = [_rollout(copy.deepcopy(env), policy, p["order"], 50_000 + r, scale, gamma) for r in range(reps)]
        res["plans"].append({"links": list(key), "count": p["count"], "vk": p["vk"], "g": g})
    res["policy_g"] = [_rollout(copy.deepcopy(env), policy, None, 50_000 + r, scale, gamma) for r in range(reps)]
    return res


def summarise(states: list[dict]) -> dict:
    multi = [s for s in states if len(s["plans"]) >= 2]
    gaps, gap_se, sig, spears, vk_spread = [], [], 0, [], []
    for s in multi:
        g = np.array([p["g"] for p in s["plans"]])  # (plans, reps), paired by rep
        q = g.mean(1)
        vk = np.array([p["vk"] for p in s["plans"]])
        vk_spread.append(float(np.ptp(vk)))
        for i in range(len(q)):
            for j in range(i + 1, len(q)):
                d = g[i] - g[j]
                se = d.std(ddof=1) / np.sqrt(len(d))
                gaps.append(abs(d.mean()))
                gap_se.append(se)
                sig += abs(d.mean()) > 2 * se
        if np.ptp(vk) > 1e-6 and len(q) >= 3:
            rq, rv = np.argsort(np.argsort(q)), np.argsort(np.argsort(vk))
            spears.append(float(np.corrcoef(rq, rv)[0, 1]))
    v0 = np.array([s["v0"] for s in states])
    qpol = np.array([np.mean(s["policy_g"]) for s in states])
    vk_all = np.array([p["vk"] for s in states for p in s["plans"]])
    q_all = np.array([np.mean(p["g"]) for s in states for p in s["plans"]])
    return {
        "states": len(states), "states_with_several_same_size_plans": len(multi),
        "share_with_several": len(multi) / max(len(states), 1),
        "pairs": len(gaps), "mean_abs_gap": float(np.mean(gaps)) if gaps else None,
        "median_abs_gap": float(np.median(gaps)) if gaps else None,
        "mean_pair_se": float(np.mean(gap_se)) if gap_se else None,
        "share_pairs_significant": sig / max(len(gaps), 1),
        "critic_vk_spread_mean": float(np.mean(vk_spread)) if vk_spread else None,
        "critic_rank_corr_mean": float(np.mean(spears)) if spears else None,
        "v0_bias": float(np.mean(v0 - qpol)), "v0_rmse": float(np.sqrt(np.mean((v0 - qpol) ** 2))),
        "vk_bias": float(np.mean(vk_all - q_all)), "vk_rmse": float(np.sqrt(np.mean((vk_all - q_all) ** 2))),
        "target_std_across_states": float(qpol.std()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--states", type=int, default=40)
    ap.add_argument("--samples", type=int, default=16, help="plans sampled per state")
    ap.add_argument("--plans", type=int, default=4, help="same-size plans rolled out per state")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    jobs = [(a.run, i, a.samples, a.plans, a.reps) for i in range(a.states)]
    with Pool(a.workers) as pool:
        states = [s for s in pool.map(probe_state, jobs) if s is not None]
    summary = summarise(states)
    out = Path(a.out) if a.out else ROOT / "results/diagnostics" / f"d2_{Path(a.run).name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run": a.run, "summary": summary, "states": states}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
