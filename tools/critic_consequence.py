"""Diagnostic D3 (no method change): does the critic judge the consequences of a plan?

The actor's advantage for a cycle's actions is dominated (lambda = 0.95 per micro step, about two
actions per cycle) by the one-step estimate r + gamma V(next cycle start): the cycle's reward plus
the critic's value of the state the plan leads to.  D2 showed that the value at the complete plan
cannot tell same-size plans apart; this asks whether the next-state value does the job instead.

For a checkpoint (selected one of a run) and dev scenes far from the selection, ablation and D2
seeds:

1. run the policy (sampling) to a random cycle t0 in [60, 150]; keep that state;
2. plans at that state: the most frequent distinct plans among K policy samples (any size), plus
   the longest-queue teacher's plan and one uniformly random complete plan when they differ;
3. per plan and repetition r = 1..R: copy the state, execute the plan, then the policy (sampling,
   random generator seeded with r in every branch) to the end of the episode; record the
   normalised rewards (scale frozen at the checkpoint's value, clip +-10), the critic's value at
   each later cycle start, and the discounted reward components (service, queue, delay, violation);
4. per pair of plans at a state, the true outcome difference is the mean paired difference of the
   returns; it is compared with three estimates:
   * myopic: the difference of the first cycle's reward only (no critic);
   * critic, one step: the difference of r + gamma V(next) (what the advantage mostly uses);
   * critic at the complete plan: the difference of V_k (policy plans only; D2);
   and with n-step estimates (rewards of n cycles + critic value after them), whose rollouts are
   split from the ones that give the true difference.

Reported (per subset of pairs): the true gap scale (maximum likelihood with per-pair noise), the
calibration slope (true difference per unit of estimated difference; 1 = usable, 0 = unrelated),
the noise-corrected correlation (its square: share of the true differences the estimate captures),
bootstrap intervals over states; and which reward components make up the true differences and
how well the critic's one-step estimate tracks each.

    .venv/bin/python tools/critic_consequence.py results/e6/no_set_summary-s0 --workers 4
"""

from __future__ import annotations

import argparse
import copy
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from confirm import ROOT
from critic_counterfactual import _setup
from fanet_next.config import apply_override
from fanet_next.experiment.assemble import build_env, build_policy
from fanet_next.loop import EndType

DEV_BASE = 10_000_000 + 600  # dev scenes beyond selection (0-15), ablation (16-47) and D2 (500-539)
COMPONENTS = ("service", "queue", "delay", "violation")
STEPS = (1, 2, 4, 8, 16)  # n of the n-step estimates


def _rollout(env, policy, plan, seed: int, scale: float, gamma: float, weights: dict, horizon: int):
    """Execute ``plan``, then the policy to episode end.  Returns the discounted return, the first
    ``horizon`` normalised rewards, the critic values at cycle starts 1..horizon, the discounted
    normalised reward components."""
    policy.generator.manual_seed(seed)
    g, disc, h = 0.0, 1.0, 0
    rew, vals, comps = [], [], np.zeros(len(COMPONENTS))
    actions = plan
    while True:
        if actions is None:
            out = policy.act([env.current], mode="sample")[0]
            if h <= horizon:
                vals.append(float(out.record["micro"].values[0]))
            actions = out.actions
        tr = env.step(list(actions))
        x = float(np.clip(tr.reward.total / scale, -10.0, 10.0))
        g += disc * x
        if h < horizon:
            rew.append(x)
        comps += disc * np.array([weights[k] * tr.reward.parts[k] for k in COMPONENTS]) / scale
        disc *= gamma
        h += 1
        if tr.end is not EndType.CONTINUE:
            return g, rew, vals, comps.tolist(), tr.end is EndType.TERMINATED
        actions = None


def reach_state(cfg, policy, idx: int, base: int = DEV_BASE, prefix: str = "d3", t0_range=(60, 150)):
    """Run the policy (sampling) to a random cycle t0 in ``t0_range`` (inclusive; default [60, 150])
    of dev scene ``base + idx``.  Returns (env, rng, t0), or None when the episode ended first."""
    rng = np.random.default_rng(1_000 + idx)
    env = build_env(cfg, run_id=f"{prefix}-{idx}")
    env.privileged = cfg.get("training", {}).get("privileged_critic")  # E21: the critic's training input
    env.reset(base + idx, episode=idx)
    t0 = int(rng.integers(t0_range[0], t0_range[1] + 1))
    policy.generator.manual_seed(800_000 + idx)
    for _ in range(t0):  # reach the state with the policy itself
        tr = env.step(list(policy.act([env.current], mode="sample")[0].actions))
        if tr.end is not EndType.CONTINUE:
            return None
    return env, rng, t0


def candidate_plans(env, policy, cfg, idx: int, rng, k_samples: int, m_policy: int):
    """The most frequent distinct plans among ``k_samples`` policy samples (at most ``m_policy``),
    plus the longest-queue plan and one uniformly random complete plan when they differ.
    Returns (plans keyed by the sorted link set, V0, number of distinct sampled plans)."""
    sampled: dict[tuple, dict] = {}
    v0 = None
    for i in range(k_samples):
        clone = copy.deepcopy(env)
        policy.generator.manual_seed(600_000 + 1_000 * idx + i)
        out = policy.act([clone.current], mode="sample")[0]
        vals = out.record["micro"].values
        v0 = float(vals[0])
        key = tuple(sorted(int(a) for a in out.actions))
        p = sampled.setdefault(key, {"order": [int(a) for a in out.actions], "source": "policy", "count": 0,
                                     "vk": float(vals[-1])})
        p["count"] += 1
    plans = dict(sorted(sampled.items(), key=lambda kv: -kv[1]["count"])[:m_policy])
    clone = copy.deepcopy(env)
    teacher = build_policy(cfg, "longest_queue", seed=0)
    lq = [int(a) for a in teacher.act([clone.current], mode="greedy")[0].actions]
    clone = copy.deepcopy(env)
    ctl = clone.constraints.start(clone.current.problem)
    while not ctl.done:
        ctl.step(int(rng.choice(np.flatnonzero(ctl.mask))))
    for source, order in (("lq", lq), ("random", [int(a) for a in ctl.selected])):
        key = tuple(sorted(order))
        if key in plans:
            plans[key].setdefault("also", []).append(source)
        else:
            plans[key] = {"order": order, "source": source, "count": sampled.get(key, {}).get("count", 0),
                          "vk": None}
    return plans, v0, len(sampled)


def probe_state(args) -> dict | None:
    run, idx, k_samples, m_policy, reps, horizon, *rest = args
    t0_range, base, sets = rest if rest else ((60, 150), DEV_BASE, [])
    cfg, policy, scale, gamma = _setup(run)
    for item in sets:
        apply_override(cfg, item)
    reached = reach_state(cfg, policy, idx, base=base, t0_range=t0_range)
    if reached is None:
        return None
    env, rng, t0 = reached
    weights = {k: w * float(getattr(env.reward, "scale", 1.0)) for k, w in env.reward.weights().items()}
    plans, v0, n_sampled = candidate_plans(env, policy, cfg, idx, rng, k_samples, m_policy)
    res = {"index": idx, "t0": t0, "v0": v0, "distinct_sampled": n_sampled, "plans": []}
    for key, p in plans.items():
        runs = [_rollout(copy.deepcopy(env), policy, p["order"], 50_000 + r, scale, gamma, weights, horizon)
                for r in range(reps)]
        r0 = [x[1][0] for x in runs]
        v1 = [x[2][0] if x[2] else 0.0 for x in runs]
        res["plans"].append({**p, "links": list(key), "size": len(key), "g": [x[0] for x in runs],
                             "rew": [x[1] for x in runs], "vals": [x[2] for x in runs],
                             "comps": [x[3] for x in runs], "terminated": [x[4] for x in runs],
                             "r0": r0[0], "v1": v1[0],
                             "first_step_spread": float(max(np.ptp(r0), np.ptp(v1)))})
    return res


# ------------------------------------------------------------------ analysis
def _nstep(p: dict, n: int, gamma: float, reps) -> float:
    """Mean over ``reps`` of sum_{h<n} gamma^h r_h + gamma^n V(s_n) (shorter if the episode ended)."""
    out = []
    for r in reps:
        rew, vals = p["rew"][r], p["vals"][r]
        boot = gamma ** n * vals[n - 1] if len(rew) >= n and len(vals) >= n else 0.0  # 0: episode ended
        out.append(sum(gamma ** h * rew[h] for h in range(min(n, len(rew)))) + boot)
    return float(np.mean(out))


def pair_rows(states: list[dict], gamma: float) -> list[list[dict]]:
    """Per state: one row per pair of plans with the true difference and the estimates."""
    blocks = []
    for s in states:
        plans = s["plans"]
        if len(plans) < 2:
            continue
        reps = len(plans[0]["g"])
        first, second = range(reps // 2), range(reps // 2, reps)
        rows = []
        for i in range(len(plans)):
            for j in range(i + 1, len(plans)):
                a, b = plans[i], plans[j]
                d = np.array(a["g"]) - np.array(b["g"])
                d2 = d[list(second)]
                dc = np.array(a["comps"]) - np.array(b["comps"])  # (reps, components)
                row = {"dbar": d.mean(), "se": d.std(ddof=1) / np.sqrt(len(d)),
                       "dbar_half": d2.mean(), "se_half": d2.std(ddof=1) / np.sqrt(len(d2)),
                       "comp": dc.mean(0), "comp_se": dc.std(0, ddof=1) / np.sqrt(len(dc)),
                       "myopic": a["r0"] - b["r0"],
                       "c1": (a["r0"] + gamma * a["v1"]) - (b["r0"] + gamma * b["v1"]),
                       "ck": (a["vk"] - b["vk"]) if a["vk"] is not None and b["vk"] is not None else None,
                       "policy": a["source"] == "policy" and b["source"] == "policy",
                       "same_size": a["size"] == b["size"]}
                for n in STEPS[1:]:
                    row[f"n{n}"] = _nstep(a, n, gamma, first) - _nstep(b, n, gamma, first)
                rows.append(row)
        blocks.append(rows)
    return blocks


def _flat(blocks, key, sub=None, truth="dbar", se="se"):
    rows = [r for b in blocks for r in b if (sub is None or sub(r)) and r[key] is not None]
    return (np.array([r[key] for r in rows]), np.array([r[truth] for r in rows]), np.array([r[se] for r in rows]))


def calibration(blocks, key, sub=None, half=False) -> dict:
    t, s = ("dbar_half", "se_half") if half else ("dbar", "se")
    e, d, se = _flat(blocks, key, sub, t, s)
    if len(e) < 3 or not np.any(e):
        return {"pairs": int(len(e))}
    true_ms = np.mean(d ** 2 - se ** 2)
    cov = np.mean(e * d)
    return {"pairs": int(len(e)), "slope": float(cov / np.mean(e ** 2)),
            "corr": float(cov / np.sqrt(np.mean(e ** 2) * true_ms)) if true_ms > 0 else float("nan"),
            "rms_estimate": float(np.sqrt(np.mean(e ** 2)))}


def ml_tau(blocks, sub=None) -> float:
    _, d, se = _flat(blocks, "dbar", sub)
    if len(d) == 0:
        return float("nan")
    grid = np.linspace(0.0, max(0.05, 3 * float(np.abs(d).max())), 3001)
    ll = [-np.sum(np.log(t * t + se ** 2) + d ** 2 / (t * t + se ** 2)) for t in grid]
    return float(grid[int(np.argmax(ll))])


def bootstrap(blocks, fn, n=1000, seed=0) -> list[float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        v = fn([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        if v is not None and np.isfinite(v):
            vals.append(v)
    return [float(x) for x in np.percentile(vals, [2.5, 97.5])] if vals else [float("nan")] * 2


SUBSETS = {"all": None, "policy": lambda r: r["policy"], "same_size": lambda r: r["same_size"],
           "different_size": lambda r: not r["same_size"]}


def summarise(states: list[dict], gamma: float) -> dict:
    blocks = pair_rows(states, gamma)
    out = {"states": len(states), "states_with_pairs": len(blocks),
           "plans_per_state": float(np.mean([len(s["plans"]) for s in states])),
           "first_step_spread_max": float(max(p["first_step_spread"] for s in states for p in s["plans"])),
           "subsets": {}}
    for name, sub in SUBSETS.items():
        res = {"tau": ml_tau(blocks, sub), "tau_ci": bootstrap(blocks, lambda b: ml_tau(b, sub))}
        for key in ("myopic", "c1", "ck"):
            c = calibration(blocks, key, sub)
            if "slope" in c:
                c["slope_ci"] = bootstrap(blocks, lambda b: calibration(b, key, sub).get("slope"))
                c["corr_ci"] = bootstrap(blocks, lambda b: calibration(b, key, sub).get("corr"))
            res[key] = c
        res["n_step_split_half"] = {
            f"n{n}": calibration(blocks, "c1" if n == 1 else f"n{n}", sub, half=True) for n in STEPS}
        out["subsets"][name] = res
    rows = [r for b in blocks for r in b]
    comp = np.array([r["comp"] for r in rows])
    comp_se = np.array([r["comp_se"] for r in rows])
    c1 = np.array([r["c1"] for r in rows])
    var_true = np.mean(comp ** 2 - comp_se ** 2, axis=0)
    out["components"] = {
        k: {"true_ms": float(var_true[i]),
            "slope_on_c1": float(np.mean(c1 * comp[:, i]) / np.mean(c1 ** 2)),
            "corr_with_c1": float(np.mean(c1 * comp[:, i]) / np.sqrt(np.mean(c1 ** 2) * var_true[i]))
            if var_true[i] > 0 else float("nan")}
        for i, k in enumerate(COMPONENTS)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--states", type=int, default=48)
    ap.add_argument("--samples", type=int, default=24, help="policy samples per state")
    ap.add_argument("--policy-plans", type=int, default=4, help="most frequent sampled plans kept")
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=max(STEPS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--summary-only", action="store_true", help="re-analyse an existing output")
    ap.add_argument("--t0-min", type=int, default=60, help="earliest cycle of the probed states")
    ap.add_argument("--t0-max", type=int, default=150, help="latest cycle of the probed states")
    ap.add_argument("--base", type=int, default=DEV_BASE, help="first dev scene seed")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. scenario.flow_rate_pps=[30.0,45.0]")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    design = {"t0_range": [a.t0_min, a.t0_max], "base": a.base, "sets": a.set}
    if (a.t0_min, a.t0_max, a.base, a.set) != (60, 150, DEV_BASE, []) and not a.out:
        raise SystemExit("non-default design: give --out so the D3 outputs are not overwritten")
    out = Path(a.out) if a.out else ROOT / "results/diagnostics" / f"d3_{Path(a.run).name}.json"
    if a.summary_only:
        data = json.loads(out.read_text())
        states, gamma = data["states"], data["gamma"]
    else:
        jobs = [(a.run, i, a.samples, a.policy_plans, a.reps, a.horizon, (a.t0_min, a.t0_max), a.base, a.set)
                for i in range(a.states)]
        with Pool(a.workers) as pool:
            states = [s for s in pool.map(probe_state, jobs, chunksize=1) if s is not None]
        gamma = _setup(a.run)[3]
    summary = summarise(states, gamma)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run": a.run, "gamma": gamma, "design": design, "summary": summary,
                               "states": states}))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
