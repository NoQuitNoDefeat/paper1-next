"""Noise-aware reading of diagnostic D2 (no new simulation): how large are the true outcome
differences between same-size complete plans, does the critic's plan distinction track them, and
how much does the complete-plan value enter the actor's advantages at all?

Reads ``results/diagnostics/d2_<group>-s<seed>.json`` (from ``critic_counterfactual.py``) and the
training logs of the same runs.  Per pair of plans at one state: ``dbar`` = mean paired outcome
difference over the R repetitions, ``se`` its standard error, ``dv`` = the critic's value difference.

* true gap scale tau: maximum likelihood with ``dbar ~ N(0, tau^2 + se^2)`` (noisy pairs weigh less);
* share of pairs with |t| > 2 against the rate pure noise gives (t with R-1 degrees of freedom);
* calibration slope ``sum(dv * dbar) / sum(dv^2)``: expected true difference per unit of predicted
  difference (1 = usable, 0 = unrelated); the noise in dbar is independent of dv, so no attenuation;
* sign agreement and what a perfect critic could reach on it (normal model with tau);
* weight of the complete-plan value V_k in the advantage of micro action j of a k-action cycle:
  ``(1 - lambda) * lambda^(k-1-j)`` (never above 1 - lambda); the realised outcome
  r + gamma V(next cycle) has ``lambda^(k-j)``;
* training regression noise: RMS of (lambda-return - V) = sqrt(2 * value loss), last 50 iterations.

Confidence intervals: bootstrap over states (pairs of one state are dependent).

    .venv/bin/python tools/d2_effects.py
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy import stats

from confirm import ROOT

RUNS = {"no_set_summary": "results/e6/no_set_summary-s{}", "critic_plan": "results/e17/runs/critic_plan-s{}"}


def pair_blocks(states: list[dict]) -> list[np.ndarray]:
    """Per state with >= 2 plans: rows (dbar, se, dv) for every pair of plans."""
    out = []
    for s in states:
        if len(s["plans"]) < 2:
            continue
        g = np.array([p["g"] for p in s["plans"]])
        vk = np.array([p["vk"] for p in s["plans"]])
        rows = [((g[i] - g[j]).mean(), (g[i] - g[j]).std(ddof=1) / np.sqrt(g.shape[1]), vk[i] - vk[j])
                for i in range(len(g)) for j in range(i + 1, len(g))]
        out.append(np.array(rows))
    return out


def ml_tau(blocks: list[np.ndarray]) -> float:
    a = np.concatenate(blocks)
    d2, s2 = a[:, 0] ** 2, a[:, 1] ** 2
    grid = np.linspace(0.0, 0.03, 601)
    ll = [-np.sum(np.log(t * t + s2) + d2 / (t * t + s2)) for t in grid]
    return float(grid[int(np.argmax(ll))])


def slope(blocks: list[np.ndarray]) -> float:
    a = np.concatenate(blocks)
    den = np.sum(a[:, 2] ** 2)
    return float(np.sum(a[:, 2] * a[:, 0]) / den) if den > 0 else float("nan")


def ceiling(blocks: list[np.ndarray], tau: float) -> float:
    se = np.concatenate(blocks)[:, 1]
    return float(np.mean(0.5 + np.arcsin(tau / np.sqrt(tau ** 2 + se ** 2)) / np.pi))


def bootstrap(blocks, fn, n=2000, seed=0) -> list[float]:
    rng = np.random.default_rng(seed)
    vals = [fn([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]) for _ in range(n)]
    return [float(x) for x in np.nanpercentile(vals, [2.5, 97.5])]


def train_noise(run: str) -> tuple[float, float]:
    rows = [json.loads(line) for line in (ROOT / run / "train_log.jsonl").read_text().splitlines() if line.strip()]
    rows = [r for r in rows if "ppo/value_loss" in r][-50:]
    return (float(np.sqrt(2 * np.mean([r["ppo/value_loss"] for r in rows]))),
            float(np.mean([r["actions_per_cycle"] for r in rows])))


def analyse(group: str, seeds: list[int], lam: float) -> dict:
    states, blocks, noise = [], [], []
    for s in seeds:
        d = json.loads((ROOT / "results/diagnostics" / f"d2_{group}-s{s}.json").read_text())
        states += d["states"]
        blocks += pair_blocks(d["states"])
        noise.append(train_noise(RUNS[group].format(s)))
    a = np.concatenate(blocks)
    reps = len(states[0]["plans"][0]["g"])
    t = np.abs(a[:, 0]) / a[:, 1]
    tau = ml_tau(blocks)
    tau_ci = bootstrap(blocks, ml_tau)
    wk = [(1 - lam) * lam ** (s["k_star"] - 1 - j) for s in states for j in range(s["k_star"])]
    wr = [lam ** (s["k_star"] - j) for s in states for j in range(s["k_star"])]
    out = {"group": group, "seeds": seeds, "states": len(states), "states_with_pairs": len(blocks), "pairs": len(a),
           "distinct_plans_mean": float(np.mean([s["distinct_plans"] for s in states])),
           "share_single_modal_plan": float(np.mean([s["distinct_same_size"] == 1 for s in states])),
           "tau": tau, "tau_ci": tau_ci,
           "share_abs_t_gt_2": float(np.mean(t > 2)), "null_share_abs_t_gt_2": float(2 * stats.t.sf(2, reps - 1)),
           "train_rms_target_minus_v": [n[0] for n in noise], "train_actions_per_cycle": [n[1] for n in noise],
           "weight_vk_mean": float(np.mean(wk)), "weight_vk_max": 1 - lam, "weight_realised_mean": float(np.mean(wr))}
    if np.any(a[:, 2] != 0):
        sig = t > 2
        agree = np.sign(a[:, 2]) == np.sign(a[:, 0])
        out.update({"critic_rms_gap": float(np.sqrt(np.mean(a[:, 2] ** 2))),
                    "slope": slope(blocks), "slope_ci": bootstrap(blocks, slope),
                    "sign_agree": float(agree.mean()), "sig_pairs": int(sig.sum()), "sig_agree": int(agree[sig].sum()),
                    "perfect_critic_ceiling": ceiling(blocks, tau),
                    "perfect_critic_ceiling_at_tau_upper": ceiling(blocks, tau_ci[1])})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=list(RUNS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lam", type=float, default=0.95)
    a = ap.parse_args()
    res = [analyse(g, a.seeds, a.lam) for g in a.groups]
    for r in res:
        print(json.dumps(r, indent=1))
    path = ROOT / "results/diagnostics/d2_effects.json"
    path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
