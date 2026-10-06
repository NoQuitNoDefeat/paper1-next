"""Pre-registered analysis of E21 (docs/experiments.md, "E21 预先登记"): asymmetric actor-critic.

Arms: P1 privileged critic (future exogenous arrivals), P2 placebo (same network, privileged input
all zero), main method (existing runs).  Writes results/e21/analysis.json.

1. Learning signal (training logs): per run, the mean over the last 50 iterations of the critic's
   explained variance and of the raw advantage standard deviation (P1 and P2 only; the main runs do
   not log the latter).
2. Capability (the critic ranks plans in the same state): D3-style one-step calibration slope of
   r + gamma V(next) differences against Monte Carlo plan differences, default family, states at
   cycles 60-150, horizon 900, 32 states per run, scene base 10 001 200, the three runs of an arm
   pooled; bootstrap over states.  Learned: P1's slope >= 0.3 and its lower bound above P2's upper
   bound.
3. Policy (dev offset 16, 32 scenes, drain 1000, three families): reward-A objective and delivery,
   P1 - P2, P1 - main, P2 - main (scenes paired, learned groups averaged over seeds).  Promising:
   P1 - P2 objective significantly > 0 in at least two families, and the P1 - P2 delivery lower
   bound >= -0.5 points in every family.

    .venv/bin/python tools/e21_analysis.py
"""

from __future__ import annotations

import json

import numpy as np

from confirm import ROOT

ARMS = {"P1": [f"results/e21/runs/privileged-s{s}" for s in range(3)],
        "P2": [f"results/e21/runs/placebo-s{s}" for s in range(3)],
        "main": [f"results/e6/no_set_summary-s{s}" for s in range(3)]}
D3_FILE = "results/e21/d3/{arm}_s{seed}.json"


def training() -> dict:
    out = {}
    for arm, runs in ARMS.items():
        rows = []
        for r in runs:
            p = ROOT / r / "train_log.jsonl"
            if not p.exists():
                continue
            its = [json.loads(x) for x in p.read_text().splitlines() if '"iteration"' in x]
            last = its[-50:]
            rows.append({"run": r, "iterations": len(its),
                         "explained_var_last50": float(np.mean([x["ppo/explained_var"] for x in last])) if last else None,
                         "adv_std_raw_last50": (float(np.mean([x["ppo/adv_std_raw"] for x in last]))
                                                if last and "ppo/adv_std_raw" in last[0] else None)})
        out[arm] = rows
    return out


def capability() -> dict | None:
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    from critic_consequence import bootstrap, calibration, pair_rows
    out = {}
    for arm in ARMS:
        files = [ROOT / D3_FILE.format(arm=arm, seed=s) for s in range(3)]
        if not all(f.exists() for f in files):
            return None
        states, gamma = [], None
        for f in files:
            d = json.loads(f.read_text())
            states += d["states"]
            gamma = d["gamma"]
        blocks = pair_rows(states, gamma)
        cal = calibration(blocks, "c1")
        out[arm] = {"states": len(states), "c1_slope": cal.get("slope"), "c1_corr": cal.get("corr"),
                    "c1_rms_estimate": cal.get("rms_estimate"),
                    "c1_slope_ci": bootstrap(blocks, lambda b: calibration(b, "c1").get("slope"))}
    p1, p2 = out["P1"], out["P2"]
    out["learned"] = bool(p1["c1_slope"] is not None and p1["c1_slope"] >= 0.3
                          and p1["c1_slope_ci"][0] > p2["c1_slope_ci"][1])
    return out


def policy() -> dict | None:
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    from reward_objective import paired, scores, weights
    spec_path = ROOT / "configs/experiments/e21_dev.json"
    spec = json.loads(spec_path.read_text())
    w = weights()
    out = {}
    try:
        for fam in spec["scenarios"]:
            _, sc = scores(spec, fam, w)
            cell = {}
            for a, b in (("P1·特权评论者", "P2·安慰剂"), ("P1·特权评论者", "主方法"), ("P2·安慰剂", "主方法")):
                cell[f"{a} - {b}"] = {"reward": paired(sc[a]["reward"], sc[b]["reward"]),
                                      "delivery_pts": [100 * x for x in paired(sc[a]["delivery"], sc[b]["delivery"])],
                                      "delay_s": paired(sc[a]["delay"], sc[b]["delay"])}
            out[fam] = cell
    except FileNotFoundError:
        return None
    key = "P1·特权评论者 - P2·安慰剂"
    sig = [f for f in out if out[f][key]["reward"][1] > 0]
    out["promising"] = len(sig) >= 2 and all(out[f][key]["delivery_pts"][1] >= -0.5 for f in spec["scenarios"])
    out["significant_families"] = sig
    return out


def main() -> None:
    res = {"training": training(), "capability": capability(), "policy": policy()}
    (ROOT / "results/e21").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/e21/analysis.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print({k: ("done" if v else "incomplete") for k, v in res.items()})


if __name__ == "__main__":
    main()
