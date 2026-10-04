"""Pooled reading of diagnostic D3 (``critic_consequence.py`` outputs) over the seeds of a run group.

States of all seeds are pooled (bootstrap over states).  Besides the pair statistics of
``critic_consequence.summarise`` it splits the critic's error at the next state,
``e = V(s') - (Q(p) - r) / gamma`` (Q(p): Monte Carlo return of plan p), into the part common to all
plans of a state (cancels when plans are compared and largely in the advantages) and the part that
differs between plans, and gives the across-state correlation of the cycle-start value V0 with the
Monte Carlo value of the state (the critic's job of telling good from bad states).

    .venv/bin/python tools/d3_summary.py                      # final method, seeds 0-2
    .venv/bin/python tools/d3_summary.py --group separate_critic
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from confirm import ROOT
from critic_consequence import bootstrap, summarise


def decompose(states: list[dict], gamma: float) -> dict:
    rows = [(k, p["v1"], (np.mean(p["g"]) - p["r0"]) / gamma, np.std(p["g"], ddof=1) / np.sqrt(len(p["g"])) / gamma,
             s["v0"], np.mean(p["g"])) for k, s in enumerate(states) for p in s["plans"]]
    st = np.array([r[0] for r in rows])
    v1, cont, se = (np.array([r[i] for r in rows]) for i in (1, 2, 3))
    keys = np.unique(st)

    def within(x):
        return np.concatenate([x[st == k] - x[st == k].mean() for k in keys])

    def between(x):
        return np.array([x[st == k].mean() for k in keys])

    err = v1 - cont
    noise = float(np.sqrt(np.mean(se ** 2)))
    true_within = float(np.sqrt(max(within(cont).var() - noise ** 2, 0.0)))
    v0 = np.array([s["v0"] for s in states])
    q = np.array([np.mean([np.mean(p["g"]) for p in s["plans"]]) for s in states])  # mean over the state's plans
    return {"error_rms": float(np.sqrt(np.mean(err ** 2))), "error_between_sd": float(between(err).std()),
            "error_within_sd": float(within(err).std()), "critic_within_sd": float(within(v1).std()),
            "mc_within_sd_observed": float(within(cont).std()), "mc_noise_sd": noise,
            "true_within_sd": true_within, "mc_between_sd": float(q.std()),
            "corr_v0_mc_across_states": float(np.corrcoef(v0, q)[0, 1])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="no_set_summary", help="d3_<group>-s<seed>.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    states, gamma, per_seed = [], None, {}
    for s in a.seeds:
        d = json.loads((ROOT / "results/diagnostics" / f"d3_{a.group}-s{s}.json").read_text())
        states += d["states"]
        gamma = d["gamma"]
        per_seed[s] = decompose(d["states"], gamma)
    out = {"group": a.group, "seeds": a.seeds, "pooled": summarise(states, gamma),
           "decomposition_pooled": decompose(states, gamma), "decomposition_per_seed": per_seed}
    from critic_consequence import pair_rows, calibration
    blocks = pair_rows(states, gamma)
    out["c1_minus_myopic_corr_ci"] = bootstrap(
        blocks, lambda b: calibration(b, "c1").get("corr", np.nan) - calibration(b, "myopic").get("corr", np.nan))
    path = ROOT / "results/diagnostics" / f"d3_summary_{a.group}.json"
    path.write_text(json.dumps(out, indent=1))
    p = out["pooled"]["subsets"]
    for name in ("all", "policy", "same_size", "different_size"):
        sub = p[name]
        line = f"[{name:14s}] tau {sub['tau']:.4f} {np.round(sub['tau_ci'], 4)}"
        for key in ("myopic", "c1", "ck"):
            c = sub[key]
            if "slope" in c:
                line += (f" | {key}: slope {c['slope']:+.3f} {np.round(c['slope_ci'], 3)} corr {c['corr']:+.2f} "
                         f"{np.round(c['corr_ci'], 2)} rms {c['rms_estimate']:.4f}")
        print(line)
    print("c1 corr - myopic corr:", np.round(out["c1_minus_myopic_corr_ci"], 2))
    print("decomposition (pooled):", json.dumps({k: round(v, 4) for k, v in out["decomposition_pooled"].items()}))
    for s, dcp in per_seed.items():
        print(f"  seed {s}: within-state critic SD {dcp['critic_within_sd']:.4f}, true {dcp['true_within_sd']:.4f}, "
              f"corr(V0, MC) across states {dcp['corr_v0_mc_across_states']:.2f}")


if __name__ == "__main__":
    main()
