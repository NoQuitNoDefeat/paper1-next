"""Pre-registered analysis of stage 1a (docs/experiments.md, "第 1 阶段 1a 预先登记"): X0 loss ledger,
X1 = S1e non-clairvoyant rollout, X2 hindsight mask probe.  Reads result files only and writes
results/stage1a/analysis.json; parts without results are skipped.

X0 (descriptive): per deadline and family, means over episodes of the ledger (all born packets and
the steady-state window).

X1: per margin arm and family, the steady-state delivery gain of the rollout over heur@R5 (paired
over scenes, t interval), wins, the non-base share, window agreement (choice with H vs 1.5 H, mean
1.5 H gain of the chosen non-base candidates), clairvoyant agreement (mean true-future gain of the
chosen non-base candidates), and how long held-back packets waited.  Passing arm: mean >= +1 point
with lower bound > 0 in at least two families, and in those families the mean 1.5 H gain of the
chosen non-base candidates > 0.  The primary arm is margin 1; margin 0 is secondary.

X2: per deadline, family and mode, the steady-state delivery change of iteration k over iteration 0
(paired over scenes), the conversion g and the mask size.  Iteration 1 is primary (the mask built
from the base trajectory); iterations 2-3 are reported.  Readings at D = 1 in both congested
families (load_high, nodes24): r2 >= +1.5 -> relay-level value; b >= +1.5 -> source-level value.

    .venv/bin/python tools/stage1a_analysis.py
"""

from __future__ import annotations

import json

import numpy as np
from scipy import stats

from confirm import ROOT

FAMILIES = ("default", "load_high", "nodes24")
CONGESTED = ("load_high", "nodes24")
DEADLINES = ("1", "5")
MODES = ("a", "b", "r1", "r2", "rd")
MARGINS = ("1", "0")


def ci(d) -> dict | None:
    d = np.asarray([x for x in d if x is not None and np.isfinite(x)], dtype=float)
    if len(d) < 3:
        return None
    m, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    h = float(stats.t.ppf(0.975, len(d) - 1) * se)
    return {"mean": m, "lo": m - h, "hi": m + h, "n": int(len(d)), "wins": int((d > 0).sum())}


def load(path):
    p = ROOT / path
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def x0() -> dict:
    out = {}
    for d in DEADLINES:
        for f in FAMILIES:
            x = load(f"results/x0/{f}_D{d}.json")
            if x is None:
                continue
            rows = x["rows"]

            def qmean(key, part):
                vals = [r[part][key] for r in rows if r[part].get(key)]
                return [float(np.mean([v[i] for v in vals])) for i in range(3)] if vals else None
            cell = {"mean_all": x["mean_all"], "mean_steady": x["mean_steady"], "window": x["window"],
                    "depart_remaining_multihop": qmean("depart_remaining_q10_50_90_multihop", "steady"),
                    "last_tx_remaining_expired_at_relay": qmean("last_tx_remaining_q10_50_90_expired_at_relay", "steady")}
            arr = {}
            for h in ("1", "2", "3"):
                vals = [r["steady"]["arrive_remaining_q10_50_90_by_relay"][h] for r in rows
                        if r["steady"].get("arrive_remaining_q10_50_90_by_relay", {}).get(h)]
                arr[h] = [float(np.mean([v[i] for v in vals])) for i in range(3)] if vals else None
            cell["arrive_remaining_by_relay"] = arr
            out[f"{f}|D{d}"] = cell
    return out


def x1() -> dict:
    out = {}
    for m in MARGINS:
        arm, passing = {}, []
        for f in FAMILIES:
            x = load(f"results/s1e/{f}_m{m}.json")
            if x is None:
                continue
            rows = x["rows"]
            gain = [r["rollout"]["steady"]["delivered"] - r["base"]["steady"]["delivered"] for r in rows]
            gain_all = [r["rollout"]["all"]["delivered"] - r["base"]["all"]["delivered"] for r in rows]
            dec = [d for r in rows for d in r.get("decisions", [])]
            nb = [d for d in dec if d["chosen"] != "base"]
            held = [h for r in rows for h in r.get("held", [])]
            cell = {"gain_steady_pts": ci([100 * g for g in gain]), "gain_all_pts": ci([100 * g for g in gain_all]),
                    "decisions": len(dec), "non_base_share": len(nb) / max(len(dec), 1),
                    "window_agreement": float(np.mean([d["chosen"] == d["chosen_15"] for d in dec])) if dec else None,
                    "chosen_gain_15_mean": float(np.mean([d["gain_15"][d["chosen"]] for d in nb])) if nb else None,
                    "chosen_gain_true_mean": float(np.mean([d["gain_true"][d["chosen"]] for d in nb])) if nb else None,
                    "true_agreement": float(np.mean([d["chosen"] == d["chosen_true"] for d in dec])) if dec else None,
                    "true_non_base_share": float(np.mean([d["chosen_true"] != "base" for d in dec])) if dec else None,
                    "chosen_kinds": {k: sum(d["chosen"].startswith(k) for d in nb) for k in ("skip", "drop_last2", "drop_last")},
                    "held_wait_mean_cycles": float(np.mean([h["waited"] for h in held])) if held else None,
                    "held_lost_share": float(np.mean([h["fate"] == "lost" for h in held])) if held else None}
            arm[f] = cell
            g = cell["gain_steady_pts"]
            if g and g["mean"] >= 1.0 and g["lo"] > 0 and (cell["chosen_gain_15_mean"] or 0) > 0:
                passing.append(f)
        arm["passing_families"] = passing
        arm["passes"] = len(passing) >= 2
        out[f"margin{m}"] = arm
    prim = out.get("margin1", {})
    if "passes" in prim:
        gains = [prim[f]["gain_steady_pts"]["mean"] for f in FAMILIES if f in prim and prim[f]["gain_steady_pts"]]
        out["reading"] = ("B to stage 2; A's waiting enabled" if prim["passes"] else
                          "B audit only; A without waiting" if gains and max(gains) > 0 else
                          "B stops; A without waiting")
    return out


def x2() -> dict:
    out = {}
    for d in DEADLINES:
        for f in FAMILIES:
            for mode in MODES:
                x = load(f"results/x2/{f}_D{d}_{mode}.json")
                if x is None:
                    continue
                rows = x["rows"]
                cell = {}
                for k in range(1, x["iterations"] + 1):
                    dl = [100 * (r["iterations"][k]["delivered_steady"] - r["iterations"][0]["delivered_steady"]) for r in rows]
                    cell[f"iter{k}"] = {"delta_steady_pts": ci(dl),
                                        "delta_all_pts": ci([100 * (r["iterations"][k]["delivered_all"] - r["iterations"][0]["delivered_all"]) for r in rows]),
                                        "mask_size_mean": float(np.mean([r["iterations"][k]["mask_size"] for r in rows])),
                                        "g_mean": float(np.mean([r["iterations"][k]["conversion_g"] for r in rows
                                                                 if r["iterations"][k]["conversion_g"] is not None])),
                                        "wasted_share_mean": float(np.mean([r["iterations"][k]["tx_wasted_share"] for r in rows]))}
                cell["base_delivered_steady"] = float(np.mean([r["iterations"][0]["delivered_steady"] for r in rows]))
                cell["base_wasted_share"] = float(np.mean([r["iterations"][0]["tx_wasted_share"] for r in rows]))
                out[f"{f}|D{d}|{mode}"] = cell

    def value(mode):
        cells = [out.get(f"{f}|D1|{mode}", {}).get("iter1", {}).get("delta_steady_pts") for f in CONGESTED]
        return None if any(c is None for c in cells) else all(c["mean"] >= 1.5 for c in cells)
    out["reading"] = {"relay_value_r2": value("r2"), "source_value_b": value("b")}
    return out


def main() -> None:
    res = {"x0": x0(), "x1": x1(), "x2": x2()}
    (ROOT / "results/stage1a").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/stage1a/analysis.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print({k: (len(v) if isinstance(v, dict) else v) for k, v in res.items()})


if __name__ == "__main__":
    main()
