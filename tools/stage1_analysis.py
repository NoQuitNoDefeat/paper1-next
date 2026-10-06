"""Pre-registered analysis of stage 1 S1 (docs/experiments.md, "第 1 阶段 S1 预先登记"): delivery
losses under end-to-end deadlines.  Reads result files only and writes results/s1/analysis.json;
parts without results yet are skipped.

1. Where delivery is lost (descriptive, R5): expiry location and wasted transmissions of LQ,
   backpressure, lq_lasthop and main s0 per deadline and family; flag when the strongest rule
   baseline (of the three, by delivery) wastes >= 20 % of its successful transmissions.
2. Rules under deadlines: per policy, R5 - R0 and R5e - R5 (delivery, mean delay, per-hop
   delivery), Holm over the two per policy, family and deadline; criteria as D6.  Ordering is
   state dependent for a policy when significant R5e - R5 differences of both signs occur.
3. Strongest rule baseline per deadline and family: the highest mean delivery among
   {LQ, backpressure, lq_lasthop} x {R5, R5e}; the frozen main method (seed average, R5 and R5e)
   minus it.
4. Delay of delivered packets (mean, p95) for every policy and rule, as the second metric.

    .venv/bin/python tools/stage1_analysis.py
"""

from __future__ import annotations

import json

import numpy as np

from confirm import ROOT
from stage0_analysis import HOPS, MIN_EFFECT, holm, paired, per_seed_points, policy_rows

SPEC = ROOT / "configs/experiments/s1_deadline.json"
FAMILIES = ("default", "load_high", "nodes24")
DEADLINES = ("0.5", "1", "2", "5")
RULES = ("R0", "R5", "R5e")
NAMES = {"LQ": "longest_queue", "BP": "backpressure", "heur": "lq_lasthop_b1.0", "main": "main"}
DROP_TAGS = {"LQ": "longest_queue", "BP": "backpressure", "heur": "lq_lasthop", "main_s0": "main_s0"}
WASTE_FLAG = 0.20


def grid() -> dict | None:
    out = json.loads(SPEC.read_text())["out"]
    data = {(f, d, r, k): policy_rows(out, f"{f}_D{d}_{r}", v)
            for f in FAMILIES for d in DEADLINES for r in RULES for k, v in NAMES.items()}
    return None if any(v[0] is None for v in data.values()) else data


def mean_of(rows: dict, key: str) -> float:
    return float(np.nanmean([r.get(key, np.nan) for r in rows.values()]))


def rules(data: dict) -> dict:
    res, signs = {}, {k: [] for k in NAMES}
    metrics = ("delivery_ratio", "e2e_delay_mean_s") + tuple(f"delivery_ratio_{h}" for h in HOPS)
    for f in FAMILIES:
        for d in DEADLINES:
            for k in NAMES:
                pairs = {"R5-R0": ("R5", "R0"), "R5e-R5": ("R5e", "R5")}
                diffs = {c: {m: paired(data[(f, d, a, k)][0], data[(f, d, b, k)][0], m) for m in metrics}
                         for c, (a, b) in pairs.items()}
                adj = holm([diffs[c]["delivery_ratio"]["p"] for c in pairs])
                for (c, (a, b)), pa in zip(pairs.items(), adj):
                    x, dl = diffs[c], diffs[c]["delivery_ratio"]
                    hop_harm = [h for h in HOPS if x[f"delivery_ratio_{h}"] and x[f"delivery_ratio_{h}"]["hi"] < 0
                                and x[f"delivery_ratio_{h}"]["mean"] <= -0.01]
                    same_dir = None
                    if k == "main":
                        pts = per_seed_points(data[(f, d, a, k)][1], data[(f, d, b, k)][1], "delivery_ratio")
                        same_dir = all(np.sign(p) == np.sign(dl["mean"]) for p in pts)
                    sig = pa < 0.05
                    res[f"{k}|{f}|D{d}|{c}"] = {
                        "delivery_ratio": dl, "e2e_delay_mean_s": x["e2e_delay_mean_s"], "p_holm": pa,
                        "hop_harm": hop_harm, "seeds_same_direction": same_dir,
                        "valuable": dl["mean"] >= MIN_EFFECT and sig and dl["lo"] > 0 and not hop_harm
                        and same_dir is not False,
                        "harmful": dl["hi"] < 0 and sig}
                    if c == "R5e-R5" and sig and same_dir is not False:
                        signs[k].append(int(np.sign(dl["mean"])))
    res["ordering_state_dependent"] = {k: len(set(v)) > 1 for k, v in signs.items()}
    res["ordering_significant_signs"] = signs
    return res


def strongest(data: dict) -> dict:
    res = {}
    for f in FAMILIES:
        for d in DEADLINES:
            cands = {(k, r): mean_of(data[(f, d, r, k)][0], "delivery_ratio")
                     for k in ("LQ", "BP", "heur") for r in ("R5", "R5e")}
            (bk, br), best = max(cands.items(), key=lambda kv: kv[1])
            ref = data[(f, d, br, bk)][0]
            res[f"{f}|D{d}"] = {
                "best": f"{bk}@{br}", "best_delivery": best,
                "best_delay": mean_of(ref, "e2e_delay_mean_s"),
                "all": {f"{k}@{r}": v for (k, r), v in cands.items()},
                "main_minus_best": {r: {m: paired(data[(f, d, r, "main")][0], ref, m)
                                        for m in ("delivery_ratio", "e2e_delay_mean_s")} for r in ("R5", "R5e")}}
    return res


def delays(data: dict) -> dict:
    return {f"{k}|{f}|D{d}|{r}": {m: mean_of(data[(f, d, r, k)][0], m)
                                  for m in ("delivery_ratio", "e2e_delay_mean_s", "e2e_delay_p95_s")}
            for f in FAMILIES for d in DEADLINES for r in RULES for k in NAMES}


def drops() -> dict | None:
    base = ROOT / "results/s1/drops"
    if not base.exists():
        return None
    res = {}
    for f in FAMILIES:
        for d in DEADLINES:
            cell = {}
            for k, tag in DROP_TAGS.items():
                p = base / f"{f}_D{d}_{tag}.json"
                if not p.exists():
                    continue
                try:
                    x = json.loads(p.read_text())
                except json.JSONDecodeError:
                    continue
                cell[k] = {"share_of_born_pct": x["share_of_born_pct"], "transmissions": x["transmissions"],
                           "queued_packets_per_cycle": x["queued_packets_per_cycle"]}
            if not cell:
                continue
            rule_pols = [k for k in ("LQ", "BP", "heur") if k in cell]
            if rule_pols:
                best = max(rule_pols, key=lambda k: cell[k]["share_of_born_pct"].get("delivered", 0.0))
                cell["strongest_rule"] = best
                cell["waste_flag"] = cell[best]["transmissions"]["tx_wasted_share"] >= WASTE_FLAG
            res[f"{f}|D{d}"] = cell
    return res


def main() -> None:
    data = grid()
    out = {"drops": drops()}
    if data is not None:
        out.update({"rules": rules(data), "strongest": strongest(data), "delays": delays(data)})
    print("grid", "done" if data is not None else "incomplete", "| drops", "done" if out["drops"] else "none")
    path = ROOT / "results/s1/analysis.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
