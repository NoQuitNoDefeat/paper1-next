"""Pre-registered analysis of stage 0 (docs/experiments.md, "第 0 阶段预先登记").  Reads result files
only and writes results/stage0/analysis.json.  Parts that have no results yet are skipped.

* D6 (rules): per policy, rule Rk minus R0 (delivery, on-time-2s over born, mean delay, per-hop
  delivery), Holm over the five rules per policy and family; destination information
  (lq_lasthop minus LQ under the same rule); learned ordering (main minus lq_lasthop and minus LQ
  under the pre-specified rules R0 and R5, all rules descriptive); exit mapping (recovery ratio,
  stability of rule effects, backpressure vs LQ under R4/R5).
* D7 (environment): in families where main minus LQ is significant at E0, the difference in
  differences (main - LQ)@Ek - (main - LQ)@E0 per scene, Holm over E1, E2, E3, E5; E4 separately.
* D8 (selection): delivery-first re-selection over all kept checkpoints on offset 16, and, when the
  confirmation runs exist, the selected checkpoints against the current ones on offset 48.
* D3c: plan-difference scale tau and its relative size (tau / across-state SD of the Monte Carlo
  value) in the 2 x 2 cells {default, load_high} x {early, late}, and the critic's calibration.
* Drop location: per episode, main (seed average) minus LQ.

Main-method rows are averaged over its training seeds per scene ("seed average"); per-seed point
estimates are reported for the same-direction check.

    .venv/bin/python tools/stage0_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from confirm import ROOT, slug

FAMILIES = ("default", "load_high", "nodes24")
RULES = ("R0", "R1", "R2", "R3", "R4", "R5")
SETTINGS = ("E0", "E1", "E2", "E3", "E4", "E5")
MAIN = [f"results/e6/no_set_summary-s{s}" for s in range(3)]
METRICS = ("delivery_ratio", "ontime_2s", "e2e_delay_mean_s")
HOPS = ("h1", "h2", "h3p")
MIN_EFFECT = 0.005  # 0.5 percentage points (as D4)


def load(out: str, scenario: str, stem: str) -> dict[int, dict] | None:
    f = ROOT / out / scenario / "lightweight" / f"{stem}.shard0.json"
    if not f.exists():
        return None
    res = json.loads(f.read_text())["results"]
    return {r["seed"]: r for r in next(iter(res.values()))["rows"]}


def policy_rows(out: str, scenario: str, name: str) -> tuple[dict | None, list[dict]]:
    """(rows by seed, per-run rows) of a policy; the main method is averaged over its runs."""
    if name != "main":
        r = load(out, scenario, slug(name))
        return r, [r] if r else []
    runs = [load(out, scenario, slug(m)) for m in MAIN]
    if any(r is None for r in runs):
        return None, []
    seeds = sorted(set.intersection(*(set(r) for r in runs)))
    avg = {}
    for s in seeds:
        keys = {k for r in runs for k, v in r[s].items() if isinstance(v, (int, float))}
        avg[s] = {k: float(np.nanmean([r[s].get(k, np.nan) for r in runs])) for k in keys}
    return avg, runs


def paired(a: dict, b: dict, key: str) -> dict | None:
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s].get(key, np.nan) - b[s].get(key, np.nan) for s in seeds], dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 3:
        return None
    m, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    h = stats.t.ppf(0.975, len(d) - 1) * se
    p = float(2 * stats.t.sf(abs(m) / se, len(d) - 1)) if se > 0 else (0.0 if m != 0 else 1.0)
    return {"mean": m, "lo": m - h, "hi": m + h, "p": p, "n": len(d)}


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def per_seed_points(runs_a: list[dict], runs_b: list[dict], key: str) -> list[float]:
    """Per training seed of the main method: point estimate of A - B (B may be a single policy)."""
    out = []
    for i, ra in enumerate(runs_a):
        rb = runs_b[i] if len(runs_b) == len(runs_a) else runs_b[0]
        seeds = sorted(set(ra) & set(rb))
        d = [ra[s][key] - rb[s][key] for s in seeds if np.isfinite(ra[s].get(key, np.nan) - rb[s].get(key, np.nan))]
        out.append(float(np.mean(d)) if d else float("nan"))
    return out


# ------------------------------------------------------------------ beta of lq_lasthop
def beta() -> dict | None:
    """beta of lq_lasthop chosen on dev offset 16 (configs/experiments/d6_beta.json): the highest mean
    delivery over the six cells (R0 and R2 in three families), ties broken by lower mean delay."""
    spec = json.loads((ROOT / "configs/experiments/d6_beta.json").read_text())
    scores = {}
    for name in spec["controls"]:
        rows = [load(spec["out"], sc, slug(name)) for sc in spec["scenarios"]]
        if any(r is None for r in rows):
            return None
        scores[name] = (float(np.mean([np.mean([x["delivery_ratio"] for x in r.values()]) for r in rows])),
                        float(np.mean([np.nanmean([x["e2e_delay_mean_s"] for x in r.values()]) for r in rows])))
    best = max(scores, key=lambda n: (scores[n][0], -scores[n][1]))
    return {"scores": scores, "chosen": best, "policy": spec["controls"][best]["policy"]}


def apply_beta() -> None:
    """Write the chosen lq_lasthop into the D6 spec (its only control)."""
    b = beta()
    if b is None:
        raise SystemExit("d6_beta results are incomplete")
    path = ROOT / "configs/experiments/d6_rules.json"
    spec = json.loads(path.read_text())
    spec["controls"] = {b["chosen"]: {"policy": b["policy"], "set": []}}
    path.write_text(json.dumps(spec, indent=1, ensure_ascii=False) + "\n")
    print("chosen", b["chosen"], b["scores"])


# ------------------------------------------------------------------ D6
def d6() -> dict | None:
    spec = json.loads((ROOT / "configs/experiments/d6_rules.json").read_text())
    out_dir = spec["out"]
    heur = next((n for n in spec.get("controls", {})), "lq_lasthop")
    names = {"LQ": "longest_queue", "BP": "backpressure", "heur": heur, "main": "main"}
    data = {(f, r, k): policy_rows(out_dir, f"{f}_{r}", v) for f in FAMILIES for r in RULES for k, v in names.items()}
    if any(v[0] is None for v in data.values()):
        return None
    res = {"rule_value": {}, "destination_info": {}, "ordering": {}, "exit": {}}
    for k in names:
        for f in FAMILIES:
            base, base_runs = data[(f, "R0", k)]
            diffs = {r: {m: paired(data[(f, r, k)][0], base, m) for m in METRICS + tuple(f"delivery_ratio_{h}" for h in HOPS)}
                     for r in RULES[1:]}
            adj = holm([diffs[r]["delivery_ratio"]["p"] for r in RULES[1:]])
            for r, pa in zip(RULES[1:], adj):
                d = diffs[r]
                dl = d["delivery_ratio"]
                hop_harm = [h for h in HOPS if d[f"delivery_ratio_{h}"] and d[f"delivery_ratio_{h}"]["hi"] < 0
                            and d[f"delivery_ratio_{h}"]["mean"] <= -0.01]
                same_dir = None
                if k == "main":
                    pts = per_seed_points(data[(f, r, k)][1], base_runs, "delivery_ratio")
                    same_dir = all(np.sign(x) == np.sign(dl["mean"]) for x in pts)
                res["rule_value"][f"{k}|{f}|{r}"] = {
                    **{m: d[m] for m in METRICS}, "p_holm": pa, "hop_harm": hop_harm, "seeds_same_direction": same_dir,
                    "positive": dl["mean"] >= MIN_EFFECT and pa < 0.05 and dl["lo"] > 0 and not hop_harm
                    and (same_dir is not False), "negative": dl["hi"] < 0 and pa < 0.05}
    for f in FAMILIES:
        for r in RULES:
            res["destination_info"][f"{f}|{r}"] = {m: paired(data[(f, r, "heur")][0], data[(f, r, "LQ")][0], m)
                                                    for m in METRICS + tuple(f"delivery_ratio_{h}" for h in HOPS)}
            res["ordering"][f"{f}|{r}"] = {
                "main_minus_heur": {m: paired(data[(f, r, "main")][0], data[(f, r, "heur")][0], m) for m in METRICS},
                "main_minus_LQ": {m: paired(data[(f, r, "main")][0], data[(f, r, "LQ")][0], m) for m in METRICS}}
    trig = {r: [f for f in FAMILIES if res["ordering"][f"{f}|{r}"]["main_minus_heur"]["delivery_ratio"]["lo"] < MIN_EFFECT]
            for r in ("R0", "R5")}
    res["exit"]["heuristic_required_baseline"] = {"R0": trig["R0"], "R5": trig["R5"],
                                                   "only_under_rules": bool(trig["R5"]) and not trig["R0"]}
    rec = {}
    for f in FAMILIES:
        gap = paired(data[(f, "R0", "main")][0], data[(f, "R0", "LQ")][0], "delivery_ratio")
        gain = paired(data[(f, "R5", "heur")][0], data[(f, "R0", "LQ")][0], "delivery_ratio")
        rec[f] = {"main_gap": gap, "heur_R5_gain": gain,
                  "recovery": gain["mean"] / gap["mean"] if gap["lo"] > 0 else None}
    res["exit"]["recovery_ratio"] = rec
    res["exit"]["M2_rules_suffice"] = all(v["recovery"] is not None and v["recovery"] >= 0.7 for v in rec.values()
                                          if v["recovery"] is not None) and any(v["recovery"] is not None for v in rec.values())
    signs = {k: {f: np.sign(res["rule_value"][f"{k}|{f}|R5"]["delivery_ratio"]["mean"]) for f in FAMILIES} for k in ("LQ", "heur")}
    res["exit"]["rule_effect_signs_R5"] = signs
    res["exit"]["M1_rules_unstable"] = any(len(set(v.values())) > 1 for v in signs.values())
    res["exit"]["M3_bp_vs_lq"] = {f"{f}|{r}": paired(data[(f, r, "BP")][0], data[(f, r, "LQ")][0], "delivery_ratio")
                                  for f in FAMILIES for r in ("R4", "R5")}
    return res


# ------------------------------------------------------------------ D7
def d7() -> dict | None:
    spec = json.loads((ROOT / "configs/experiments/d7_env.json").read_text())
    out_dir = spec["out"]
    data = {(f, e, k): policy_rows(out_dir, f"{f}_{e}", v)[0] for f in FAMILIES for e in SETTINGS
            for k, v in (("LQ", "longest_queue"), ("main", "main"))}
    if any(v is None for v in data.values()):
        return None
    res = {}
    for f in FAMILIES:
        g0 = {s: {"x": data[(f, "E0", "main")][s]["delivery_ratio"] - data[(f, "E0", "LQ")][s]["delivery_ratio"]}
              for s in set(data[(f, "E0", "main")]) & set(data[(f, "E0", "LQ")])}
        gap0 = paired({s: {"x": v["x"]} for s, v in g0.items()}, {s: {"x": 0.0} for s in g0}, "x")
        entry = {"gap_E0": gap0, "eligible": gap0["lo"] > 0, "did": {}}
        for e in SETTINGS[1:]:
            ge = {s: {"x": data[(f, e, "main")][s]["delivery_ratio"] - data[(f, e, "LQ")][s]["delivery_ratio"]}
                  for s in set(data[(f, e, "main")]) & set(data[(f, e, "LQ")])}
            entry["did"][e] = paired(ge, g0, "x")
        tested = [e for e in ("E1", "E2", "E3", "E5")]
        adj = holm([entry["did"][e]["p"] for e in tested])
        for e, pa in zip(tested, adj):
            d = entry["did"][e]
            d["p_holm"] = pa
            d["shrinks_by_half"] = (entry["eligible"] and d["hi"] < 0 and d["mean"] <= -0.5 * gap0["mean"]
                                    and pa < 0.05)
        entry["did"]["E4"]["note"] = "secondary: frozen main method sees changed queue capacities"
        res[f] = entry
    return res


# ------------------------------------------------------------------ D8
def d8() -> dict | None:
    f = ROOT / "configs/experiments/d8_select.json"
    if not f.exists():
        return None
    spec = json.loads(f.read_text())
    out_dir = spec["out"]
    res = {"selected": {}, "confirm": {}}
    for s, run in enumerate(MAIN):
        cands = [n for n in spec["fixed"] if n.startswith(f"s{s}_")]
        scores = {}
        for n in cands:
            rows = [load(out_dir, fam, slug(n)) for fam in FAMILIES]
            if any(r is None for r in rows):
                return None
            scores[n] = (float(np.mean([np.mean([x["delivery_ratio"] for x in r.values()]) for r in rows])),
                         float(np.mean([np.nanmean([x["e2e_delay_mean_s"] for x in r.values()]) for r in rows])))
        best = max(scores, key=lambda n: (scores[n][0], -scores[n][1]))
        current = json.loads((ROOT / run / "selection.json").read_text())["selected"]["iteration"]
        res["selected"][run] = {"delivery_first": best, "scores": scores, "current_iteration": current}
    conf = ROOT / "results/d8/confirm"
    if conf.exists():
        for fam in FAMILIES:
            new = [load("results/d8/confirm", fam, slug(n)) for n in
                   [res["selected"][r]["delivery_first"] for r in MAIN]]
            cur = [load("results/d6/rules", f"{fam}_R0", slug(r)) for r in MAIN]
            if any(x is None for x in new + cur):
                continue
            seeds = sorted(set.intersection(*(set(x) for x in new + cur)))
            a = {s: {"d": float(np.mean([x[s]["delivery_ratio"] for x in new])),
                     "t": float(np.nanmean([x[s]["e2e_delay_mean_s"] for x in new]))} for s in seeds}
            b = {s: {"d": float(np.mean([x[s]["delivery_ratio"] for x in cur])),
                     "t": float(np.nanmean([x[s]["e2e_delay_mean_s"] for x in cur]))} for s in seeds}
            res["confirm"][fam] = {"delivery": paired(a, b, "d"), "delay": paired(a, b, "t")}
    return res


# ------------------------------------------------------------------ D3c
def d3c() -> dict | None:
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    from critic_consequence import bootstrap, calibration, ml_tau, pair_rows

    cells = {}
    for name in ("default_early", "default_late", "load_high_early", "load_high_late"):  # 2 x 2 cells of D3c
        files = [ROOT / "results/diagnostics" / f"d3c_{name}_s{s}.json" for s in range(3)]
        if not all(f.exists() for f in files):
            return None
        states, gamma = [], None
        for f in files:
            d = json.loads(f.read_text())
            states += d["states"]
            gamma = d["gamma"]
        blocks = pair_rows(states, gamma)
        q = np.array([np.mean([np.mean(p["g"]) for p in s["plans"]]) for s in states])
        sd = float(q.std())
        tau = ml_tau(blocks)

        def rel(b, _sd=sd):
            return ml_tau(b) / _sd
        cal = calibration(blocks, "c1")
        cells[name] = {"states": len(states), "tau": tau, "value_sd": sd, "rel_tau": tau / sd,
                       "rel_tau_ci": bootstrap(blocks, rel), "c1_slope": cal.get("slope"),
                       "c1_slope_ci": bootstrap(blocks, lambda b: calibration(b, "c1").get("slope")),
                       "myopic_corr": calibration(blocks, "myopic").get("corr")}
    early = cells["default_early"]["rel_tau"]
    cells["late_much_larger"] = cells["default_late"]["rel_tau_ci"][0] > 2 * early
    return cells


# ------------------------------------------------------------------ drop location
def drops() -> dict | None:
    base = ROOT / "results/stage0/drops"  # tools/drop_causes.py outputs, offset 48
    if not base.exists():
        return None
    res = {}
    for fam in FAMILIES:
        lq = base / f"{fam}_lq.json"
        mains = [base / f"{fam}_main_s{s}.json" for s in range(3)]
        if not lq.exists() or not all(m.exists() for m in mains):
            continue
        L = {r["seed"]: r for r in json.loads(lq.read_text())["rows"]}
        Ms = [{r["seed"]: r for r in json.loads(m.read_text())["rows"]} for m in mains]
        seeds = sorted(set(L).intersection(*Ms))
        M = {s: {k: float(np.mean([m[s].get(k, 0.0) for m in Ms])) for k in L[s] if k != "seed"} for s in seeds}
        tot = json.loads(lq.read_text())["share_of_born_pct"]
        rehome = tot.get("overflow_rehome", 0.0)
        res[fam] = {
            "rehome_at_relay_share_LQ": tot.get("overflow_rehome_at_relay", 0.0) / rehome if rehome else None,
            "queued_at_relay_main_minus_LQ": paired(M, L, "queued_at_relay_mean"),
            "queued_at_source_main_minus_LQ": paired(M, L, "queued_at_source_mean"),
            **{k: paired(M, L, k) for k in ("overflow_birth", "overflow_relay_arrival", "overflow_rehome_at_source",
                                           "overflow_rehome_at_relay", "delivered")}}
        q = res[fam]["queued_at_relay_main_minus_LQ"]
        res[fam]["supports_queues_kept_at_sources"] = bool(
            res[fam]["rehome_at_relay_share_LQ"] is not None and res[fam]["rehome_at_relay_share_LQ"] >= 0.5
            and q is not None and q["hi"] < 0)
    return res


def main() -> None:
    import sys
    if "--apply-beta" in sys.argv:
        apply_beta()
        return
    out = {"beta": beta(), "d6": d6(), "d7": d7(), "d8": d8(), "d3c": d3c(), "drops": drops()}
    path = ROOT / "results/stage0/analysis.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    for k, v in out.items():
        print(k, "done" if v is not None else "no results yet")


if __name__ == "__main__":
    main()
