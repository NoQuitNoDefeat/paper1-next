"""Collect the numbers shown on the results report page (tools/build_report.py).

Everything is read from results/ so the page can be regenerated; nothing is
typed by hand.  Statistics follow tools/confirm.py: seed-averaged per
scenario, then paired over scenarios (mean and 95% CI).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from confirm import by_seed, ci, slug  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCEN_LABEL = {"default": "默认", "nodes24_dense": "24 节点·更密", "nodes24_same_density": "24 节点·同密度",
              "load_high": "高负载", "channel_noisy": "信道有误差",
              "load_5_15": "负载 5–15", "load_20_40": "负载 20–40", "load_40_60": "负载 40–60",
              "nodes32_same_density": "32 节点", "nodes48_same_density": "48 节点",
              "speed_0_5": "速度 0–5", "speed_30_60": "速度 30–60"}
BASELINE_LABEL = {"max_weight_opt": "精确最大权（MILP）", "backpressure_opt": "精确反压（MILP）",
                  "longest_queue": "longest_queue", "hol_weighted": "hol_weighted", "oldest_hol": "oldest_hol",
                  "backpressure": "反压（贪心）", "lq_local_search": "LQ + 局部搜索",
                  "spatial_tdma": "空间 TDMA", "random": "random", "仅模仿": "仅模仿（自有变体）",
                  "Zhao-GCN": "Zhao 等 GCN（TWC 2023）", "GRLinQ": "GRLinQ 式图强化学习",
                  "独立决定PPO": "独立决定 PPO（同模型）"}
METRICS = ["delivery_ratio", "termination_ratio", "e2e_delay_mean_s", "e2e_delay_p95_s", "ontime_2s"]


def _load_spec(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def _policies(spec: dict, variant: str) -> dict[str, list[list[dict]]]:
    out = ROOT / spec["out"] / variant
    base = json.loads((out / "baselines.json").read_text())["results"]
    pol = {b: [base[b]["rows"]] for b in spec["baselines"]}
    for name in spec.get("fixed", {}):
        pol[name] = [json.loads((out / f"fixed_{slug(name)}.json").read_text())["results"]["ppo"]["rows"]]
    for g, runs in spec["groups"].items():
        pol[g] = [json.loads((out / f"{slug(r)}.json").read_text())["results"]["ppo"]["rows"] for r in runs]
    return pol


def _avg(seeds: list[list[dict]], key: str) -> np.ndarray:
    return np.mean([by_seed(r, key) for r in seeds], axis=0)


def _diff(a, b, key) -> dict:
    m, lo, hi = ci(_avg(a, key) - _avg(b, key))
    return {"m": m, "lo": lo, "hi": hi}


def e9() -> dict:
    spec = _load_spec("configs/experiments/e9_test.json")
    main = "主方法（无摘要）"
    scen = []
    for v in spec["variants"]:
        pol = _policies(spec, v)
        means = {p: {k: float(np.nanmean(_avg(s, k))) for k in METRICS} for p, s in pol.items()}
        diffs = {ref: {k: _diff(pol[main], pol[ref], k) for k in METRICS}
                 for ref in ("longest_queue", "hol_weighted", "原门控求和（原设计）", "仅模仿")}
        scen.append({"id": v, "label": SCEN_LABEL[v], "means": means, "diffs": diffs,
                     "n": len(by_seed(pol["longest_queue"][0], "delivery_ratio"))})
    sel = {}
    for runs in spec["groups"].values():
        for r in runs:
            sel[r] = json.loads((ROOT / r / "selection.json").read_text())["selected"]["iteration"]
    return {"scenarios": scen, "episodes": spec["eval"]["episodes"], "drain": spec["eval"]["drain"],
            "seeds": {g: len(r) for g, r in spec["groups"].items()}, "selected": sel}


def curves() -> dict:
    groups = {"主方法（无摘要）": ["results/e6/no_set_summary-s0", "results/e6/no_set_summary-s1",
                                "results/e6/no_set_summary-s2", "results/e9/final-s3", "results/e9/final-s4"],
              "原门控求和（原设计）": ["results/e5/A-lv20-s0", "results/e5/A-lv20-s1", "results/e5/A-lv20-s2"]}
    out = {}
    for g, runs in groups.items():
        out[g] = []
        for r in runs:
            pts = []
            for line in (ROOT / r / "eval_log.jsonl").read_text().splitlines():
                e = json.loads(line)
                pts.append([e["iteration"], e["dev/delivery_ratio"], e["dev/e2e_delay_mean_s"]])
            sel = json.loads((ROOT / r / "selection.json").read_text())["selected"]["iteration"]
            out[g].append({"run": r.split("/")[-1], "points": pts, "selected": sel})
    lq = json.loads((ROOT / "results/baselines/base-dev16-drain250.json").read_text())["results"]
    rows = lq["longest_queue"]["rows"][:8]  # training-time dev evaluation used dev seeds 0-7
    ref = {"delivery_ratio": float(np.mean([r["delivery_ratio"] for r in rows])),
           "e2e_delay_mean_s": float(np.mean([r["e2e_delay_mean_s"] for r in rows]))}
    return {"groups": out, "lq": ref}


def ablations() -> dict:
    e6 = _load_spec("configs/experiments/e6_ablation.json")
    e6b = _load_spec("configs/experiments/e6b_hard_ablation.json")
    comps = [g for g in e6["groups"] if g != "完整模型"]
    table = {c: {} for c in comps}
    for spec, variants in ((e6, ["default"]), (e6b, list(e6b["variants"]))):
        for v in variants:
            pol = _policies(spec, v)
            for c in comps:
                table[c][v] = {k: _diff(pol[c], pol["完整模型"], k)
                               for k in ("delivery_ratio", "e2e_delay_mean_s", "e2e_delay_p95_s")}
    return {"components": comps, "scenarios": ["default"] + list(e6b["variants"]), "table": table}


def summary_redesign() -> dict:
    d = _load_spec("configs/experiments/e8_summary_default.json")
    h = _load_spec("configs/experiments/e8_summary_hard.json")
    cands = ["归一化摘要", "控制器动态特征", "完整模型（门控求和）"]
    table = {c: {} for c in cands}
    for spec in (d, h):
        for v in spec["variants"]:
            pol = _policies(spec, v)
            for c in cands:
                table[c][v] = {k: _diff(pol[c], pol["无摘要"], k)
                               for k in ("delivery_ratio", "e2e_delay_mean_s", "e2e_delay_p95_s")}
    return {"candidates": cands, "scenarios": list(d["variants"]) + list(h["variants"]), "table": table}


def dev_history() -> dict:
    """A few dev-set facts quoted in the timeline."""
    pure = []
    for s in range(3):
        f = ROOT / f"results/e5/confirm/runs_base-lv5-s{s}.json"
        pure.append(json.loads(f.read_text())["results"]["ppo"]["mean"]["delivery_ratio"])
    base = json.loads((ROOT / "results/e5/confirm/baselines.json").read_text())["results"]
    imit = json.loads((ROOT / "results/e5/confirm/runs_imit-lq-bc-only.json").read_text())["results"]["ppo"]["mean"]
    return {"pure_ppo_dr": pure, "lq_dr": base["longest_queue"]["mean"]["delivery_ratio"],
            "imit_dr": imit["delivery_ratio"]}


def e10() -> dict:
    """ns-3 PHY validation: the method's advantage in both executors, and per-policy fidelity."""
    from e10_ns3 import load_rows, policies, seed_avg

    spec = _load_spec("configs/experiments/e10_ns3.json")
    group, ref = next(iter(spec["groups"])), spec["baselines"][0]
    ns3 = next(x for x, b in spec["executors"].items() if b)
    light = next(x for x, b in spec["executors"].items() if not b)
    control = next(iter(spec.get("controls", {})), None)

    def d(a, b, key):
        m, lo, hi = ci(seed_avg(a, key) - seed_avg(b, key))
        return {"m": m, "lo": lo, "hi": hi}

    scen = []
    for v in spec["scenarios"]:
        rows = {x: {n: load_rows(spec, v, x, n) for n in policies(spec)} for x in spec["executors"]}
        adv = {x: {k: d(rows[x][group], rows[x][ref], k) for k in METRICS} for x in (light, ns3)}
        fid = {}
        for n in policies(spec):
            a, b = rows[ns3][n], rows[light][n]
            if a is None or b is None:
                continue
            fid[n] = {"delivery_ratio": d(a, b, "delivery_ratio"),
                      "e2e_delay_mean_s": d(a, b, "e2e_delay_mean_s"),
                      "per": float(np.mean(seed_avg(a, "per"))),
                      "failed": float(np.mean(seed_avg(a, "exec_failed_link_frac"))),
                      "dr_ns3": float(np.mean(seed_avg(a, "delivery_ratio"))),
                      "control": n == control}
        scen.append({"id": v, "label": SCEN_LABEL[v], "adv": adv, "fid": fid})
    return {"scenarios": scen, "episodes": spec["eval"]["episodes"], "drain": spec["eval"]["drain"],
            "seed_offset": spec["eval"]["seed_offset"], "main": group, "ref": ref, "control": control,
            "executors": {"light": light, "ns3": ns3}}


def e11() -> dict:
    """Main method minus every other policy (E11 comparison and sweeps), optgap, ns-3 re-check."""
    from e10_ns3 import load_rows, policies, seed_avg

    def table(spec_path: str) -> dict:
        spec = _load_spec(spec_path)
        main = spec.get("main", next(iter(spec["groups"])))
        ex = next(iter(spec["executors"]))
        out, means = {}, {}
        for v in spec["scenarios"]:
            rows = {n: load_rows(spec, v, ex, n) for n in policies(spec)
                    if n not in spec.get("exclude", {}).get(v, [])}
            rows = {n: r for n, r in rows.items() if r is not None}
            means[v] = {n: {k: float(np.mean(seed_avg(r, k))) for k in METRICS + ["decision_ms_mean"]}
                        for n, r in rows.items()}
            for n, r in rows.items():
                if n == main:
                    continue
                out.setdefault(n, {})[v] = {k: dict(zip(("m", "lo", "hi"), ci(seed_avg(rows[main], k)
                                                                             - seed_avg(r, k))))
                                            for k in ("delivery_ratio", "e2e_delay_mean_s", "e2e_delay_p95_s")}
        return {"diffs": out, "means": means, "scenarios": list(spec["scenarios"]),
                "episodes": spec["eval"]["episodes"], "seed_offset": spec["eval"]["seed_offset"]}

    comp, sweeps = table("configs/experiments/e11_compare.json"), table("configs/experiments/e11_sweeps.json")
    opt = {}
    spec = _load_spec("configs/experiments/e11_compare.json")
    for v in ("default", "nodes24_dense"):
        for name, p in policies(spec).items():
            stems = [slug(name)] if p["runs"] is None else [slug(r) for r in p["runs"]]
            files = [ROOT / spec["out"] / "optgap" / v / f"{s}.json" for s in stems]
            if all(f.exists() for f in files):
                res = [next(iter(json.loads(f.read_text())["results"].values())) for f in files]
                opt.setdefault(name, {})[v] = {k: float(np.mean([r[k] for r in res]))
                                               for k in ("ratio_mean", "decision_ms_mean", "decision_ms_p99")}
    ns3 = None
    ns3_spec = _load_spec("configs/experiments/e11_ns3.json")
    names = list(policies(ns3_spec))
    ready = all(load_rows(ns3_spec, v, x, n) is not None for v in ns3_spec["scenarios"]
                for x in ns3_spec["executors"] for n in names)
    if ready:
        main = next(iter(ns3_spec["groups"]))
        ns3 = {"scenarios": list(ns3_spec["scenarios"]), "episodes": ns3_spec["eval"]["episodes"], "rows": {}}
        for v in ns3_spec["scenarios"]:
            data = {x: {n: load_rows(ns3_spec, v, x, n) for n in names} for x in ns3_spec["executors"]}
            for n in names:
                r = {"dr_light": float(np.mean(seed_avg(data["lightweight"][n], "delivery_ratio"))),
                     "dr_ns3": float(np.mean(seed_avg(data["ns3"][n], "delivery_ratio"))),
                     "delay_ns3": float(np.mean(seed_avg(data["ns3"][n], "e2e_delay_mean_s"))),
                     "per": float(np.mean(seed_avg(data["ns3"][n], "per")))}
                if n != main:
                    for k in ("delivery_ratio", "e2e_delay_mean_s"):
                        m, lo, hi = ci(seed_avg(data["ns3"][main], k) - seed_avg(data["ns3"][n], k))
                        r[f"diff_{k}"] = {"m": m, "lo": lo, "hi": hi}
                ns3["rows"].setdefault(n, {})[v] = r
    return {"compare": comp, "sweeps": sweeps, "optgap": opt, "ns3": ns3, "labels": BASELINE_LABEL}


def collect() -> dict:
    return {"e9": e9(), "e10": e10(), "e11": e11(), "curves": curves(), "ablations": ablations(),
            "e8": summary_redesign(), "dev": dev_history(), "labels": SCEN_LABEL}


if __name__ == "__main__":
    d = collect()
    s = d["e9"]["scenarios"][0]
    print(json.dumps({"default_lq_diff": s["diffs"]["longest_queue"]["delivery_ratio"],
                      "curves": {g: [len(r["points"]) for r in runs] for g, runs in d["curves"]["groups"].items()},
                      "lq_ref": d["curves"]["lq"], "abl": d["ablations"]["components"],
                      "e8_scen": d["e8"]["scenarios"], "dev": d["dev"], "sel": d["e9"]["selected"]},
                     ensure_ascii=False, indent=1))
