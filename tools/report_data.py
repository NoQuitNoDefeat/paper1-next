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
              "speed_0_5": "速度 0–5", "speed_30_60": "速度 30–60",
              # E12b-E15
              "c2_default": "C2 · 默认", "c2_load_5_15": "C2 · 负载 5–15",
              "c2_nodes24_same_density": "C2 · 24 节点", "c2s5_default": "C2S5 · 默认",
              "c2s5_load_5_15": "C2S5 · 负载 5–15", "c2s5_nodes24_same_density": "C2S5 · 24 节点",
              "flock": "群集 30（三次飞行合并）", "flock_c2s5": "群集 30 · C2S5",
              "fold_4mps": "4 m/s 斜线", "fold_6mps": "6 m/s 圆周", "fold_8mps": "8 m/s 圆周",
              "rpgm": "BonnMotion RPGM", "gauss_markov": "BonnMotion 高斯-马尔可夫",
              "n30": "30 架", "ideal": "理想信道", "r161": "单跳 161 m",
              "slow_compact": "低速紧凑", "nodes30_dense": "30 节点·更密",
              "flock16": "群集 30 · 16 架（合并）", "flock30": "群集 30 · 30 架（合并）"}
BASELINE_LABEL = {"max_weight_opt": "精确最大权（MILP）", "backpressure_opt": "精确反压（MILP）",
                  "longest_queue": "longest_queue", "hol_weighted": "hol_weighted", "oldest_hol": "oldest_hol",
                  "backpressure": "反压（贪心）", "lq_local_search": "LQ + 局部搜索",
                  "spatial_tdma": "空间 TDMA", "random": "random", "仅模仿": "仅模仿（自有变体）",
                  "Zhao-GCN": "Zhao 等 GCN（TWC 2023）", "GRLinQ": "GRLinQ 式图强化学习",
                  "独立决定PPO": "独立决定 PPO（同模型）",
                  "主方法": "定稿方法", "主方法·重训": "定稿方法 · 重训", "Zhao-GCN·重训": "Zhao 等 GCN · 重训",
                  "GRLinQ·重训": "GRLinQ 式 · 重训", "独立决定PPO·重训": "独立决定 PPO · 重训",
                  "仅模仿·重训": "仅模仿 · 重训", "主方法·宽": "定稿方法 · 宽分布",
                  "Zhao-GCN·宽": "Zhao 等 GCN · 宽分布", "GRLinQ·宽": "GRLinQ 式 · 宽分布",
                  "独立决定PPO·宽": "独立决定 PPO · 宽分布", "仅模仿·宽": "仅模仿 · 宽分布"}
KEYS3 = ("delivery_ratio", "e2e_delay_mean_s", "e2e_delay_p95_s")
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


# ---------------------------------------------------------------- E12b onwards
def _per_scene(spec: dict, scenarios: list[str], name: str, key: str, seeds=None, executor=None):
    """Seed-averaged per-scene values of one policy, concatenated over scenarios (None if missing)."""
    from e10_ns3 import load_rows, seed_avg

    ex = executor or next(iter(spec["executors"]))
    out = []
    for s in scenarios:
        rows = load_rows(spec, s, ex, name)
        if rows is None:
            return None
        if seeds is not None:
            rows = [[r for r in run if r["seed"] in seeds] for run in rows]
        out.append(seed_avg(rows, key))
    return np.concatenate(out)


def _cmp(spec: dict, scenarios: list[str], a: str, b: str, keys=KEYS3, seeds=None, executor=None,
         scenarios_b: list[str] | None = None) -> dict | None:
    """a - b per scene (b on ``scenarios_b`` if given, scene by scene): mean and 95% CI per metric."""
    out = {}
    for k in keys:
        x = _per_scene(spec, scenarios, a, k, seeds, executor)
        y = _per_scene(spec, scenarios_b or scenarios, b, k, seeds, executor)
        if x is None or y is None:
            return None
        m, lo, hi = ci(x - y)
        out[k] = {"m": m, "lo": lo, "hi": hi}
    return out


def _mean(spec: dict, scenarios: list[str], name: str, key: str, seeds=None, executor=None) -> float | None:
    v = _per_scene(spec, scenarios, name, key, seeds, executor)
    return None if v is None else float(np.nanmean(v))


def _versus(spec: dict, columns: dict[str, list[str]], main: str, names: list[str]) -> dict:
    """{other: {column: main - other}} for column -> scenarios to pool; missing cells left out."""
    out = {}
    for n in names:
        if n == main:
            continue
        for col, scen in columns.items():
            d = _cmp(spec, scen, main, n)
            if d is not None:
                out.setdefault(n, {})[col] = d
    return out


def _complete(spec: dict) -> bool:
    from e10_ns3 import build_jobs

    return not build_jobs(spec)


def _first_seeds(spec: dict, scenario: str, n: int) -> set:
    from e10_ns3 import load_rows

    return set(sorted({r["seed"] for r in load_rows(spec, scenario, "lightweight", "longest_queue")[0]})[:n])


def _effect(spec: dict, names: list[str], new: list[str], ref: list[str]) -> dict:
    """Per policy: new - ref on the same scenes (delivery, mean delay) and both delivery means."""
    out = {}
    for n in names:
        row = _cmp(spec, new, n, n, ("delivery_ratio", "e2e_delay_mean_s"), scenarios_b=ref)
        if row is not None:
            out[n] = {**row, "ref": _mean(spec, ref, n, "delivery_ratio"), "new": _mean(spec, new, n, "delivery_ratio")}
    return out


def e12b() -> dict:
    """Measured-parameter channel C2 (E12b): main vs every policy, channel effect, sensitivity."""
    from e10_ns3 import policies

    spec = _load_spec("configs/experiments/e12b_channel.json")
    sens = _load_spec("configs/experiments/e12b_sensitivity.json")
    names = list(policies(spec))
    fams = ["default", "load_5_15", "nodes24_same_density"]
    effect = {}
    for f in fams:
        for n, row in _effect(spec, names, [f"c2_{f}"], [f"c0_{f}"]).items():
            effect.setdefault(n, {})[f] = row
    seeds = _first_seeds(spec, "c2_default", sens["eval"]["episodes"])
    labels = {"n2.0_k10": "指数 2.0", "n2.5_k10": "指数 2.5", "n2.2_k5": "K = 5 dB", "n2.2_k15": "K = 15 dB",
              "n2.2_k10_outage1": "中断目标 1%", "n2.2_k10_outage30": "中断目标 30%"}
    conds = [("C2（指数 2.2，K = 10 dB，中断 10%）", spec, "c2_default", seeds)]
    conds += [(labels.get(c, c), sens, c, None) for c in sens["scenarios"]]
    sensitivity = []
    for label, sp, scen, sd in conds:
        d = _cmp(sp, [scen], "主方法", "longest_queue", ("delivery_ratio", "e2e_delay_mean_s"), sd)
        if d is not None:
            sensitivity.append({"label": label, "dr_main": _mean(sp, [scen], "主方法", "delivery_ratio", sd), **d})
    return {"versus": _versus(spec, {f"c2_{f}": [f"c2_{f}"] for f in fams}, "主方法", names),
            "columns": [f"c2_{f}" for f in fams], "effect": effect, "families": fams,
            "sensitivity": sensitivity, "episodes": spec["eval"]["episodes"],
            "seed_offset": spec["eval"]["seed_offset"]}


def e13() -> dict:
    """Real flocking trajectories and BonnMotion (E13; E13b adds every trainable baseline)."""
    from e10_ns3 import policies

    e13b = _load_spec("configs/experiments/e13b_flock.json")
    flock = e13b if _complete(e13b) else _load_spec("configs/experiments/e13_flock.json")
    bonn = _load_spec("configs/experiments/e13_bonnmotion.json")
    folds = list(flock["scenarios"])
    cols = {"flock": folds, **{f: [f] for f in folds}}
    out = {"flock_spec": "e13b" if flock is e13b else "e13", "folds": folds,
           "columns": ["flock", *folds, *bonn["scenarios"]]}
    for main in ("主方法", "主方法·重训"):
        v = _versus(flock, cols, main, list(policies(flock)))
        for n, d in _versus(bonn, {b: [b] for b in bonn["scenarios"]}, main, list(policies(bonn))).items():
            v.setdefault(n, {}).update(d)
        out[main] = v
    out["means"] = {col: {n: {k: _mean(sp, scen, n, k) for k in ("delivery_ratio", "e2e_delay_mean_s")}
                          for n in policies(sp)}
                    for sp, colmap in ((flock, cols), (bonn, {b: [b] for b in bonn["scenarios"]}))
                    for col, scen in colmap.items()}
    extra_b = _load_spec("configs/experiments/e13b_flock_extra.json")
    extra = extra_b if _complete(extra_b) else _load_spec("configs/experiments/e13_flock_extra.json")
    out["extra"] = _versus(extra, {s: [s] for s in extra["scenarios"]}, "主方法", list(policies(extra)))
    out["extra_columns"] = list(extra["scenarios"])
    ns3 = _load_spec("configs/experiments/e13_ns3.json")
    out["ns3"] = []
    for f in ns3["scenarios"]:
        row = {"fold": f, **{x: _cmp(ns3, [f], "主方法", "longest_queue", ("delivery_ratio", "e2e_delay_mean_s"),
                                     executor=x) for x in ns3["executors"]}}
        for n in policies(ns3):
            a = _per_scene(ns3, [f], n, "delivery_ratio", executor="ns3")
            b = _per_scene(ns3, [f], n, "delivery_ratio", executor="lightweight")
            m, lo, hi = ci(a - b)
            row[f"fid_{n}"] = {"m": m, "lo": lo, "hi": hi}
        out["ns3"].append(row)
    out["ns3_episodes"] = ns3["eval"]["episodes"]
    for key, path in (("ucsb", "results/e13/ucsb_linkmodel.json"), ("rssi", "results/e15/ucsb_rssi.json")):
        f = ROOT / path
        out[key] = json.loads(f.read_text()) if f.exists() else None
    return out


def e14() -> dict | None:
    """Exploration: training mixture covering slow and compact swarms (None until complete)."""
    from e10_ns3 import policies

    syn, fl = _load_spec("configs/experiments/e14_synthetic.json"), _load_spec("configs/experiments/e14_flock.json")
    if not (_complete(syn) and _complete(fl)):
        return None
    f16 = [s for s in fl["scenarios"] if not s.endswith("_n30")]
    f30 = [s for s in fl["scenarios"] if s.endswith("_n30")]
    cells = [*((s, syn, [s]) for s in syn["scenarios"]), ("flock16", fl, f16), ("flock30", fl, f30)]
    out = {"columns": [c for c, _, _ in cells]}
    for main in ("主方法·宽", "主方法"):
        v = {}
        for col, sp, scen in cells:
            for n, d in _versus(sp, {col: scen}, main, list(policies(sp))).items():
                v.setdefault(n, {}).update(d)
        out[main] = v
    pairs = [("主方法·宽", "主方法"), ("Zhao-GCN·宽", "Zhao-GCN"), ("GRLinQ·宽", "GRLinQ"),
             ("独立决定PPO·宽", "独立决定PPO"), ("仅模仿·宽", "仅模仿")]
    out["broad_vs_orig"] = {}
    for a, b in pairs:
        for col, sp, scen in cells:
            d = _cmp(sp, scen, a, b)
            if d is not None:
                out["broad_vs_orig"].setdefault(a, {})[col] = d
    out["means"] = {col: {n: _mean(sp, scen, n, "delivery_ratio") for n in policies(sp)} for col, sp, scen in cells}
    return out


def e15() -> dict | None:
    """Exploration: 5 dB known shadowing as a controlled variable (None until complete)."""
    from e10_ns3 import policies

    ch, fl = _load_spec("configs/experiments/e15_channel.json"), _load_spec("configs/experiments/e15_flock.json")
    sg, ns3 = _load_spec("configs/experiments/e15_sigma.json"), _load_spec("configs/experiments/e15_ns3.json")
    if not all(_complete(s) for s in (ch, fl, sg)):
        return None
    fams = ["default", "load_5_15", "nodes24_same_density"]
    folds = sorted({s.split("_", 1)[1] for s in fl["scenarios"]})
    groups = [*((f, ch, [f"c2_{f}"], [f"c2s5_{f}"]) for f in fams),
              ("flock", fl, [f"c2_{f}" for f in folds], [f"c2s5_{f}" for f in folds])]
    out = {"families": [g[0] for g in groups], "columns": [f"c2s5_{f}" if f != "flock" else "flock_c2s5"
                                                             for f, *_ in groups],
           "versus": {}, "effect": {}, "did": {}}
    for fam, sp, s2, s5 in groups:
        col = "flock_c2s5" if fam == "flock" else f"c2s5_{fam}"
        for n, d in _versus(sp, {col: s5}, "主方法", list(policies(sp))).items():
            out["versus"].setdefault(n, {}).update(d)
        for n, row in _effect(sp, list(policies(sp)), s5, s2).items():
            out["effect"].setdefault(n, {})[fam] = row
        a5 = _per_scene(sp, s5, "主方法", "delivery_ratio") - _per_scene(sp, s5, "longest_queue", "delivery_ratio")
        a2 = _per_scene(sp, s2, "主方法", "delivery_ratio") - _per_scene(sp, s2, "longest_queue", "delivery_ratio")
        m, lo, hi = ci(a5 - a2)
        out["did"][fam] = {"m": m, "lo": lo, "hi": hi}
    seeds = _first_seeds(ch, "c2_default", sg["eval"]["episodes"])
    sigma = []
    for label, sp, scen, sd in [("无阴影（C2）", ch, "c2_default", seeds), ("σ = 3 dB", sg, "c2s3_default", None),
                                ("σ = 5 dB", ch, "c2s5_default", seeds), ("σ = 7 dB", sg, "c2s7_default", None),
                                ("σ = 5 dB 且 K = 5 dB", sg, "c2s5k5_default", None)]:
        d = _cmp(sp, [scen], "主方法", "longest_queue", ("delivery_ratio", "e2e_delay_mean_s"), sd)
        sigma.append({"label": label, "dr_main": _mean(sp, [scen], "主方法", "delivery_ratio", sd),
                      "dr_lq": _mean(sp, [scen], "longest_queue", "delivery_ratio", sd), **d})
    out["sigma"] = sigma
    out["ns3"] = None
    if _complete(ns3):
        row = {x: _cmp(ns3, ["c2s5_default"], "主方法", "longest_queue", ("delivery_ratio", "e2e_delay_mean_s"),
                       executor=x) for x in ns3["executors"]}
        for n in policies(ns3):
            a = _per_scene(ns3, ["c2s5_default"], n, "delivery_ratio", executor="ns3")
            b = _per_scene(ns3, ["c2s5_default"], n, "delivery_ratio", executor="lightweight")
            m, lo, hi = ci(a - b)
            row[f"fid_{n}"] = {"m": m, "lo": lo, "hi": hi}
        out["ns3"] = row
    return out


def collect() -> dict:
    return {"e9": e9(), "e10": e10(), "e11": e11(), "curves": curves(), "ablations": ablations(),
            "e8": summary_redesign(), "dev": dev_history(), "e12b": e12b(), "e13": e13(), "e14": e14(),
            "e15": e15(), "labels": SCEN_LABEL, "names": BASELINE_LABEL}


if __name__ == "__main__":
    d = collect()
    s = d["e9"]["scenarios"][0]
    print(json.dumps({"default_lq_diff": s["diffs"]["longest_queue"]["delivery_ratio"],
                      "curves": {g: [len(r["points"]) for r in runs] for g, runs in d["curves"]["groups"].items()},
                      "lq_ref": d["curves"]["lq"], "abl": d["ablations"]["components"],
                      "e8_scen": d["e8"]["scenarios"], "dev": d["dev"], "sel": d["e9"]["selected"]},
                     ensure_ascii=False, indent=1))
