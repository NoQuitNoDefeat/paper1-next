"""E11: the final method against the added baselines (spec: configs/experiments/e11_*.json).

Job building, staggered launch, retry and the progress board are shared with
tools/e10_ns3.py (one evaluation job per policy / training seed and scenario);
this script only adds the E11 summary: every scenario's mean table and the
paired differences of the final method against every other policy, plus a
cross-scenario overview.

    .venv/bin/python tools/e11_compare.py --spec configs/experiments/e11_compare.json --parallel 8
    .venv/bin/python tools/e11_compare.py --spec ... --status --watch 30
    .venv/bin/python tools/e11_compare.py --spec ... --summary
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import e10_ns3 as base
from confirm import ROOT, ci, resolve

KEYS = [("delivery_ratio", "交付率", "{:.4f}"), ("termination_ratio", "终止比例", "{:.4f}"),
        ("ontime_2s", "2s 送达", "{:.4f}"), ("e2e_delay_mean_s", "平均时延", "{:.3f}"),
        ("e2e_delay_p95_s", "p95 时延", "{:.3f}"), ("decision_ms_mean", "决策 ms", "{:.2f}")]
PAIR_KEYS = ["delivery_ratio", "ontime_2s", "e2e_delay_mean_s", "e2e_delay_p95_s"]
NI_MARGIN = -0.005


def _mark(lo: float, hi: float, better_high: bool) -> str:
    if lo > 0:
        return "↑" if better_high else "↓差"
    if hi < 0:
        return "↓差" if better_high else "↑"
    return "="


def summarise(spec: dict) -> str:
    e = spec["eval"]
    names = [n for n in base.policies(spec)]
    main = spec.get("main", next(iter(spec["groups"])))
    executor = next(iter(spec["executors"]))
    lines = [f"# {spec['name']}", "",
             f"{e['split']} 种子偏移 {e['seed_offset']}，{e['episodes']} 个场景，排空 {e['drain']} 周期。"
             "学习型策略先按训练种子平均，再按场景配对；均值 [95% CI]。", ""]
    overview: dict[str, dict[str, str]] = {}
    for scenario in spec["scenarios"]:
        skip = set(spec.get("exclude", {}).get(scenario, []))
        data = {n: base.load_rows(spec, scenario, executor, n) for n in names if n not in skip}
        lines += [f"## 场景 `{scenario}`", "", "| 策略 | " + " | ".join(k[1] for k in KEYS) + " |",
                  "| --- | " + " | ".join("---" for _ in KEYS) + " |"]
        ready = {n: r for n, r in data.items() if r is not None}
        order = sorted(ready, key=lambda n: -float(np.mean(base.seed_avg(ready[n], "delivery_ratio"))))
        for n in order:
            vals = [float(np.nanmean(base.seed_avg(ready[n], k))) for k, _, _ in KEYS]
            label = f"**{n}**" if n == main else n
            lines.append(f"| {label} | " + " | ".join(f.format(v) for v, (_, _, f) in zip(vals, KEYS)) + " |")
        lines.append("")
        if main not in ready:
            continue
        lines += [f"**{main} 相对各策略的配对差**", "",
                  "| 对照 | " + " | ".join(dict((k, l) for k, l, _ in KEYS)[k] for k in PAIR_KEYS)
                  + " | 可靠性可接受 |", "| --- | " + " | ".join("---" for _ in PAIR_KEYS) + " | --- |"]
        for n in order:
            if n == main:
                continue
            cells, marks = [], []
            for k in PAIR_KEYS:
                m, lo, hi = ci(base.seed_avg(ready[main], k) - base.seed_avg(ready[n], k))
                d = 4 if "delay" not in k else 3
                cells.append(f"{m:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]")
                marks.append(_mark(lo, hi, "delay" not in k))
            lo = ci(base.seed_avg(ready[main], "delivery_ratio")
                    - base.seed_avg(ready[n], "delivery_ratio"))[1]
            lines.append(f"| {n} | " + " | ".join(cells) + f" | {'是' if lo >= NI_MARGIN else '否'} |")
            overview.setdefault(n, {})[scenario] = f"{marks[0]}/{marks[2]}"
        lines.append("")
    if overview:
        scen = list(spec["scenarios"])
        lines += ["## 总览：主方法相对各策略（交付率 / 平均时延）", "",
                  "↑ 显著更好，↓差 显著更差，= 不显著。", "",
                  "| 对照 | " + " | ".join(scen) + " |", "| --- | " + " | ".join("---" for _ in scen) + " |"]
        for n, row in overview.items():
            lines.append(f"| {n} | " + " | ".join(row.get(s, "–") for s in scen) + " |")
        lines.append("")
    lines += [f"判定：可靠性可接受 = 交付率配对差 95% CI 下界 ≥ {NI_MARGIN}。没有做多重比较校正。"]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--gap", type=float, default=5.0, help="seconds between job starts")
    p.add_argument("--summary", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--watch", type=float, default=0.0)
    a = p.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    out = ROOT / spec["out"]
    out.mkdir(parents=True, exist_ok=True)
    if a.status:
        while True:
            board = base.status(spec)
            print("\033[2J\033[H" + board if a.watch else board, flush=True)
            if not a.watch:
                return
            time.sleep(a.watch)

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with (out / "run.log").open("a") as fh:
            fh.write(line + "\n")

    if not a.summary:
        (out / "driver.pid").write_text(str(os.getpid()))
        jobs = base.build_jobs(spec)
        log(f"{len(jobs)} evaluation jobs")
        base.launch([resolve(j) for j in jobs], a.parallel, a.gap, log)
        log("all jobs finished")
    text = summarise(spec)
    (out / "summary.md").write_text(text + "\n")
    log(f"wrote {out / 'summary.md'}")
    print(text)


if __name__ == "__main__":
    main()
