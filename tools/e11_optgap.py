"""E11 approximation ratio and decision time of every policy (`fanet-next optgap`).

Uses the policies of configs/experiments/e11_compare.json (learned ones at their
selected checkpoints) on the first ``--episodes`` test scenes (offset of the
spec), first ``--cycles`` cycles, for the default and nodes24_dense families.

    .venv/bin/python tools/e11_optgap.py --parallel 8
    .venv/bin/python tools/e11_optgap.py --summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import e10_ns3 as base
from confirm import CLI, ROOT, resolve, slug

SPEC = "configs/experiments/e11_compare.json"
SCENARIOS = ("default", "nodes24_dense")
COLS = [("ratio_mean", "近似比", "{:.3f}"), ("ratio_p5", "近似比 p5", "{:.3f}"),
        ("optimal_frac", "达到最优的周期占比", "{:.3f}"), ("decision_ms_mean", "决策 ms", "{:.2f}"),
        ("decision_ms_p99", "决策 p99 ms", "{:.1f}")]


def jobs(spec: dict, episodes: int, cycles: int) -> list[tuple[list[str], Path]]:
    out = []
    e = spec["eval"]
    for scenario in SCENARIOS:
        sets = [a for o in spec["scenarios"][scenario] for a in ("--set", o)]
        for name, p in base.policies(spec).items():
            targets = ([(p["args"], slug(name))] if p["runs"] is None else
                       [(["--run-dir", r, "--checkpoint", "@selected", "--policies", "ppo"], slug(r))
                        for r in p["runs"]])
            for args, stem in targets:
                f = ROOT / spec["out"] / "optgap" / scenario / f"{stem}.json"
                cmd = [CLI, "optgap", *args, "--split", e["split"], "--seed-offset",
                       str(e["seed_offset"]), "--episodes", str(episodes), "--cycles", str(cycles),
                       "--threads", "1", "--out", str(f), *sets]
                out.append((cmd, f))
    return out


def summarise(spec: dict, episodes: int, cycles: int) -> str:
    lines = [f"# E11 近似比与决策时间", "",
             f"test 偏移 {spec['eval']['seed_offset']} 的前 {episodes} 个场景，每个场景前 {cycles} 个周期。"
             "近似比 = 计划的单周期最大权（队列长度加权）÷ 同一决策状态上的 MILP 最优值；"
             "学习型策略按训练种子平均。", ""]
    for scenario in SCENARIOS:
        lines += [f"## `{scenario}`", "", "| 策略 | " + " | ".join(c[1] for c in COLS) + " |",
                  "| --- | " + " | ".join("---" for _ in COLS) + " |"]
        rows = []
        for name, p in base.policies(spec).items():
            stems = [slug(name)] if p["runs"] is None else [slug(r) for r in p["runs"]]
            files = [ROOT / spec["out"] / "optgap" / scenario / f"{s}.json" for s in stems]
            if not all(f.exists() for f in files):
                continue
            res = [next(iter(json.loads(f.read_text())["results"].values())) for f in files]
            rows.append((name, [float(np.mean([r[k] for r in res])) for k, _, _ in COLS]))
        for name, vals in sorted(rows, key=lambda r: -r[1][0]):
            lines.append(f"| {name} | " + " | ".join(f.format(v) for v, (_, _, f) in zip(vals, COLS)) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--cycles", type=int, default=500)
    p.add_argument("--summary", action="store_true")
    a = p.parse_args()
    spec = json.loads((ROOT / SPEC).read_text())
    if not a.summary:
        todo = []
        for cmd, f in jobs(spec, a.episodes, a.cycles):
            if not f.exists():
                f.parent.mkdir(parents=True, exist_ok=True)
                todo.append(resolve(cmd))
        print(f"{len(todo)} optgap jobs", flush=True)
        base.launch(todo, a.parallel, 2.0, lambda m: print(m, flush=True))
    text = summarise(spec, a.episodes, a.cycles)
    out = ROOT / spec["out"] / "optgap" / "summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
