"""E5 held-out confirmation and summary (protocol in docs/experiments.md, E5).

Evaluates each run's selected checkpoint (selection.json) on dev seeds 16..47
with drain, plus heuristics and the imitation-only model on the same seeds,
then reports paired differences per training seed and across seeds.

    .venv/bin/python tools/e5_confirm.py            # evaluate missing pieces, then summarise
    .venv/bin/python tools/e5_confirm.py --summary  # summarise existing results only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from math import sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLI = str(ROOT / ".venv" / "bin" / "fanet-next")
OUT = ROOT / "results" / "e5" / "confirm"
EVAL = ["--split", "dev", "--seed-offset", "16", "--episodes", "32", "--drain", "250",
        "--threads", "3"]
GROUPS = {
    "A-λ5（主）": [f"results/e5/A-lv5-s{s}" for s in range(3)],
    "A-λ20（主）": [f"results/e5/A-lv20-s{s}" for s in range(3)],
    "B-λ5（奖励对照）": [f"results/e5/B-lv5-s{s}" for s in range(3)],
    "纯 PPO λ5（从零）": [f"results/runs/base-lv5-s{s}" for s in range(3)],
}
IMIT = "results/runs/imit-lq-bc-only"
KEYS = [("delivery_ratio", "交付率", 1), ("ontime_1s", "1s 送达", 1), ("ontime_2s", "2s 送达", 1),
        ("termination_ratio", "终止比例", -1), ("e2e_delay_mean_s", "平均时延", -1),
        ("e2e_delay_p95_s", "p95 时延", -1)]
NI_MARGIN = -0.005  # registered: delivery CI lower bound >= -0.5 percentage points


def name_of(run: str) -> str:
    return run.replace("results/", "").replace("/", "_")


def jobs() -> list[list[str]]:
    todo = []
    base = OUT / "baselines.json"
    if not base.exists():
        todo.append([CLI, "eval", "--config", "configs/base.toml", "--policies", "longest_queue",
                     "hol_weighted", "random", *EVAL, "--out", str(base)])
    imit = OUT / f"{name_of(IMIT)}.json"
    if not imit.exists():
        todo.append([CLI, "eval", "--run-dir", IMIT, "--policies", "ppo", *EVAL, "--out", str(imit)])
    for runs in GROUPS.values():
        for run in runs:
            sel = ROOT / run / "selection.json"
            out = OUT / f"{name_of(run)}.json"
            if out.exists():
                continue
            if not sel.exists():
                print(f"missing selection for {run}; run `fanet-next select` first", file=sys.stderr)
                continue
            ckpt = json.loads(sel.read_text())["selected"]["checkpoint"]
            todo.append([CLI, "eval", "--run-dir", run, "--checkpoint", ckpt, "--policies", "ppo",
                         *EVAL, "--out", str(out)])
    return todo


def run_jobs(todo: list[list[str]], parallel: int) -> None:
    running: list[subprocess.Popen] = []
    for cmd in todo:
        while len(running) >= parallel:
            running = [p for p in running if p.poll() is None]
            time.sleep(2)
        log = open(str(cmd[cmd.index("--out") + 1]) + ".log", "w")
        running.append(subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT))
        print(f"{time.strftime('%H:%M:%S')} started {cmd[cmd.index('--out') + 1]}", flush=True)
    for p in running:
        p.wait()


def paired(a: list[dict], b: list[dict], key: str) -> tuple[float, float, float, int]:
    x = np.array([r[key] for r in a], float) - np.array([r[key] for r in b], float)
    x = x[~np.isnan(x)]
    n = len(x)
    half = (2.04 if n >= 30 else 2.13) * x.std(ddof=1) / sqrt(n)
    return float(x.mean()), float(x.mean() - half), float(x.mean() + half), n


def summarise() -> str:
    base = json.loads((OUT / "baselines.json").read_text())["results"]
    lq, hw = base["longest_queue"]["rows"], base["hol_weighted"]["rows"]
    rows_of = {"longest_queue": lq, "hol_weighted": hw, "random": base["random"]["rows"]}
    imit = OUT / f"{name_of(IMIT)}.json"
    if imit.exists():
        rows_of["仅模仿"] = json.loads(imit.read_text())["results"]["ppo"]["rows"]
    sel_info = {}
    for group, runs in GROUPS.items():
        for run in runs:
            f = OUT / f"{name_of(run)}.json"
            if f.exists():
                rows_of[f"{group}|{run}"] = json.loads(f.read_text())["results"]["ppo"]["rows"]
                sel = json.loads((ROOT / run / "selection.json").read_text())
                sel_info[run] = (sel["selected"]["iteration"], sel["any_eligible"])

    lines = ["# E5 留出集确认（dev 种子 16–47，排空 250 周期，n = 32）", ""]
    lines += ["## 各策略平均值", "",
              "| 策略 | 选中迭代 | " + " | ".join(k[1] for k in KEYS) + " |",
              "| --- | --- | " + " | ".join("---" for _ in KEYS) + " |"]
    for name, rows in rows_of.items():
        it = ""
        if "|" in name:
            run = name.split("|")[1]
            it = f"{sel_info[run][0]}" + ("" if sel_info[run][1] else "（无可入选）")
            label = f"{name.split('|')[0]} s{run[-1]}"
        else:
            label = name
        vals = [np.nanmean([r[k] for r in rows]) for k, _, _ in KEYS]
        lines.append(f"| {label} | {it} | " + " | ".join(f"{v:.4f}" if k[0] != "e2e_delay_mean_s" and
                                                             k[0] != "e2e_delay_p95_s" else f"{v:.3f} s"
                                                             for v, k in zip(vals, KEYS)) + " |")

    for ref_name, ref in (("longest_queue", lq), ("hol_weighted", hw)):
        lines += ["", f"## 相对 {ref_name} 的配对差（均值 [95% CI]）", "",
                  "| 组 | 种子 | " + " | ".join(k[1] for k in KEYS) + " | 可靠性可接受 |",
                  "| --- | --- | " + " | ".join("---" for _ in KEYS) + " | --- |"]
        for group, runs in GROUPS.items():
            seed_means = {k: [] for k, _, _ in KEYS}
            pooled_a, pooled_b = [], []
            for run in runs:
                key = f"{group}|{run}"
                if key not in rows_of:
                    continue
                cells, ok = [], None
                for k, _, _ in KEYS:
                    m, lo, hi, n = paired(rows_of[key], ref, k)
                    seed_means[k].append(m)
                    cells.append(f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
                    if k == "delivery_ratio":
                        ok = lo >= NI_MARGIN
                pooled_a += rows_of[key]
                pooled_b += ref
                lines.append(f"| {group} | s{run[-1]} | " + " | ".join(cells) + f" | {'是' if ok else '否'} |")
            if len(seed_means["delivery_ratio"]) >= 2:
                cells, ok = [], None
                for k, _, _ in KEYS:
                    m, lo, hi, n = paired(pooled_a, pooled_b, k)
                    cells.append(f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
                    if k == "delivery_ratio":
                        ok = lo >= NI_MARGIN
                lines.append(f"| {group} | 合并 {len(pooled_a)} 对 | " + " | ".join(cells)
                             + f" | {'是' if ok else '否'} |")
    lines += ["", "注：合并行把 3 个训练种子 × 32 个场景作为配对样本，置信区间未计入同一场景重复使用的相关性，"
              "应与逐种子结果一起看。可靠性可接受 = 交付率配对差的 95% CI 下界 ≥ −0.005。"]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", action="store_true")
    p.add_argument("--parallel", type=int, default=3)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not a.summary:
        run_jobs(jobs(), a.parallel)
    text = summarise()
    (OUT / "summary.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
