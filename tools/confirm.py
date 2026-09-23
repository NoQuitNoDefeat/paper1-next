"""Held-out confirmation driven by an experiment spec (JSON).

For every scenario variant in the spec, evaluates each group's selected
checkpoints (running `fanet-next select` first where selection.json is missing),
fixed reference checkpoints and heuristics on the same seeds, then writes
paired comparisons to ``<out>/summary.md``.

Group statistics: per training seed (paired over scenarios) and a seed-averaged
row — each scenario's metric is first averaged over the group's training seeds,
then paired over scenarios — so no scenario is counted more than once.

    .venv/bin/python tools/confirm.py --spec configs/experiments/e6_ablation.json
    .venv/bin/python tools/confirm.py --spec ... --summary      # tables only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from math import sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLI = str(ROOT / ".venv" / "bin" / "fanet-next")
KEYS = [("delivery_ratio", "交付率", "{:.4f}"), ("termination_ratio", "终止比例", "{:.4f}"),
        ("ontime_1s", "1s 送达", "{:.4f}"), ("ontime_2s", "2s 送达", "{:.4f}"),
        ("e2e_delay_mean_s", "平均时延", "{:.3f}"), ("e2e_delay_p95_s", "p95 时延", "{:.3f}"),
        ("exec_failed_link_frac", "执行失败链路", "{:.3f}"), ("in_system_end", "结束时在网", "{:.1f}")]
PAIR_KEYS = ["delivery_ratio", "termination_ratio", "ontime_2s", "e2e_delay_mean_s", "e2e_delay_p95_s"]
NI_MARGIN = -0.005
T95 = {n: t for n, t in [(8, 2.36), (16, 2.13), (32, 2.04), (64, 2.00)]}


def slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)


def eval_args(spec: dict, overrides: list[str]) -> list[str]:
    e = spec["eval"]
    args = ["--split", e["split"], "--seed-offset", str(e["seed_offset"]), "--episodes",
            str(e["episodes"]), "--drain", str(e["drain"]), "--threads", str(e.get("threads", 3))]
    for o in overrides:
        args += ["--set", o]
    return args


def build_jobs(spec: dict) -> tuple[list[list[str]], list[list[str]]]:
    """(selection jobs, evaluation jobs) still missing."""
    sel_jobs, eval_jobs = [], []
    for runs in spec["groups"].values():
        for run in runs:
            if not (ROOT / run / "selection.json").exists():
                sel_jobs.append([CLI, "select", "--run-dir", run, "--threads", "3"])
    for variant, overrides in spec["variants"].items():
        out = ROOT / spec["out"] / variant
        out.mkdir(parents=True, exist_ok=True)
        ea = eval_args(spec, overrides)
        base = out / "baselines.json"
        if not base.exists():
            eval_jobs.append([CLI, "eval", "--config", spec["baseline_config"], "--policies",
                              *spec["baselines"], *ea, "--out", str(base)])
        for name, ckpt in spec.get("fixed", {}).items():
            f = out / f"fixed_{slug(name)}.json"
            if not f.exists():
                run_dir = str(Path(ckpt).parent.parent)
                eval_jobs.append([CLI, "eval", "--run-dir", run_dir, "--checkpoint", ckpt,
                                  "--policies", "ppo", *ea, "--out", str(f)])
        for runs in spec["groups"].values():
            for run in runs:
                f = out / f"{slug(run)}.json"
                if not f.exists():
                    eval_jobs.append([CLI, "eval", "--run-dir", run, "--checkpoint", "@selected",
                                      "--policies", "ppo", *ea, "--out", str(f)])
    return sel_jobs, eval_jobs


def resolve(cmd: list[str]) -> list[str]:
    if "@selected" in cmd:
        run = cmd[cmd.index("--run-dir") + 1]
        sel = json.loads((ROOT / run / "selection.json").read_text())
        cmd = [sel["selected"]["checkpoint"] if c == "@selected" else c for c in cmd]
    return cmd


def run_jobs(jobs: list[list[str]], parallel: int, log) -> None:
    running: list[subprocess.Popen] = []
    for cmd in jobs:
        while len(running) >= parallel:
            running = [p for p in running if p.poll() is None]
            time.sleep(2)
        cmd = resolve(cmd)
        target = cmd[cmd.index("--out") + 1] if "--out" in cmd else cmd[cmd.index("--run-dir") + 1]
        fh = open(f"{target}.{cmd[1]}.log", "w")
        running.append(subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT))
        log(f"started {cmd[1]} {target}")
    for p in running:
        p.wait()


# ------------------------------------------------------------------ statistics
def ci(d: np.ndarray) -> tuple[float, float, float]:
    d = d[~np.isnan(d)]
    n = len(d)
    t = T95.get(n, 2.0 if n > 32 else 2.2)
    half = t * d.std(ddof=1) / sqrt(n) if n > 1 else float("nan")
    return float(d.mean()), float(d.mean() - half), float(d.mean() + half)


def by_seed(rows: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in sorted(rows, key=lambda r: r["seed"])], float)


def summarise(spec: dict) -> str:
    lines = [f"# {spec['name']}", "",
             f"评估：{spec['eval']['split']} 种子偏移 {spec['eval']['seed_offset']}，"
             f"{spec['eval']['episodes']} 个场景，排空 {spec['eval']['drain']} 周期。"
             "“种子平均”行：每个场景先对训练种子取平均，再按场景配对。", ""]
    for variant, overrides in spec["variants"].items():
        out = ROOT / spec["out"] / variant
        lines += [f"## 变体 `{variant}`" + (f"（{', '.join(overrides)}）" if overrides else ""), ""]
        policies: dict[str, list[list[dict]]] = {}
        base = json.loads((out / "baselines.json").read_text())["results"]
        for b in spec["baselines"]:
            policies[b] = [base[b]["rows"]]
        for name in spec.get("fixed", {}):
            f = out / f"fixed_{slug(name)}.json"
            policies[name] = [json.loads(f.read_text())["results"]["ppo"]["rows"]]
        selected = {}
        for group, runs in spec["groups"].items():
            policies[group] = []
            for run in runs:
                policies[group].append(json.loads((out / f"{slug(run)}.json").read_text())["results"]["ppo"]["rows"])
                s = json.loads((ROOT / run / "selection.json").read_text())
                selected.setdefault(group, []).append(
                    f"{s['selected']['iteration']}" + ("" if s["any_eligible"] else "*"))
        lines += ["| 策略 | 选中迭代 | " + " | ".join(k[1] for k in KEYS) + " |",
                  "| --- | --- | " + " | ".join("---" for _ in KEYS) + " |"]
        for name, seeds in policies.items():
            vals = [np.nanmean([np.nanmean(by_seed(rows, k)) for rows in seeds]) for k, _, _ in KEYS]
            lines.append(f"| {name} | {', '.join(selected.get(name, []))} | "
                         + " | ".join(fmt.format(v) for v, (_, _, fmt) in zip(vals, KEYS)) + " |")
        lines.append("")
        for ref in spec["compare"]:
            ref_mean = {k: np.mean([by_seed(r, k) for r in policies[ref]], axis=0) for k in PAIR_KEYS}
            lines += [f"**相对 {ref} 的配对差**（均值 [95% CI]）", "",
                      "| 策略 | 种子 | " + " | ".join(dict((k, l) for k, l, _ in KEYS)[k] for k in PAIR_KEYS)
                      + " | 可靠性可接受 |",
                      "| --- | --- | " + " | ".join("---" for _ in PAIR_KEYS) + " | --- |"]
            for name, seeds in policies.items():
                if name == ref:
                    continue
                rows_out = [(f"s{i}", {k: by_seed(r, k) for k in PAIR_KEYS}) for i, r in enumerate(seeds)] \
                    if len(seeds) > 1 else []
                rows_out.append(("种子平均" if len(seeds) > 1 else "-",
                                 {k: np.mean([by_seed(r, k) for r in seeds], axis=0) for k in PAIR_KEYS}))
                for label, vals in rows_out:
                    cells, ok = [], None
                    for k in PAIR_KEYS:
                        m, lo, hi = ci(vals[k] - ref_mean[k])
                        cells.append(f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
                        if k == "delivery_ratio":
                            ok = lo >= NI_MARGIN
                    lines.append(f"| {name} | {label} | " + " | ".join(cells) + f" | {'是' if ok else '否'} |")
            lines.append("")
    lines += ["注：`*` 表示该运行没有检查点满足入选条件，取交付率最高者。可靠性可接受 = 交付率配对差 95% CI "
              "下界 ≥ −0.005（相对 longest_queue 时才是登记的判定标准）。"]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--parallel", type=int, default=3)
    p.add_argument("--summary", action="store_true")
    a = p.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    out = ROOT / spec["out"]
    out.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with (out / "confirm.log").open("a") as fh:
            fh.write(line + "\n")

    if not a.summary:
        sel_jobs, eval_jobs = build_jobs(spec)
        log(f"{len(sel_jobs)} selections, {len(eval_jobs)} evaluations")
        run_jobs(sel_jobs, a.parallel, log)
        run_jobs(eval_jobs, a.parallel, log)
    text = summarise(spec)
    (out / "summary.md").write_text(text + "\n")
    log(f"wrote {out / 'summary.md'}")
    print(text)


if __name__ == "__main__":
    main()
