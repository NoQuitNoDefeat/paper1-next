"""E13 pooled summaries (the per-fold / per-scene tables come from tools/e11_compare.py).

* flock30: the three folds pooled (96 test scenes, each fold's retrained models on their
  own test flight), main method (zero-shot and retrained) against every other policy;
* BonnMotion RPGM and Gauss-Markov: same comparisons per mobility model.

    .venv/bin/python tools/e13_data.py
"""

from __future__ import annotations

import json

import numpy as np

import e10_ns3 as base
from confirm import ROOT, ci

NI_MARGIN = -0.005
METRICS = (("delivery_ratio", "交付率", 4), ("e2e_delay_mean_s", "平均时延", 3))


def per_scene(spec: dict, scenarios: list[str], name: str, key: str) -> np.ndarray | None:
    """Seed-averaged per-scene values of one policy, concatenated over scenarios."""
    out = []
    for s in scenarios:
        rows = base.load_rows(spec, s, "lightweight", name)
        if rows is None:
            return None
        out.append(base.seed_avg(rows, key))
    return np.concatenate(out)


def table(spec: dict, scenarios: list[str], title: str, mains=("主方法", "主方法·重训")) -> list[str]:
    names = list(base.policies(spec))
    lines = [f"## {title}", "", "| 策略 | 交付率 | 平均时延 |", "| --- | --- | --- |"]
    for n in names:
        d = per_scene(spec, scenarios, n, "delivery_ratio")
        t = per_scene(spec, scenarios, n, "e2e_delay_mean_s")
        if d is not None:
            lines.append(f"| {n} | {d.mean():.4f} | {np.nanmean(t):.3f} |")
    lines.append("")
    for m in mains:
        lines += [f"**{m} 相对各策略的配对差**（{len(per_scene(spec, scenarios, m, 'delivery_ratio'))} 个场景）", "",
                  "| 对照 | 交付率差 | 平均时延差 | 可靠性可接受 |", "| --- | --- | --- | --- |"]
        for n in names:
            if n == m:
                continue
            cells = []
            for key, _, digits in METRICS:
                a, b = per_scene(spec, scenarios, m, key), per_scene(spec, scenarios, n, key)
                mm, lo, hi = ci(a - b)
                cells.append(f"{mm:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]")
            lb = ci(per_scene(spec, scenarios, m, "delivery_ratio") - per_scene(spec, scenarios, n, "delivery_ratio"))[1]
            lines.append(f"| {n} | {cells[0]} | {cells[1]} | {'是' if lb >= NI_MARGIN else '否'} |")
        lines.append("")
    return lines


def main() -> None:
    flock = json.loads((ROOT / "configs/experiments/e13_flock.json").read_text())
    bonn = json.loads((ROOT / "configs/experiments/e13_bonnmotion.json").read_text())
    lines = ["# E13 合并汇总", "", "学习型策略先按训练种子平均，再按场景配对；均值 [95% CI]。", ""]
    lines += table(flock, list(flock["scenarios"]), "群集 30 主条件（三折合并）")
    for s in bonn["scenarios"]:
        if base.load_rows(bonn, s, "lightweight", "主方法") is not None:
            lines += table(bonn, [s], f"BonnMotion `{s}`")
    lines.append(f"判定：可靠性可接受 = 交付率差 95% CI 下界 ≥ {NI_MARGIN}。没有做多重比较校正。")
    text = "\n".join(lines)
    (ROOT / "results/e13/summary_pooled.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
