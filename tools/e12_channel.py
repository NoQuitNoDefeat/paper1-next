"""E12 cross-condition summaries (the within-condition tables come from tools/e11_compare.py).

1. Channel effect per policy on the same scenes: C2 - C0 per scenario family and
   C1 - C0 on the default family (delivery, mean delay; plus plan size and the
   fraction of executed links that failed).
2. Sensitivity: main method minus LQ in every channel condition of
   configs/experiments/e12_sensitivity.json, next to C2 default on the same
   (first 32) scenes.

    .venv/bin/python tools/e12_channel.py
"""

from __future__ import annotations

import json

import numpy as np

import e10_ns3 as base
from confirm import ROOT, ci

CHANNEL = "configs/experiments/e12_channel.json"
SENS = "configs/experiments/e12_sensitivity.json"
FAMILIES = ("default", "load_5_15", "nodes24_same_density")
NI_MARGIN = -0.005


def _rows(spec, scenario, name, seeds=None):
    runs = base.load_rows(spec, scenario, "lightweight", name)
    if runs is None:
        return None
    if seeds is not None:
        runs = [[r for r in rows if r["seed"] in seeds] for rows in runs]
    return runs


def _cell(d: np.ndarray, digits: int) -> str:
    m, lo, hi = ci(d)
    mark = "↑" if lo > 0 else ("↓" if hi < 0 else "=")
    return f"{m:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}] {mark}"


def channel_effect(spec: dict) -> list[str]:
    names = list(base.policies(spec))
    lines = ["## 信道的影响：同一批场景上 C2 − C0、C1 − C0（每个策略）", "",
             "交付率差与平均时延差（s）；↑ 显著增加，↓ 显著减少，= 不显著。"
             "计划链路数与执行失败链路占比为两种条件下的均值。", ""]
    pairs = [(f"c2_{f}", f"c0_{f}", f"C2 − C0 · {f}") for f in FAMILIES]
    pairs.append(("c1_default", "c0_default", "C1 − C0 · default"))
    for new, ref, title in pairs:
        lines += [f"### {title}", "", "| 策略 | 交付率差 | 平均时延差 | 计划链路数 | 执行失败占比 |",
                  "| --- | --- | --- | --- | --- |"]
        for n in names:
            a, b = _rows(spec, new, n), _rows(spec, ref, n)
            if a is None or b is None:
                lines.append(f"| {n} | 未完成 | | | |")
                continue
            dd = base.seed_avg(a, "delivery_ratio") - base.seed_avg(b, "delivery_ratio")
            dl = base.seed_avg(a, "e2e_delay_mean_s") - base.seed_avg(b, "e2e_delay_mean_s")
            ps = (np.mean(base.seed_avg(b, "plan_size_mean")), np.mean(base.seed_avg(a, "plan_size_mean")))
            fail = np.mean(base.seed_avg(a, "exec_failed_link_frac"))
            lines.append(f"| {n} | {_cell(dd, 4)} | {_cell(dl, 3)} | {ps[0]:.2f} → {ps[1]:.2f} "
                         f"| {fail:.3f} |")
        lines.append("")
    return lines


def sensitivity(channel: dict, sens: dict) -> list[str]:
    main = sens.get("main", next(iter(sens["groups"])))
    lines = ["## 敏感性：主方法 − LQ（默认场景族前 32 个场景）", "",
             "| 条件 | 主方法交付率 | 交付率差 | 平均时延差 | 可靠性可接受 |",
             "| --- | --- | --- | --- | --- |"]
    ref_rows = _rows(channel, "c2_default", "longest_queue")
    seeds = None
    if ref_rows is not None:
        e = sens["eval"]
        all_seeds = sorted({r["seed"] for r in ref_rows[0]})
        seeds = set(all_seeds[:e["episodes"]])
    conds = [("C2（n 2.2，K 10 dB，中断 10%）", channel, "c2_default", seeds)]
    conds += [(s, sens, s, None) for s in sens["scenarios"]]
    for label, spec, scen, sd in conds:
        m, l = _rows(spec, scen, main, sd), _rows(spec, scen, "longest_queue", sd)
        if m is None or l is None:
            lines.append(f"| {label} | 未完成 | | | |")
            continue
        dd = base.seed_avg(m, "delivery_ratio") - base.seed_avg(l, "delivery_ratio")
        dl = base.seed_avg(m, "e2e_delay_mean_s") - base.seed_avg(l, "e2e_delay_mean_s")
        lb = ci(dd)[1]
        lines.append(f"| {label} | {np.mean(base.seed_avg(m, 'delivery_ratio')):.4f} | {_cell(dd, 4)} "
                     f"| {_cell(dl, 3)} | {'是' if lb >= NI_MARGIN else '否'} |")
    lines.append("")
    return lines


def main() -> None:
    channel = json.loads((ROOT / CHANNEL).read_text())
    sens = json.loads((ROOT / SENS).read_text())
    lines = ["# E12 跨条件汇总", "",
             f"test 种子偏移 {channel['eval']['seed_offset']}，排空 {channel['eval']['drain']} 周期；"
             "学习型策略先按训练种子平均，再按场景配对；均值 [95% CI]。", ""]
    lines += channel_effect(channel) + sensitivity(channel, sens)
    lines.append(f"判定：可靠性可接受 = 交付率差 95% CI 下界 ≥ {NI_MARGIN}。没有做多重比较校正。")
    text = "\n".join(lines)
    out = ROOT / "results/e12/summary_cross.md"
    out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
