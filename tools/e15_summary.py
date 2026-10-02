"""E15 summary (exploration): 5 dB known shadowing as a controlled variable.

Pre-registered comparisons (docs/experiments.md, E15):

1. main - LQ under C2S5 per synthetic family and on flock30 (three flights pooled):
   reliability acceptable when the delivery CI lower bound >= -0.005;
2. the shadowing effect C2S5 - C2 per policy on the same scenes and fading draws, and the
   difference in differences (main - LQ under C2S5) - (main - LQ under C2);
3. ns-3: fidelity within +-0.01 and the same direction of main - LQ;
4. secondary: shadowing strength (sigma 3 / 7 dB, sigma 5 dB with K = 5 dB) and the models
   retrained on the flock flights.

    .venv/bin/python tools/e15_summary.py      # writes results/e15/summary_pre.md
"""

from __future__ import annotations

from confirm import ROOT, ci
from e10_ns3 import build_jobs, policies
from report_data import _cmp, _first_seeds, _load_spec, _mean, _per_scene

NI = -0.005
FAMS = ("default", "load_5_15", "nodes24_same_density")


def cell(d: dict, digits: int, scale: float) -> str:
    m, lo, hi = (d[x] * scale for x in ("m", "lo", "hi"))
    mark = "↑" if lo > 0 else ("↓" if hi < 0 else "=")
    return f"{m:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}] {mark}"


def main() -> None:
    ch, fl = _load_spec("configs/experiments/e15_channel.json"), _load_spec("configs/experiments/e15_flock.json")
    sg, ns3 = _load_spec("configs/experiments/e15_sigma.json"), _load_spec("configs/experiments/e15_ns3.json")
    folds = sorted({s.split("_", 1)[1] for s in fl["scenarios"]})
    groups = [*((f, ch, [f"c2_{f}"], [f"c2s5_{f}"]) for f in FAMS),
              ("群集 30（三次飞行合并）", fl, [f"c2_{f}" for f in folds], [f"c2s5_{f}" for f in folds])]
    groups += [(f"群集 {f}", fl, [f"c2_{f}"], [f"c2s5_{f}"]) for f in folds]
    lines = ["# E15 汇总（探索）：阴影 σ = 5 dB 作为控制变量", "",
             f"测试种子偏移 {ch['eval']['seed_offset']}；C2 与 C2S5 用同一批场景、同一组快衰落样本。"
             "学习型策略先按训练种子平均，再按场景配对；均值 [95% CI]，↑ 显著为正，↓ 显著为负，= 不显著。"
             "交付率差单位为个百分点。", ""]
    lines += ["## 判定 1（主要）：C2S5 下主方法 − LQ", "",
              "| 场景 | 主方法交付率 | 交付率差 | 平均时延差（s） | 可靠性可接受 |", "| --- | --- | --- | --- | --- |"]
    for label, sp, _, s5 in groups:
        d = _cmp(sp, s5, "主方法", "longest_queue", ("delivery_ratio", "e2e_delay_mean_s"))
        if d is None:
            lines.append(f"| {label} | 未完成 | | | |")
            continue
        lines.append(f"| {label} | {_mean(sp, s5, '主方法', 'delivery_ratio'):.4f} | {cell(d['delivery_ratio'], 2, 100)} "
                     f"| {cell(d['e2e_delay_mean_s'], 3, 1)} | {'是' if d['delivery_ratio']['lo'] >= NI else '否'} |")
    lines += ["", "## 判定 2：阴影的影响（C2S5 − C2，同一批场景）", "",
              "每格：C2 → C2S5 的平均交付率；交付率差 [CI]；平均时延差 [CI]（s）。", ""]
    for label, sp, s2, s5 in groups:
        lines += [f"### {label}", "", "| 策略 | C2 → C2S5 | 交付率差 | 平均时延差（s） |", "| --- | --- | --- | --- |"]
        for n in policies(sp):
            d = _cmp(sp, s5, n, n, ("delivery_ratio", "e2e_delay_mean_s"), scenarios_b=s2)
            if d is None:
                continue
            lines.append(f"| {n} | {_mean(sp, s2, n, 'delivery_ratio'):.4f} → {_mean(sp, s5, n, 'delivery_ratio'):.4f} "
                         f"| {cell(d['delivery_ratio'], 2, 100)} | {cell(d['e2e_delay_mean_s'], 3, 1)} |")
        a5 = _per_scene(sp, s5, "主方法", "delivery_ratio")
        b5 = _per_scene(sp, s5, "longest_queue", "delivery_ratio")
        a2 = _per_scene(sp, s2, "主方法", "delivery_ratio")
        b2 = _per_scene(sp, s2, "longest_queue", "delivery_ratio")
        if all(x is not None for x in (a5, b5, a2, b2)):
            m, lo, hi = ci((a5 - b5) - (a2 - b2))
            lines += ["", f"差中差（主方法 − LQ 在 C2S5 下减在 C2 下）：{cell({'m': m, 'lo': lo, 'hi': hi}, 2, 100)} 个百分点"]
        lines.append("")
    seeds = _first_seeds(ch, "c2_default", sg["eval"]["episodes"])
    lines += ["## 阴影强度（默认场景族前 32 个场景）", "",
              "| 条件 | 主方法交付率 | LQ 交付率 | 交付率差 | 平均时延差（s） |", "| --- | --- | --- | --- | --- |"]
    for label, sp, scen, sd in [("无阴影（C2）", ch, "c2_default", seeds), ("σ = 3 dB", sg, "c2s3_default", None),
                                ("σ = 5 dB", ch, "c2s5_default", seeds), ("σ = 7 dB", sg, "c2s7_default", None),
                                ("σ = 5 dB 且 K = 5 dB", sg, "c2s5k5_default", None)]:
        d = _cmp(sp, [scen], "主方法", "longest_queue", ("delivery_ratio", "e2e_delay_mean_s"), sd)
        if d is None:
            lines.append(f"| {label} | 未完成 | | | |")
            continue
        lines.append(f"| {label} | {_mean(sp, [scen], '主方法', 'delivery_ratio', sd):.4f} "
                     f"| {_mean(sp, [scen], 'longest_queue', 'delivery_ratio', sd):.4f} "
                     f"| {cell(d['delivery_ratio'], 2, 100)} | {cell(d['e2e_delay_mean_s'], 3, 1)} |")
    lines += ["", "## 判定 3：ns-3 复核（C2S5 默认场景族）", ""]
    if build_jobs(ns3):
        lines.append("未完成。")
    else:
        lines += ["| 项目 | 交付率差 | 平均时延差（s） |", "| --- | --- | --- |"]
        for x in ns3["executors"]:
            d = _cmp(ns3, ["c2s5_default"], "主方法", "longest_queue", ("delivery_ratio", "e2e_delay_mean_s"), executor=x)
            lines.append(f"| 主方法 − LQ（{x}） | {cell(d['delivery_ratio'], 2, 100)} | {cell(d['e2e_delay_mean_s'], 3, 1)} |")
        for n in policies(ns3):
            a = _per_scene(ns3, ["c2s5_default"], n, "delivery_ratio", executor="ns3")
            b = _per_scene(ns3, ["c2s5_default"], n, "delivery_ratio", executor="lightweight")
            m, lo, hi = ci(a - b)
            lines.append(f"| {n}：ns-3 − 轻量 | {cell({'m': m, 'lo': lo, 'hi': hi}, 2, 100)} | |")
    lines += ["", f"判定：可靠性可接受 = 交付率差 95% CI 下界 ≥ {NI}；ns-3 保真度 = 交付率差 CI 在 ±1 个百分点内。"
              "没有做多重比较校正。"]
    text = "\n".join(lines)
    (ROOT / "results/e15/summary_pre.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
