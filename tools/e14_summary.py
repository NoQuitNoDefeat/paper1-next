"""E14 summary (exploration): broadened training distribution vs the original one.

Pre-registered comparisons (docs/experiments.md, E14):

1. main·宽 - LQ on speed_0_5, slow_compact, flock30 16 drones (three flights pooled) and
   30 drones (pooled): reliability acceptable when the delivery CI lower bound >= -0.005;
2. no regression: main·宽 - main (original) on default, same bound;
3. secondary: every learned method broad - original; main·宽 against every policy; on
   flock30 also against the main method retrained on the real flights (E13).

    .venv/bin/python tools/e14_summary.py      # writes results/e14/summary_pre.md
"""

from __future__ import annotations

from confirm import ROOT
from e10_ns3 import policies
from report_data import _cmp, _load_spec, _mean

NI = -0.005


def cell(d: dict, key: str, digits: int, scale: float) -> str:
    m, lo, hi = (d[key][x] * scale for x in ("m", "lo", "hi"))
    mark = "↑" if lo > 0 else ("↓" if hi < 0 else "=")
    return f"{m:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}] {mark}"


def main() -> None:
    syn = _load_spec("configs/experiments/e14_synthetic.json")
    fl = _load_spec("configs/experiments/e14_flock.json")
    f16 = [s for s in fl["scenarios"] if not s.endswith("_n30")]
    f30 = [s for s in fl["scenarios"] if s.endswith("_n30")]
    cols = [*((s, syn, [s]) for s in syn["scenarios"]), ("群集 16 架（合并）", fl, f16), ("群集 30 架（合并）", fl, f30)]
    cols += [(s, fl, [s]) for s in fl["scenarios"]]
    lines = ["# E14 汇总（探索）：宽训练分布", "",
             f"测试种子偏移 {syn['eval']['seed_offset']}；合成场景族每族 {syn['eval']['episodes']} 个场景，群集每次飞行 "
             f"{fl['eval']['episodes']} 个场景。学习型策略先按训练种子平均，再按场景配对；均值 [95% CI]，"
             "↑ 显著为正，↓ 显著为负，= 不显著。交付率差单位为个百分点。", ""]

    def table(title: str, pairs: list[tuple[str, str]], only: list[str] | None = None) -> None:
        nonlocal lines
        lines += [f"## {title}", "", "| 场景 | 比较 | 交付率差 | 平均时延差（s） | 可靠性可接受 |", "| --- | --- | --- | --- | --- |"]
        for label, sp, scen in cols:
            if only is not None and label not in only:
                continue
            for a, b in pairs:
                d = _cmp(sp, scen, a, b, ("delivery_ratio", "e2e_delay_mean_s"))
                if d is None:
                    continue
                ok = "是" if d["delivery_ratio"]["lo"] >= NI else "否"
                lines.append(f"| {label} | {a} − {b} | {cell(d, 'delivery_ratio', 2, 100)} "
                             f"| {cell(d, 'e2e_delay_mean_s', 3, 1)} | {ok} |")
        lines.append("")

    primary = ["speed_0_5", "slow_compact", "群集 16 架（合并）", "群集 30 架（合并）"]
    table("判定 1（主要）：主方法·宽 − LQ", [("主方法·宽", "longest_queue")], primary)
    table("判定 2（不退步）：主方法·宽 − 主方法（原分布）", [("主方法·宽", "主方法")], ["default"])
    table("对照：原分布的主方法 − LQ", [("主方法", "longest_queue")])
    table("次要：每个学习方法 宽 − 原", [("主方法·宽", "主方法"), ("Zhao-GCN·宽", "Zhao-GCN"), ("GRLinQ·宽", "GRLinQ"),
                                    ("独立决定PPO·宽", "独立决定PPO"), ("仅模仿·宽", "仅模仿")])
    table("次要：主方法·宽 − 宽分布训练的学习基线",
          [("主方法·宽", n) for n in ("Zhao-GCN·宽", "GRLinQ·宽", "独立决定PPO·宽", "仅模仿·宽", "backpressure")])
    table("次要：主方法·宽 − 在真实飞行上重新训练的主方法（E13）", [("主方法·宽", "主方法·重训")],
          ["群集 16 架（合并）", "群集 30 架（合并）", *fl["scenarios"]])
    lines += ["## 各策略的平均交付率", "", "| 策略 | " + " | ".join(c[0] for c in cols) + " |",
              "| --- |" + " --- |" * len(cols)]
    for n in policies(fl):
        vals = [_mean(sp, scen, n, "delivery_ratio") if n in policies(sp) else None for _, sp, scen in cols]
        lines.append(f"| {n} | " + " | ".join("–" if v is None else f"{v:.4f}" for v in vals) + " |")
    lines += ["", f"判定：可靠性可接受 = 交付率差 95% CI 下界 ≥ {NI}。没有做多重比较校正。"]
    text = "\n".join(lines)
    (ROOT / "results/e14/summary_pre.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
