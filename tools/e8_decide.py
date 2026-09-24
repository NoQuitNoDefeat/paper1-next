"""Apply the pre-registered E8 decision rule (docs/experiments.md, E8).

A summary redesign earns its place when, versus "no summary" (seed-averaged,
scenario-paired): the delivery CI lower bound is >= -0.005 in all five
scenarios, and the mean delay is significantly lower in >= 3 scenarios and
significantly higher in none.  Also reports p95 delay and the check against
longest_queue.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from confirm import by_seed, ci, slug  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPECS = ["configs/experiments/e8_summary_default.json", "configs/experiments/e8_summary_hard.json"]
REF, CANDS = "无摘要", ["归一化摘要", "控制器动态特征", "完整模型（门控求和）"]


def group_mean(out: Path, runs: list[str], key: str) -> np.ndarray:
    return np.mean([by_seed(json.loads((out / f"{slug(r)}.json").read_text())["results"]["ppo"]["rows"], key)
                    for r in runs], axis=0)


def main() -> None:
    table = {c: [] for c in CANDS}
    for spec_path in SPECS:
        spec = json.loads((ROOT / spec_path).read_text())
        for variant in spec["variants"]:
            out = ROOT / spec["out"] / variant
            lq = json.loads((out / "baselines.json").read_text())["results"]["longest_queue"]["rows"]
            ref = {k: group_mean(out, spec["groups"][REF], k) for k in
                   ("delivery_ratio", "e2e_delay_mean_s", "e2e_delay_p95_s")}
            for c in CANDS:
                g = {k: group_mean(out, spec["groups"][c], k) for k in ref}
                dr = ci(g["delivery_ratio"] - ref["delivery_ratio"])
                md = ci(g["e2e_delay_mean_s"] - ref["e2e_delay_mean_s"])
                p95 = ci(g["e2e_delay_p95_s"] - ref["e2e_delay_p95_s"])
                vs_lq = ci(g["delivery_ratio"] - by_seed(lq, "delivery_ratio"))
                table[c].append((variant, dr, md, p95, vs_lq))
    print(f"相对「{REF}」（种子平均，32 场景配对；均值 [95% CI]）")
    for c, rows in table.items():
        print(f"\n== {c}")
        for v, dr, md, p95, lq in rows:
            print(f"  {v:22} 交付率 {dr[0]:+.4f} [{dr[1]:+.4f},{dr[2]:+.4f}]  平均时延 {md[0]:+.3f} "
                  f"[{md[1]:+.3f},{md[2]:+.3f}]  p95 {p95[0]:+.3f} [{p95[1]:+.3f},{p95[2]:+.3f}]  "
                  f"相对LQ交付率下界 {lq[1]:+.4f}")
        ok_dr = all(dr[1] >= -0.005 for _, dr, _, _, _ in rows)
        better = sum(md[2] < 0 for _, _, md, _, _ in rows)
        worse = sum(md[1] > 0 for _, _, md, _, _ in rows)
        ok_lq = all(lq[1] >= -0.005 for *_, lq in rows)
        passed = ok_dr and better >= 3 and worse == 0
        print(f"  -> 交付率全部达标={ok_dr}  时延显著更低 {better}/5  显著更高 {worse}/5  "
              f"相对LQ可靠性={ok_lq}  判定：{'通过' if passed else '未通过'}"
              f"  平均时延改进均值={np.mean([md[0] for _, _, md, _, _ in rows]):+.3f} s")


if __name__ == "__main__":
    main()
