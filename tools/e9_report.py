"""E9 test-set report: pre-registered claims (docs/experiments.md, E9).

Seed-averaged, scenario-paired differences (64 test scenarios per family) of the
final method against longest_queue, hol_weighted, the original gated-sum design
and imitation only; plus per-policy means.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from confirm import by_seed, ci, slug  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "configs/experiments/e9_test.json").read_text())
MAIN = "主方法（无摘要）"
KEYS = [("delivery_ratio", "交付率"), ("e2e_delay_mean_s", "平均时延"), ("e2e_delay_p95_s", "p95 时延"),
        ("ontime_2s", "2s 送达")]


def policy_rows(out: Path) -> dict[str, list[list[dict]]]:
    base = json.loads((out / "baselines.json").read_text())["results"]
    pol = {b: [base[b]["rows"]] for b in SPEC["baselines"]}
    for name in SPEC["fixed"]:
        pol[name] = [json.loads((out / f"fixed_{slug(name)}.json").read_text())["results"]["ppo"]["rows"]]
    for g, runs in SPEC["groups"].items():
        pol[g] = [json.loads((out / f"{slug(r)}.json").read_text())["results"]["ppo"]["rows"] for r in runs]
    return pol


def avg(seeds: list[list[dict]], key: str) -> np.ndarray:
    return np.mean([by_seed(r, key) for r in seeds], axis=0)


def main() -> None:
    claims = []
    for variant in SPEC["variants"]:
        pol = policy_rows(ROOT / SPEC["out"] / variant)
        print(f"\n## {variant}")
        print("均值：" + "  ".join(f"{p}: dr={np.mean(avg(s, 'delivery_ratio')):.4f} "
                                  f"delay={np.mean(avg(s, 'e2e_delay_mean_s')):.3f} "
                                  f"p95={np.mean(avg(s, 'e2e_delay_p95_s')):.3f}" for p, s in pol.items()))
        for ref in SPEC["compare"]:
            cells = []
            for k, label in KEYS:
                m, lo, hi = ci(avg(pol[MAIN], k) - avg(pol[ref], k))
                cells.append(f"{label} {m:+.4f} [{lo:+.4f},{hi:+.4f}]")
                if ref == "longest_queue":
                    claims.append((variant, k, m, lo, hi))
            print(f"  主方法 − {ref}: " + "  ".join(cells))
    print("\n预登记判定（相对 longest_queue）")
    for variant in SPEC["variants"]:
        c = {k: (m, lo, hi) for v, k, m, lo, hi in claims if v == variant}
        rel = c["delivery_ratio"][1] >= -0.005
        md = "显著更低" if c["e2e_delay_mean_s"][2] < 0 else ("显著更高" if c["e2e_delay_mean_s"][1] > 0 else "不显著")
        p95 = "显著更低" if c["e2e_delay_p95_s"][2] < 0 else ("显著更高" if c["e2e_delay_p95_s"][1] > 0 else "不显著")
        print(f"  {variant:22} 可靠性可接受={'是' if rel else '否'}（下界 {c['delivery_ratio'][1]:+.4f}）"
              f"  平均时延{md}  p95{p95}")


if __name__ == "__main__":
    main()
