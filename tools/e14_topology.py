"""E14 design check: method-independent topology statistics of training and test scenes.

For each scene family: fraction of node pairs within single-hop range, mean number of
neighbours, link on/off changes per pair and second, and the mean rate of change of the
pair distances (also divided by the single-hop range).  Synthetic scenes use the base
radio (161.1 m); flock30 flights use the E13 main condition (66 m), 16 and 30 drones,
16 random 10 s windows per flight.  No policy is run.

    .venv/bin/python tools/e14_topology.py      # writes results/e14/topology.md
"""

from __future__ import annotations

import numpy as np

from confirm import ROOT
from fanet_next.config import load_config
from fanet_next.scenario import SCENARIO
from fanet_next.scenario.trace import load_trace

FLIGHTS = ("4mps_diagonal", "6mps_circular", "6mps_obstacles", "8mps_circular")
SEEDS = range(1_000_000, 1_000_080)  # training seeds: never used by any test


def stats(pos: np.ndarray, rng_m: float, dt: float) -> np.ndarray:
    n = pos.shape[1]
    iu = np.triu_indices(n, 1)
    d = np.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[:, iu[0], iu[1]]
    inr = d <= rng_m
    rate = np.abs(np.diff(d, axis=0)).mean() / dt
    return np.array([inr.mean(), inr.sum(1).mean() * 2 / n, (inr[1:] != inr[:-1]).mean() / dt,
                     rate, rate / rng_m])


def synthetic(cfg_scenario: dict, label_of=None) -> dict[str, list[np.ndarray]]:
    src = SCENARIO.build(cfg_scenario)
    out: dict[str, list[np.ndarray]] = {}
    for s in SEEDS:
        sc = src.make(s)
        p = np.stack([sc.positions(k) for k in range(sc.horizon + 1)])
        out.setdefault(label_of(s) if label_of else "", []).append(stats(p, 161.1, sc.cycle_length))
    return out


def row(label: str, x: list[np.ndarray]) -> str:
    a = np.array(x)
    lo, hi = np.percentile(a, 5, 0), np.percentile(a, 95, 0)
    return (f"| {label} | {a[:, 0].mean():.2f}（{lo[0]:.2f}–{hi[0]:.2f}） | {a[:, 1].mean():.1f}（{lo[1]:.1f}–{hi[1]:.1f}） "
            f"| {a[:, 2].mean():.4f} | {a[:, 3].mean():.1f} | {a[:, 4].mean():.4f} |")


def main() -> None:
    lines = ["# E14 拓扑统计（与方法无关）", "",
             "| 场景 | 单跳内节点对比例（5–95%） | 平均邻居数（5–95%） | 链路通断变化 / 对 / s | 距离变化率 m/s | 距离变化率 / 单跳距离 |",
             "| --- | --- | --- | --- | --- | --- |"]
    base = load_config(ROOT / "configs/protocol_final.toml")["scenario"]
    for label, over in (("合成训练分布（5–30 m/s）", {}), ("合成 0–5 m/s", {"speed_mps": [0.0, 5.0]}),
                        ("合成 30 节点 300 m", {"num_nodes": 30})):
        lines.append(row(label, synthetic({**base, **over})[""]))
    mix_cfg = load_config(ROOT / "configs/explore/e14_broad.toml")["scenario"]
    mix = SCENARIO.build(mix_cfg)
    parts = synthetic(mix_cfg, lambda s: mix.component_of(s))
    lines.append(row("宽分布 · 原分布部分", parts[0]))
    lines.append(row("宽分布 · 低速紧凑部分", parts[1]))
    rng = np.random.default_rng(0)
    for f in FLIGHTS:
        t, pos = load_trace(str(ROOT / f"data/processed/vasarhelyi2018/{f}.npz"))
        for n in (16, 30):
            res = []
            for _ in range(16):
                times = rng.uniform(t[0], t[-1] - 10.02) + 0.02 * np.arange(501)
                nodes = np.sort(rng.choice(pos.shape[1], size=n, replace=False))
                p = np.stack([np.stack([np.interp(times, t, pos[:, i, j]) for j in range(3)], 1) for i in nodes], 1)
                res.append(stats(p, 66.0, 0.02))
            lines.append(row(f"群集 30 `{f}`，{n} 架（66 m）", res))
    text = "\n".join(lines)
    out = ROOT / "results/e14/topology.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
