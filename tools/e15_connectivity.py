"""E15 mechanism check (no policy is run): links the scheduler may use and flows that have a
route at the start of an episode, default family, 40 training seeds.

Shadowing with a zero dB mean (the convention of E15 and of measured path-loss fits) raises
the linear mean gain by sigma^2 / (2 xi) dB (xi = 10 / ln 10) and adds long links; the
"power-normalised" rows shift the dB mean by -sigma^2 / (2 xi) so that the mean received
power equals C2's, leaving only the spread.

    .venv/bin/python tools/e15_connectivity.py      # writes results/e15/connectivity.md
"""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.sparse.csgraph import shortest_path

from confirm import ROOT
from fanet_next.config import load_config
from fanet_next.physics import db_to_lin
from fanet_next.scenario import CHANNEL, SCENARIO

XI = 10 / math.log(10)
SEEDS = range(1_000_000, 1_000_040)


def stats(radio: dict, channel: dict) -> tuple[float, float, float]:
    base = load_config(ROOT / "configs/protocol_final.toml")["scenario"]
    src = SCENARIO.build({**base, "radio": {**base["radio"], **radio}})
    frac, reach, hops = [], [], []
    for s in SEEDS:
        sc = src.make(s)
        ch = CHANNEL.build(channel)
        rng = np.random.default_rng(s)
        ch.reset(sc.num_nodes, rng)
        g = ch.estimate(sc.positions(0), sc.radio, rng)
        snr = db_to_lin(sc.radio.tx_power_dbm) * g / db_to_lin(sc.radio.noise_dbm)
        adj = snr >= sc.radio.threshold * db_to_lin(ch.fade_margin_db)
        np.fill_diagonal(adj, False)
        frac.append(adj[np.triu_indices(sc.num_nodes, 1)].mean())
        d = shortest_path(adj.astype(float), unweighted=True)
        fl = np.array(sc.flows)
        dd = d[fl[:, 0], fl[:, 1]]
        reach.append(np.isfinite(dd).mean())
        hops.append(dd[np.isfinite(dd)].mean())
    return float(np.mean(frac)), float(np.mean(reach)), float(np.mean(hops))


def main() -> None:
    c2 = {"pathloss_exponent": 2.2, "tx_power_dbm": 13.379}
    rows = [("C0（原信道）", {}, {"type": "ideal"}), ("C2", c2, {"type": "rician", "k_factor_db": 10.0})]
    for sigma in (3.0, 5.0, 7.0):
        rows.append((f"C2 + 阴影 σ = {sigma:g} dB", c2, {"type": "rician", "k_factor_db": 10.0, "shadowing_db": sigma}))
    rows.append(("C2 + 阴影 σ = 5 dB，K = 5 dB", c2, {"type": "rician", "k_factor_db": 5.0, "shadowing_db": 5.0}))
    for sigma in (3.0, 5.0, 7.0):
        shift = -sigma ** 2 / (2 * XI)
        rows.append((f"C2 + 阴影 σ = {sigma:g} dB，功率归一（dB 均值 {shift:+.2f}）",
                     {**c2, "tx_power_dbm": c2["tx_power_dbm"] + shift},
                     {"type": "rician", "k_factor_db": 10.0, "shadowing_db": sigma}))
    table = []
    lines = ["# E15 机理检查：可用链路与可达业务（与方法无关）", "",
             "默认场景族，40 个训练种子，回合开始时。可用链路 = 平均 SNR 达到规划门限的节点对。", "",
             "| 信道 | 可用节点对比例 | 有路由的业务比例 | 平均跳数 |", "| --- | --- | --- | --- |"]
    for label, radio, channel in rows:
        f, r, h = stats(radio, channel)
        table.append({"label": label, "usable": f, "routable": r, "hops": h})
        lines.append(f"| {label} | {f:.3f} | {r:.3f} | {h:.2f} |")
    text = "\n".join(lines)
    (ROOT / "results/e15/connectivity.md").write_text(text + "\n")
    (ROOT / "results/e15/connectivity.json").write_text(json.dumps(table, ensure_ascii=False, indent=1))
    print(text)


if __name__ == "__main__":
    main()
