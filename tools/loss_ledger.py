"""X0 loss ledger (stage 1a, docs/next-method.md section 7): where delivery is lost under a deadline,
per packet, for one policy, over all born packets and over the steady-state window.

Reports fates (delivered; deadline expiry at source / relay / late / waiting; overflow kinds),
successful transmissions per born packet, the share spent on packets later lost, the share spent
after a packet became hopeless under the current routes, the pipeline efficiency (delivered among
packets that left their source; multi-hop only), the remaining time of multi-hop packets when they
leave their source and arrive at each relay, the remaining time at the last hop of packets that
then expired at a relay, and the offered hop load (born x route hops) over successful
transmissions.

    .venv/bin/python tools/loss_ledger.py load_high lq_lasthop --deadline 1 --episodes 16 --seed-offset 1000
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from confirm import ROOT
from fanet_next.config import apply_override, load_config
from fanet_next.experiment.assemble import build_env, build_policy
from fanet_next.experiment.evaluate import eval_scenario, split_seeds
from fanet_next.loop import EndType
from packet_tracker import PacketTracker, steady_window

FAMILIES = {"default": {}, "load_high": {"flow_rate_pps": [30.0, 45.0]},
            "nodes24": {"num_nodes": 24, "area_m": 367.4}}
R5 = ['backend.service_order="fewest_hops"', "candidates.require_room=true", "backend.room_aware_service=true"]


def setup(family: str, deadline: float, rules: str, sets: list[str], horizon: int | None):
    cfg = load_config(str(ROOT / "configs/protocol_final.toml"))
    for item in (R5 if rules == "R5" else []) + [f"backend.deadline_s={deadline}"] + sets:
        apply_override(cfg, item)
    stop = int(horizon or cfg["scenario"]["horizon"])
    over = {**FAMILIES[family], "horizon": stop}
    cyc = float(cfg["scenario"]["cycle_length"])
    drain = int(np.ceil(deadline / cyc)) + 5
    sc = eval_scenario(cfg, drain, overrides=over)
    return cfg, sc, steady_window(deadline, cyc, stop)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=list(FAMILIES))
    ap.add_argument("policy", nargs="?", default="lq_lasthop")
    ap.add_argument("--deadline", type=float, default=1.0)
    ap.add_argument("--rules", default="R5", choices=["R0", "R5"])
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seed-offset", type=int, default=1000)
    ap.add_argument("--horizon", type=int, default=None, help="traffic cycles (default: config horizon)")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    cfg, sc, window = setup(a.family, a.deadline, a.rules, a.set, a.horizon)
    policy = build_policy(cfg, a.policy, seed=0)
    env = build_env(cfg, run_id="x0", build_graph=policy.needs_graph, scenario_override=sc)
    rows = []
    for i, seed in enumerate(split_seeds("dev", a.episodes, a.seed_offset)):
        env.reset(seed, episode=i)
        tk = PacketTracker(a.deadline)
        while True:
            tr = tk.step(env, policy.act([env.current], mode="greedy")[0].actions)
            if tr.end is not EndType.CONTINUE:
                break
        rows.append({"seed": seed, "all": tk.summary(), "steady": tk.summary(window)})

    def pooled(key):
        out = {}
        for k in ("delivered", "tx_wasted_share", "tx_after_hopeless_share", "hopeless_share", "pipeline_efficiency",
                  "pipeline_efficiency_multihop", "offered_hop_load_over_tx", "tx_per_born"):
            out[k] = float(np.mean([r[key][k] for r in rows]))  # mean over episodes
        fates = sorted({f for r in rows for f in r[key]["fates"]})
        out["fates"] = {f: float(np.mean([r[key]["fates"].get(f, 0.0) for r in rows])) for f in fates}
        return out
    print(json.dumps({"family": a.family, "policy": a.policy, "deadline": a.deadline, "rules": a.rules,
                      "episodes": a.episodes, "seed_offset": a.seed_offset, "window": window, "sets": a.set,
                      "mean_all": pooled("all"), "mean_steady": pooled("steady"), "rows": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
