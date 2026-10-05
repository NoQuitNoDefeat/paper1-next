"""Split packet losses by cause (no method change): buffer overflow of new packets at the source,
of relay arrivals (packets transmitted this cycle into a full next queue at the receiver), and of
re-homed packets (a route update moved queued packets to the new next-hop queue at the same node,
which was full); plus waiting-area overflow and timeout.  Shares are of all born packets.

    .venv/bin/python tools/drop_causes.py load_high results/e6/no_set_summary-s0 --episodes 16
    .venv/bin/python tools/drop_causes.py load_high longest_queue --episodes 16
"""

from __future__ import annotations

import argparse
import collections
import json

from confirm import ROOT
from fanet_next.config import load_config
from fanet_next.experiment.assemble import build_env, build_policy, policy_from_checkpoint
from fanet_next.experiment.evaluate import eval_scenario, split_seeds
from fanet_next.loop import EndType
from fanet_next.training.checkpoint import load_checkpoint

FAMILIES = {"default": {}, "load_high": {"flow_rate_pps": [30.0, 45.0]},
            "nodes24_same_density": {"num_nodes": 24, "area_m": 367.4}}


def policy_and_config(which: str):
    if (ROOT / which / "selection.json").exists():
        sel = json.loads((ROOT / which / "selection.json").read_text())["selected"]["checkpoint"]
        data = load_checkpoint(ROOT / sel)
        return data["config"], policy_from_checkpoint(data["config"], data)
    cfg = load_config(str(ROOT / "configs/protocol_final.toml"))
    return cfg, build_policy(cfg, which, seed=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=list(FAMILIES))
    ap.add_argument("policy", help="a run directory (selected checkpoint) or a registered policy name")
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seed-offset", type=int, default=16)
    ap.add_argument("--drain", type=int, default=1000)
    a = ap.parse_args()
    cfg, policy = policy_and_config(a.policy)
    sc = eval_scenario(cfg, a.drain, overrides=FAMILIES[a.family])
    env = build_env(cfg, run_id="drops", build_graph=policy.needs_graph, scenario_override=sc)
    tot = collections.Counter()
    for i, seed in enumerate(split_seeds("dev", a.episodes, a.seed_offset)):
        env.reset(seed, episode=i)
        be = env.backend
        while True:
            queued = {p.pid: (p, p.node) for q in be.queues for p in q}  # packet and its node now
            tr = env.step(policy.act([env.current], mode="greedy")[0].actions)
            f = tr.facts
            tot["born"] += f.births
            tot["delivered"] += len(f.delivered_ids)
            for reason, ids in f.terminations.items():
                for pid in ids:
                    if reason != "queue_overflow":
                        tot[reason] += 1
                    elif pid not in queued:
                        tot["overflow_birth"] += 1
                    else:  # transmitted this cycle (its node changed) or re-homed at the same node
                        pkt, node0 = queued[pid]
                        tot["overflow_relay_arrival" if pkt.node != node0 else "overflow_rehome"] += 1
            if tr.end is not EndType.CONTINUE:
                break
    born = tot.pop("born")
    out = {"family": a.family, "policy": a.policy, "episodes": a.episodes, "born": born,
           "share_of_born_pct": {k: round(100 * v / born, 2) for k, v in sorted(tot.items())}}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
