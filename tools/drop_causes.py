"""Split packet losses by cause (no method change): buffer overflow of new packets at the source,
of relay arrivals (packets transmitted this cycle into a full next queue at the receiver), and of
re-homed packets (a route update moved queued packets to the new next-hop queue at the same node,
which was full), the latter split by where the packet was (its source or a relay); plus
waiting-area overflow and timeout.  Shares are of all born packets.  Also the mean number of
queued packets at their source and at relays per cycle of the traffic period (drain excluded).
``--set`` applies further overrides (rules or environment options of stage 0).

    .venv/bin/python tools/drop_causes.py load_high results/e6/no_set_summary-s0 --episodes 16
    .venv/bin/python tools/drop_causes.py load_high longest_queue --episodes 16
"""

from __future__ import annotations

import argparse
import collections
import json

from confirm import ROOT
from fanet_next.config import apply_override, load_config
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
    ap.add_argument("--set", action="append", default=[], help="override, e.g. backend.rehome_overflow=\"exceed\"")
    a = ap.parse_args()
    cfg, policy = policy_and_config(a.policy)
    for item in a.set:
        apply_override(cfg, item)
    sc = eval_scenario(cfg, a.drain, overrides=FAMILIES[a.family])
    env = build_env(cfg, run_id="drops", build_graph=policy.needs_graph, scenario_override=sc)
    tot = collections.Counter()
    rows = []
    for i, seed in enumerate(split_seeds("dev", a.episodes, a.seed_offset)):
        env.reset(seed, episode=i)
        be = env.backend
        before = collections.Counter(tot)
        while True:
            queued = {p.pid: (p, p.node) for q in be.queues for p in q}  # packet and its node now
            if not be.sc.traffic_done(be.cycle):  # occupancy over the traffic period only
                at_src = sum(1 for p, node in queued.values() if node == p.src)
                tot["queued_at_source_cycles"] += at_src
                tot["queued_at_relay_cycles"] += len(queued) - at_src
                tot["cycles"] += 1
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
                        if pkt.node != node0:
                            tot["overflow_relay_arrival"] += 1  # dropped at the receiver, a relay
                        else:  # re-homed where it was: at its source or at a relay
                            tot["overflow_rehome_at_source" if pkt.node == pkt.src else "overflow_rehome_at_relay"] += 1
            if tr.end is not EndType.CONTINUE:
                break
        ep = tot - before  # this episode's counts
        b, cyc = max(ep["born"], 1), max(ep["cycles"], 1)
        row = {"seed": seed, **{k: ep[k] / b for k in ep if k not in ("born", "cycles")
                                and not k.startswith("queued_")}}
        row["overflow_rehome"] = (ep["overflow_rehome_at_source"] + ep["overflow_rehome_at_relay"]) / b
        row["queued_at_source_mean"] = ep["queued_at_source_cycles"] / cyc
        row["queued_at_relay_mean"] = ep["queued_at_relay_cycles"] / cyc
        rows.append(row)
    tot["overflow_rehome"] = tot["overflow_rehome_at_source"] + tot["overflow_rehome_at_relay"]
    born, cycles = tot.pop("born"), max(tot.pop("cycles"), 1)
    occ = {"queued_at_source_mean": round(tot.pop("queued_at_source_cycles") / cycles, 2),
           "queued_at_relay_mean": round(tot.pop("queued_at_relay_cycles") / cycles, 2)}
    out = {"family": a.family, "policy": a.policy, "episodes": a.episodes, "sets": a.set, "born": born,
           "share_of_born_pct": {k: round(100 * v / born, 2) for k, v in sorted(tot.items())},
           "queued_packets_per_cycle": occ, "rows": rows}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
