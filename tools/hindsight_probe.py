"""X2 hindsight mask probe (stage 1a, docs/next-method.md section 7): how much delivery would a
scheduler gain if it never spent airtime on packets that will be lost anyway?  A VALUE probe with
hindsight (not deployable, not an upper bound).

Per scene (births are pre-drawn and packet ids follow birth order, so ids match across runs):
iteration 0 runs the base policy; the mask M collects the packets that were transmitted at least
once and still lost.  Iteration k runs again with M under one mode, adds the packets newly
transmitted-and-lost, and repeats (``--iterations``, union of masks).  Modes
(``LightweightBackend.set_hindsight_mask``): ``a`` drop at birth; ``b`` keep at the source, unsent
and hidden from the report; ``r1`` never forwarded from a relay; ``r2`` r1 and hidden at relays; ``rd`` dropped on reaching a relay.

Reported per iteration: delivery over all born packets and over the steady-state window, the mask
size, the wasted-transmission share, and the conversion g = extra delivered packets per masked
packet.

    .venv/bin/python tools/hindsight_probe.py load_high --mode r2 --deadline 1 --episodes 32 --seed-offset 1040
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool

from fanet_next.experiment.assemble import build_env, build_policy
from fanet_next.experiment.evaluate import split_seeds
from fanet_next.loop import EndType
from loss_ledger import FAMILIES, setup
from packet_tracker import PacketTracker


def run(env, policy, seed, idx, deadline, mask, mode):
    env.reset(seed, episode=idx)
    env.backend.set_hindsight_mask(mask, mode if mask else None)
    tk = PacketTracker(deadline)
    while True:
        tr = tk.step(env, policy.act([env.current], mode="greedy")[0].actions)
        if tr.end is not EndType.CONTINUE:
            break
    lost_after_tx = {pid for pid, lg in tk.log.items() if lg.tx > 0 and lg.fate not in ("delivered", None)}
    return tk, lost_after_tx


def scene(args) -> dict:
    a, idx, seed = args
    cfg, sc, window = setup(a.family, a.deadline, a.rules, a.set, a.horizon)
    policy = build_policy(cfg, a.policy, seed=0)
    env = build_env(cfg, run_id=f"x2_{idx}", build_graph=False, scenario_override=sc)
    mask: set[int] = set()
    iters = []
    for it in range(a.iterations + 1):
        tk, lost = run(env, policy, seed, idx, a.deadline, mask, a.mode)
        s_all, s_st = tk.summary(), tk.summary(window)
        iters.append({"iteration": it, "mask_size": len(mask), "delivered_all": s_all["delivered"],
                      "delivered_steady": s_st["delivered"], "born_all": s_all["born"], "born_steady": s_st["born"],
                      "tx_wasted_share": s_all["tx_wasted_share"], "fates_steady": s_st["fates"]})
        mask |= lost
    base = iters[0]
    for r in iters[1:]:
        extra = (r["delivered_all"] - base["delivered_all"]) * base["born_all"]
        r["conversion_g"] = extra / r["mask_size"] if r["mask_size"] else None
    return {"seed": seed, "iterations": iters}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=list(FAMILIES))
    ap.add_argument("--mode", required=True, choices=["a", "b", "r1", "r2", "rd"])
    ap.add_argument("--policy", default="lq_lasthop")
    ap.add_argument("--deadline", type=float, default=1.0)
    ap.add_argument("--rules", default="R5", choices=["R0", "R5"])
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--seed-offset", type=int, default=1040)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    jobs = [(a, i, s) for i, s in enumerate(split_seeds("dev", a.episodes, a.seed_offset))]
    with Pool(min(a.workers, len(jobs))) as pool:
        rows = pool.map(scene, jobs)
    print(json.dumps({"family": a.family, "mode": a.mode, "policy": a.policy, "deadline": a.deadline, "rules": a.rules,
                      "iterations": a.iterations, "episodes": a.episodes, "seed_offset": a.seed_offset,
                      "horizon": a.horizon, "sets": a.set, "rows": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
