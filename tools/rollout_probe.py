"""Rollout probe (no method change): does one-step lookahead with the simulator improve on a rule
base policy under a packet deadline?  It is a non-deployable policy that shows an improvement is
reachable (a lower bound of the available improvement, never an upper bound: few candidates and a
finite look-ahead can miss better plans), and the teacher a search-distilled method would imitate.

Every cycle of an episode (traffic and drain):

1. candidate plans at the current state: the plans of the base policy and of a few other rules
   (backpressure, longest queue, oldest head of line), and "wait" variants of the base plan with
   its last-chosen link dropped; duplicates removed;
2. each candidate is executed on a copy of the state and the base policy continues for
   ``horizon - 1`` cycles; ``reps`` repetitions per candidate, every candidate sees the same
   channel randomness in the same repetition (common random numbers);
3. score = deliveries in the window - eps * total delay of those deliveries (delivery first, delay
   second); the best mean score is executed in the real episode.

With a deadline D and horizon >= D / cycle, every packet present at the decision has met its fate
inside the window, so the score covers the consequences for them; later births are the same in
every branch.  The base policy alone runs on the same scenes for a paired comparison.

    .venv/bin/python tools/rollout_probe.py load_high --deadline 1.0 --episodes 8 --workers 4
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
from multiprocessing import Pool

import numpy as np

from confirm import ROOT
from fanet_next.config import apply_override, load_config
from fanet_next.experiment.assemble import build_env, build_policy
from fanet_next.experiment.evaluate import eval_scenario, split_seeds
from fanet_next.loop import EndType

FAMILIES = {"default": {}, "load_high": {"flow_rate_pps": [30.0, 45.0]},
            "nodes24": {"num_nodes": 24, "area_m": 367.4}}
R5 = ['backend.service_order="fewest_hops"', "candidates.require_room=true", "backend.room_aware_service=true"]
OTHERS = ("backpressure", "longest_queue", "oldest_hol")


def _cfg(a):
    cfg = load_config(str(ROOT / "configs/protocol_final.toml"))
    for item in (R5 if a.rules == "R5" else []) + [f"backend.deadline_s={a.deadline}"] + a.set:
        apply_override(cfg, item)
    return cfg


def _plans(env, base, others) -> dict[str, tuple]:
    inp = env.current
    out = {"base": tuple(base.act([inp], mode="greedy")[0].actions)}
    for name, pol in others.items():
        out[name] = tuple(pol.act([inp], mode="greedy")[0].actions)
    b = out["base"]
    if len(b) >= 1:
        out["wait_last"] = b[:-1]
    if len(b) >= 2:
        out["wait_last2"] = b[:-2]
    seen, uniq = set(), {}
    for k, p in out.items():
        if p not in seen:
            seen.add(p)
            uniq[k] = p
    return uniq


def _score(env, plan, base, horizon: int, eps: float) -> float:
    delivered, delay = 0, 0.0
    tr = env.step(list(plan))
    for i in range(horizon):
        delivered += len(tr.facts.delivered_ids)
        delay += float(sum(tr.facts.delivered_delays))
        if tr.end is not EndType.CONTINUE or i == horizon - 1:
            break
        tr = env.step(base.act([env.current], mode="greedy")[0].actions)
    return delivered - eps * delay


def episode(args) -> dict:
    a, idx, seed = args
    cfg = _cfg(a)
    sc = eval_scenario(cfg, a.drain, overrides=FAMILIES[a.family])
    base = build_policy(cfg, a.base, seed=0)
    others = {n: build_policy(cfg, n, seed=0) for n in OTHERS}
    out = {"seed": seed}
    for mode in ("base", "rollout"):
        env = build_env(cfg, run_id=f"rp{idx}", build_graph=False, scenario_override=sc)
        env.reset(seed, episode=idx)
        chosen = collections.Counter()
        while True:
            if mode == "base":
                plan = tuple(base.act([env.current], mode="greedy")[0].actions)
            else:
                cands = _plans(env, base, others)
                if len(cands) == 1:
                    plan = next(iter(cands.values()))
                    chosen["only"] += 1
                else:
                    scores = {k: 0.0 for k in cands}
                    for r in range(a.reps):
                        for k, p in cands.items():
                            clone = copy.deepcopy(env)
                            clone.backend.rng = np.random.default_rng([seed, int(env.backend.cycle), r])
                            scores[k] += _score(clone, p, base, a.horizon, a.eps)
                    best = max(scores, key=lambda k: (scores[k], k == "base"))
                    plan = cands[best]
                    chosen[best] += 1
            tr = env.step(list(plan))
            if tr.end is not EndType.CONTINUE:
                break
        s = env.metrics.summary()
        out[mode] = {k: s[k] for k in ("delivery_ratio", "e2e_delay_mean_s", "e2e_delay_p95_s", "born")}
        if mode == "rollout":
            out["chosen"] = dict(chosen)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=list(FAMILIES))
    ap.add_argument("--base", default="lq_lasthop")
    ap.add_argument("--rules", default="R5", choices=["R0", "R5"])
    ap.add_argument("--deadline", type=float, default=1.0)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed-offset", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=0, help="look-ahead cycles (0: deadline / cycle + 1)")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--eps", type=float, default=1e-3, help="weight of delay (s) against one delivery")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    cycle = float(load_config(str(ROOT / "configs/protocol_final.toml"))["scenario"]["cycle_length"])
    a.horizon = a.horizon or int(np.ceil(a.deadline / cycle)) + 1
    a.drain = int(np.ceil(a.deadline / cycle)) + 5
    jobs = [(a, i, s) for i, s in enumerate(split_seeds("dev", a.episodes, a.seed_offset))]
    with Pool(min(a.workers, len(jobs))) as pool:
        rows = pool.map(episode, jobs)
    d = np.array([r["rollout"]["delivery_ratio"] - r["base"]["delivery_ratio"] for r in rows])
    t = np.array([r["rollout"]["e2e_delay_mean_s"] - r["base"]["e2e_delay_mean_s"] for r in rows])
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    out = {"family": a.family, "base": a.base, "rules": a.rules, "deadline": a.deadline, "horizon": a.horizon,
           "reps": a.reps, "eps": a.eps, "episodes": a.episodes, "seed_offset": a.seed_offset, "sets": a.set,
           "delivery_gain": {"mean": float(d.mean()), "se": float(se)}, "delay_change_s": float(t.mean()),
           "base_delivery": float(np.mean([r["base"]["delivery_ratio"] for r in rows])),
           "chosen": dict(sum((collections.Counter(r["chosen"]) for r in rows), collections.Counter())),
           "rows": rows}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
