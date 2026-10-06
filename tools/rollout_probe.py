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
   second); the best mean score is executed in the real episode.  ``--score window`` (S1b) counts
   every delivery inside a window of ``horizon`` cycles; it leaves the backlog at the window end
   free, which favours holding back (S1b found this bias).  ``--score cohort`` (S1c) counts only
   the packets in the system at the decision plus those born in the next ``horizon`` cycles, over
   ``2 * horizon`` cycles, so every counted packet has met its fate (deadline) inside the window.

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


def _score(env, plan, base, horizon: int, eps: float, mode: str = "window") -> float:
    be = env.backend
    if mode == "cohort":
        cohort = {p.pid for q in be.queues for p in q} | {p.pid for w in be.waiting for p in w}
        first_new, steps = be.next_pid, 2 * horizon
    else:
        steps = horizon
    delivered, delay = 0, 0.0
    tr = env.step(list(plan))
    for i in range(steps):
        if mode == "cohort":
            if i == horizon:
                last_new = be.next_pid  # births of the first horizon cycles belong to the cohort
            for pid, dl in zip(tr.facts.delivered_ids, tr.facts.delivered_delays):
                if pid in cohort or (pid >= first_new and (i < horizon or pid < last_new)):
                    delivered += 1
                    delay += float(dl)
        else:
            delivered += len(tr.facts.delivered_ids)
            delay += float(sum(tr.facts.delivered_delays))
        if tr.end is not EndType.CONTINUE or i == steps - 1:
            break
        tr = env.step(base.act([env.current], mode="greedy")[0].actions)
    return delivered - eps * delay


def _link_features(env, link, rank: int, plan_size: int, deadline: float) -> dict:
    """Observable features of a planned link (S1d): its queue's packets in service order are not
    recomputed; all packets of the queue are summarised."""
    be, sc = env.backend, env.backend.sc
    tx, rx = int(link[0]), int(link[1])
    now = be.cycle * sc.cycle_length
    q = be.queues[be.qindex[(tx, rx)]]
    rem = [1 if p.dst == rx else 1 + max(be._route_len(rx, p.dst), 0) for p in q]
    age = [now - p.born for p in q]
    slack = [deadline - a for a in age]
    rx_q = sum(len(be.queues[i]) for i in be.node_queues[rx])
    pos = sc.positions(be.cycle)
    nh = [int(be.next_hop[rx, p.dst]) for p in q if p.dst != rx]
    room = [int(be.qcap[be.qindex[(rx, h)]]) - len(be.queues[be.qindex[(rx, h)]]) for h in set(nh) if h >= 0]
    n = max(len(q), 1)
    return {"rank": rank, "plan_size": plan_size, "n_pk": len(q),
            "share_last_hop": sum(r == 1 for r in rem) / n, "rem_hops_mean": float(np.mean(rem)) if rem else 0.0,
            "rem_hops_max": max(rem, default=0), "age_mean": float(np.mean(age)) if age else 0.0,
            "age_max": max(age, default=0.0), "slack_min": min(slack, default=deadline),
            "share_hopeless": sum(sl < r * sc.cycle_length for sl, r in zip(slack, rem)) / n,
            "share_tight": sum(sl < r * 0.1 for sl, r in zip(slack, rem)) / n,
            "share_at_source": sum(p.src == tx for p in q) / n, "rx_queued": rx_q,
            "rx_room_min": min(room, default=64), "dist_m": float(np.linalg.norm(pos[tx] - pos[rx])),
            "net_queued": sum(len(x) for x in be.queues), "cycle": int(be.cycle)}


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
        decisions: list[dict] = []
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
                            scores[k] += _score(clone, p, base, a.horizon, a.eps, a.score)
                    best = max(scores, key=lambda k: (scores[k], k == "base"))
                    plan = cands[best]
                    chosen[best] += 1
                    if a.log_decisions:
                        by_plan = {cands[k]: scores[k] for k in cands}
                        b, inp = cands["base"], env.current
                        links = [tuple(int(x) for x in inp.problem.links[i]) for i in b]
                        row = {"best": best, "plan_size": len(b),
                               "gain1": by_plan.get(b[:-1], float("nan")) - by_plan[b] if len(b) >= 1 else float("nan"),
                               "gain2": by_plan.get(b[:-2], float("nan")) - by_plan[b] if len(b) >= 2 else float("nan")}
                        if links:
                            row["last"] = _link_features(env, links[-1], 1, len(b), a.deadline)
                        if len(links) >= 2:
                            row["second"] = _link_features(env, links[-2], 2, len(b), a.deadline)
                        decisions.append(row)
            tr = env.step(list(plan))
            if tr.end is not EndType.CONTINUE:
                break
        s = env.metrics.summary()
        out[mode] = {k: s[k] for k in ("delivery_ratio", "e2e_delay_mean_s", "e2e_delay_p95_s", "born")}
        if mode == "rollout":
            out["chosen"] = dict(chosen)
            if a.log_decisions:
                out["decisions"] = decisions
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
    ap.add_argument("--score", default="window", choices=["window", "cohort"])
    ap.add_argument("--log-decisions", action="store_true", help="S1d: features of the base plan's last links")
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
           "score": a.score,
           "reps": a.reps, "eps": a.eps, "episodes": a.episodes, "seed_offset": a.seed_offset, "sets": a.set,
           "delivery_gain": {"mean": float(d.mean()), "se": float(se)}, "delay_change_s": float(t.mean()),
           "base_delivery": float(np.mean([r["base"]["delivery_ratio"] for r in rows])),
           "chosen": dict(sum((collections.Counter(r["chosen"]) for r in rows), collections.Counter())),
           "rows": rows}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
