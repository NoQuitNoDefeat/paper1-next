"""S1e (stage 1a X1, docs/next-method.md section 7): a NON-clairvoyant one-step rollout over the
rule base heur@R5, and the anatomy of its waits.  It checks whether the S1c/S1d gain survives
when the planner does not know the future arrivals and only the steady state counts.

Each cycle in [W, traffic stop) (the base policy acts before W and after the stop):

1. candidates, all from fresh controllers: the base plan; "skip j" for every link j of the base
   plan (link j is barred, the greedy continues: the student's action); the base plan without its
   last one or two links (the S1c/S1d waits);
2. K futures: the arrivals from this cycle on are redrawn per flow from its Poisson rate
   (memoryless, so exact in distribution); every candidate sees the same K futures (common random
   numbers); plus the true future, used only for the clairvoyant-agreement statistic;
3. score (delivery only) = on-time deliveries of the packets in the system at the decision and of
   those born in the next Hc cycles, over 2.5H cycles (every counted packet meets its deadline
   inside); the choice uses Hc = H (H = D / cycle + 1): with ``--select margin`` a candidate
   replaces the base plan only when its mean gain over the K futures exceeds ``--margin`` standard
   errors (the best such candidate by mean); with ``--select crossfit`` the margin test and the pick
   use the first half of the futures and the pick is kept only if its mean gain on the second half
   is positive.  Hc = 1.5 H is scored too, and both windows are scored on the true future, which
   the choice never sees (independent window and clairvoyance checks).  The redrawn futures also
   reseed the channel generator, so no future fading leaks under a fading channel.

Per episode the base policy alone runs on the same scene; delivery is reported over all born
packets and over the steady-state window [W, stop - D/T) (tools/packet_tracker.py).  Every decision
with more than one candidate is logged (phase, chosen, per-candidate mean gain under both windows
and under the true future, link features for the "skip j" candidates), and the packets on links the
planner held back are followed until their next hop (postponement in cycles) or loss.

    .venv/bin/python tools/rollout_teacher.py load_high --episodes 12 --seed-offset 1020 --workers 4
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

from fanet_next.experiment.assemble import build_env, build_policy
from fanet_next.experiment.evaluate import split_seeds
from fanet_next.loop import EndType
from fanet_next.policy.classical import _greedy_allowed
from fanet_next.scenario.base import Birth
from loss_ledger import FAMILIES, setup
from packet_tracker import PacketTracker
from rollout_probe import _link_features


def resample_births(sc, start: int, rng: np.random.Generator) -> None:
    """Redraw the arrivals of cycles >= start (Poisson per flow, the episode's rate)."""
    T, stop = sc.cycle_length, sc.traffic_stop_cycle
    for k in range(start, sc.horizon):
        sc._births[k] = []
    if start >= stop or sc.flow_rate_pps <= 0:
        return
    t_end = stop * T
    for src, dst in sc.flows:
        t = start * T
        while True:
            t += rng.exponential(1.0 / sc.flow_rate_pps)
            if t > t_end:
                break
            k = min(max(math.ceil(t / T) - 1, start), sc.horizon - 1)
            sc._births[k].append(Birth(t, src, dst, sc.packet_size))
    for k in range(start, sc.horizon):
        sc._births[k].sort(key=lambda b: (b.time, b.src))


def _fresh(env):
    inp = env.current
    return replace(inp, controller=env.constraints.start(inp.problem))


def candidates(env, base) -> tuple[dict[str, tuple], dict[str, list[str]]]:
    """Unique candidate plans in the order base, drop_last, drop_last2, skip0.. (the first name of
    an identical plan is kept), and every name of each kept plan (aliases)."""
    score = base.scores(env.current)
    n = len(score)
    plan = tuple(_greedy_allowed(_fresh(env), score, np.ones(n, dtype=bool)))
    out = [("base", plan)]
    if len(plan) >= 1:
        out.append(("drop_last", plan[:-1]))
    if len(plan) >= 2:
        out.append(("drop_last2", plan[:-2]))
    for j, i in enumerate(plan):
        allowed = np.ones(n, dtype=bool)
        allowed[i] = False
        out.append((f"skip{j}", tuple(_greedy_allowed(_fresh(env), score, allowed))))
    first, uniq, aliases = {}, {}, {}
    for k, p in out:
        if p not in first:
            first[p] = k
            uniq[k] = p
        aliases.setdefault(first[p], []).append(k)
    return uniq, aliases


def moved_by(env, plan) -> set[int]:
    """Packets the plan moves one hop (or delivers) this cycle, from a copy of the state."""
    clone = copy.deepcopy(env)
    be = clone.backend
    where = {p.pid: (p, p.node) for q in be.queues for p in q}
    tr = clone.step(list(plan))
    done = set(tr.facts.delivered_ids)
    return {pid for pid, (p, node0) in where.items() if pid in done or p.node != node0}


def choose(gains: dict[str, np.ndarray], means: dict[str, float], margin: float, crossfit: bool) -> str:
    """The executed candidate.  ``gains[nm]``: per-future gain over the base plan.  margin: the mean
    gain must exceed margin standard errors; crossfit: pass the margin on the first half of the
    futures, pick the best of those by that half's mean, and keep it only if its mean gain on the
    second half is positive."""
    def passes(g):
        se = g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0.0
        return g.mean() > margin * se
    if not crossfit:
        ok = [nm for nm, g in gains.items() if passes(g)]
        return max(ok, key=lambda nm: means[nm]) if ok else "base"
    half = len(next(iter(gains.values()))) // 2
    ok = [nm for nm, g in gains.items() if passes(g[:half])]
    if not ok:
        return "base"
    best = max(ok, key=lambda nm: gains[nm][:half].mean())
    return best if gains[best][half:].mean() > 0 else "base"


def score(env, plan, base, H: int) -> tuple[int, int]:
    """On-time deliveries of the cohort with birth windows H and 1.5 H, over 2.5 H cycles."""
    be = env.backend
    cohort = {p.pid for q in be.queues for p in q} | {p.pid for w in be.waiting for p in w}
    first_new = be.next_pid
    h15, steps = int(1.5 * H), int(2.5 * H)
    last = {}
    d_h = d_h15 = 0
    tr = env.step(list(plan))
    for i in range(steps):
        for c in (H, h15):
            if i == c - 1:  # births of cycles t .. t + c - 1 are numbered by now
                last[c] = be.next_pid
        for pid in tr.facts.delivered_ids:
            if pid in cohort:
                d_h += 1
                d_h15 += 1
            elif pid >= first_new:
                d_h += int(i < H or pid < last.get(H, 1 << 60))
                d_h15 += int(i < h15 or pid < last.get(h15, 1 << 60))
        if tr.end is not EndType.CONTINUE or i == steps - 1:
            break
        tr = env.step(base.act([env.current], mode="greedy")[0].actions)
    return d_h, d_h15


def episode(args) -> dict:
    a, idx, seed = args
    cfg, sc, window = setup(a.family, a.deadline, "R5", a.set, None)
    base = build_policy(cfg, "lq_lasthop", seed=0)
    H = int(math.ceil(a.deadline / float(cfg["scenario"]["cycle_length"]))) + 1
    W, _ = window
    out = {"seed": seed, "window": window, "H": H}
    for mode in ("base", "rollout"):
        env = build_env(cfg, run_id=f"s1e{idx}", build_graph=False, scenario_override=sc)
        env.reset(seed, episode=idx)
        stop = env.backend.sc.traffic_stop_cycle
        tk = PacketTracker(a.deadline)
        decisions, watch, held = [], {}, []
        while True:
            k = env.backend.cycle
            if mode == "rollout" and W <= k < stop:
                cands, aliases = candidates(env, base)
                plan = cands["base"]
                if len(cands) > 1:
                    names = list(cands)
                    res = {nm: [] for nm in names}
                    true = {}
                    for j in range(a.futures + 1):  # the last one is the true future (agreement only)
                        for nm in names:
                            clone = copy.deepcopy(env)
                            if j < a.futures:  # same redrawn future for every candidate
                                clone.backend.rng = np.random.default_rng([seed, k, j, 1])
                                resample_births(clone.backend.sc, k, np.random.default_rng([seed, k, j]))
                                res[nm].append(score(clone, cands[nm], base, H))
                            else:
                                true[nm] = score(clone, cands[nm], base, H)
                    other = [nm for nm in names if nm != "base"]
                    g_h = {nm: np.array([r[0] - b[0] for r, b in zip(res[nm], res["base"])], dtype=float) for nm in other}
                    g_15 = {nm: np.array([r[1] - b[1] for r, b in zip(res[nm], res["base"])], dtype=float) for nm in other}
                    best = choose(g_h, {nm: float(g.mean()) for nm, g in g_h.items()}, a.margin, a.select == "crossfit")
                    best15 = choose(g_15, {nm: float(g.mean()) for nm, g in g_15.items()}, a.margin, a.select == "crossfit")
                    best_true = max(names, key=lambda nm: (true[nm][0], nm == "base"))
                    plan = cands[best]
                    links = [tuple(int(x) for x in env.current.problem.links[i]) for i in cands["base"]]
                    row = {"cycle": k, "plan_size": len(cands["base"]), "chosen": best, "aliases": aliases.get(best, [best]),
                           "chosen_15": best15, "chosen_true": best_true,
                           "gain_h": {nm: float(g_h[nm].mean()) for nm in other},
                           "gain_15": {nm: float(g_15[nm].mean()) for nm in other},
                           "gain_true": {nm: true[nm][0] - true["base"][0] for nm in other},
                           "gain_true_15": {nm: true[nm][1] - true["base"][1] for nm in other},
                           "gain_h_futures": {nm: g_h[nm].tolist() for nm in other}}
                    if a.features:
                        row["skip_features"] = {}
                        for j in range(len(links)):
                            if f"skip{j}" in cands:
                                ft = _link_features(env, links[j], len(links) - j, len(links), a.deadline)
                                ft["rank_from_first"] = j + 1  # rank (as in S1d) counts from the last link
                                row["skip_features"][f"skip{j}"] = ft
                    decisions.append(row)
                    if best != "base":  # follow the packets the base plan would have moved
                        be = env.backend
                        node_of = {p.pid: p.node for q in be.queues for p in q}
                        for pid in moved_by(env, cands["base"]) - moved_by(env, plan):
                            watch.setdefault(pid, (k, node_of[pid]))
            else:
                plan = tuple(base.act([env.current], mode="greedy")[0].actions)
            tr = tk.step(env, list(plan))
            if watch:
                be = env.backend
                f = tr.facts
                nodes = {p.pid: p.node for q in be.queues for p in q}
                nodes.update({p.pid: p.node for w in be.waiting for p in w})
                lost = {pid for ids in f.terminations.values() for pid in ids}
                done = set(f.delivered_ids)
                for pid in list(watch):
                    k0, node0 = watch[pid]
                    if pid in done or (pid in nodes and nodes[pid] != node0):
                        held.append({"waited": env.backend.cycle - 1 - k0, "fate": "moved"})
                        del watch[pid]
                    elif pid in lost:
                        held.append({"waited": env.backend.cycle - 1 - k0, "fate": "lost"})
                        del watch[pid]
            if tr.end is not EndType.CONTINUE:
                break
        out[mode] = {"all": tk.summary(), "steady": tk.summary(window)}
        if mode == "rollout":
            out["decisions"], out["held"] = decisions, held
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=list(FAMILIES))
    ap.add_argument("--deadline", type=float, default=1.0)
    ap.add_argument("--futures", type=int, default=4)
    ap.add_argument("--margin", type=float, default=0.0, help="standard errors a candidate must beat the base by")
    ap.add_argument("--select", default="margin", choices=["margin", "crossfit"])
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--seed-offset", type=int, default=1020)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-features", dest="features", action="store_false")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    jobs = [(a, i, s) for i, s in enumerate(split_seeds("dev", a.episodes, a.seed_offset))]
    with Pool(min(a.workers, len(jobs))) as pool:
        rows = pool.map(episode, jobs)
    print(json.dumps({"family": a.family, "deadline": a.deadline, "futures": a.futures, "margin": a.margin,
                      "select": a.select,
                      "episodes": a.episodes,
                      "seed_offset": a.seed_offset, "sets": a.set, "rows": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
