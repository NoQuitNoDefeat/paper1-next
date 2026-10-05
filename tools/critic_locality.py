"""Diagnostic D5 (no method change): where in the network do the consequences of a scheduling
decision land, and how much noise comes with each part?

Same states and plans as the D3 design (``critic_consequence.reach_state`` / ``candidate_plans``,
other dev scenes), but every branch records the discounted return split over nodes
(``reward.by_node``: each queue's service / queue / delay term at its transmitter, each dropped
packet at the node where it was dropped; the parts sum exactly to the reward).

For a pair of plans at a state, the "decision footprint" is the set of endpoints of the links in
which the two plans differ.  Nodes are grouped

* by hop distance from the footprint on the communication graph at decision time
  (ring 0 = footprint, 1, 2, 3, and "far" = 4+ hops or unreachable), and
* by route: footprint / downstream (on the current routes of the packets queued on the differing
  links, from their receivers to the destinations) / elsewhere.

Per pair and repetition, X_g = sum over the group's nodes of the paired return difference.  Pooled
over pairs (and bootstrapped over states):

* signal share of group g: sum(mean_g * mean_tot - cov_r(X_g, X_tot) / R) / sum(mean_tot^2 - var_r(X_tot) / R);
  shares add up to one;
* noise of an estimate restricted to a footprint F (cumulative rings, or footprint + downstream):
  the variance over repetitions of X_F (paired) and of a single branch's return over F (unpaired;
  what a single-sample advantage would carry, apart from the critic);
* the gain in signal-to-noise of the restricted estimate over the global one:
  share_F^2 * var_tot / var_F (> 1: the restricted estimate is more informative per sample, at the
  price of a bias equal to the signal outside F).

Limitation (as D3): the repetitions share the scene's arrivals and mobility, so the noise is
smaller than in training; the comparison between groups is what this reports.

    .venv/bin/python tools/critic_locality.py results/e6/no_set_summary-s0 --workers 3
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import deque
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from confirm import ROOT
from critic_consequence import _setup, candidate_plans, reach_state
from fanet_next.loop import EndType

DEV_BASE = 10_000_000 + 700  # dev scenes beyond selection, ablation, D2 (500-539) and D3 (600-647)
RINGS = ("ring0", "ring1", "ring2", "ring3", "far")
ROUTE = ("footprint", "downstream", "elsewhere")


def hop_matrix(report) -> np.ndarray:
    """Hop distances on the (undirected) communication graph at decision time; -1 = unreachable."""
    snr = report.tx_power[:, None] * report.gain / report.noise
    adj = snr >= report.threshold * (1 - 1e-9)
    np.fill_diagonal(adj, False)
    adj = adj | adj.T
    n = len(adj)
    dist = -np.ones((n, n), dtype=np.int64)
    for s in range(n):
        dist[s, s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in np.nonzero(adj[u])[0]:
                if dist[s, v] < 0:
                    dist[s, v] = dist[s, u] + 1
                    q.append(v)
    return dist


def downstream_nodes(report, link) -> list[int]:
    """Nodes on the current routes of the packets queued on ``link`` = (u, v), from v onwards."""
    u, v = link
    q = np.nonzero((report.queue_links == [u, v]).all(axis=1))[0]
    if len(q) == 0 or report.queue_dst is None:
        return [v]
    out = {v}
    for d in np.nonzero(report.queue_dst[q[0]])[0]:
        x, steps = v, 0
        while x != d and steps < report.num_nodes:
            nh = int(report.next_hop[x, d])
            if nh < 0:
                break
            out.add(nh)
            x, steps = nh, steps + 1
    return sorted(int(x) for x in out)


def _rollout_nodes(env, policy, plan, seed: int, scale: float, gamma: float):
    policy.generator.manual_seed(seed)
    n = env.backend.sc.num_nodes
    qlinks = env.backend.qlinks
    g_nodes, total, disc = np.zeros(n), 0.0, 1.0
    actions = plan
    while True:
        if actions is None:
            actions = policy.act([env.current], mode="sample")[0].actions
        tr = env.step(list(actions))
        g_nodes += disc * env.reward.by_node(tr.facts, qlinks, n) / scale
        total += disc * float(np.clip(tr.reward.total / scale, -10.0, 10.0))
        disc *= gamma
        if tr.end is not EndType.CONTINUE:
            return g_nodes.tolist(), total
        actions = None


def probe_state(args) -> dict | None:
    run, idx, k_samples, m_policy, reps = args
    cfg, policy, scale, gamma = _setup(run)
    reached = reach_state(cfg, policy, idx, base=DEV_BASE, prefix="d5")
    if reached is None:
        return None
    env, rng, t0 = reached
    plans, v0, n_sampled = candidate_plans(env, policy, cfg, idx, rng, k_samples, m_policy)
    rep = env.current.report
    cand = env.current.problem.links
    res = {"index": idx, "t0": t0, "num_nodes": int(rep.num_nodes), "hops": hop_matrix(rep).tolist(),
           "plans": []}
    downstream = {}
    for key, p in plans.items():
        links = [tuple(int(x) for x in cand[a]) for a in p["order"]]
        for link in links:
            downstream.setdefault(link, downstream_nodes(rep, link))
        runs = [_rollout_nodes(copy.deepcopy(env), policy, p["order"], 50_000 + r, scale, gamma) for r in range(reps)]
        res["plans"].append({"source": p["source"], "links": [list(x) for x in links],
                             "g_nodes": [x[0] for x in runs], "g": [x[1] for x in runs]})
    res["downstream"] = [[list(k), v] for k, v in downstream.items()]
    return res


# ------------------------------------------------------------------ analysis
def pair_stats(states: list[dict]) -> list[list[dict]]:
    """Per state: per pair of plans, group sums of the paired node differences per repetition."""
    blocks = []
    for s in states:
        plans = s["plans"]
        if len(plans) < 2:
            continue
        hops = np.array(s["hops"])
        down = {tuple(k): set(v) for k, v in s["downstream"]}
        n = s["num_nodes"]
        rows = []
        for i in range(len(plans)):
            for j in range(i + 1, len(plans)):
                a, b = plans[i], plans[j]
                diff = set(map(tuple, a["links"])) ^ set(map(tuple, b["links"]))
                if not diff:
                    continue
                foot = sorted({x for link in diff for x in link})
                d = hops[foot]  # (|F|, n)
                ring = np.where((d >= 0).any(0), np.where(d >= 0, d, 10 ** 6).min(0), -1)
                ring_group = np.where(ring < 0, 4, np.minimum(ring, 4))
                dn = set().union(*(down.get(link, set()) for link in diff)) - set(foot)
                route_group = np.array([0 if x in foot else (1 if x in dn else 2) for x in range(n)])
                combo_group = np.where((ring_group <= 1) | (route_group <= 1), 0, 1)
                x = np.array(a["g_nodes"]) - np.array(b["g_nodes"])  # (R, n)
                ga, gb = np.array(a["g_nodes"]), np.array(b["g_nodes"])
                rows.append({
                    "rings": np.stack([x[:, ring_group == g].sum(1) for g in range(5)], 1),  # (R, 5)
                    "route": np.stack([x[:, route_group == g].sum(1) for g in range(3)], 1),  # (R, 3)
                    "combo": np.stack([x[:, combo_group == g].sum(1) for g in range(2)], 1),  # (R, 2)
                    "single_rings": [np.stack([y[:, ring_group == g].sum(1) for g in range(5)], 1) for y in (ga, gb)],
                    "single_route": [np.stack([y[:, route_group == g].sum(1) for g in range(3)], 1) for y in (ga, gb)],
                    "single_combo": [np.stack([y[:, combo_group == g].sum(1) for g in range(2)], 1) for y in (ga, gb)],
                })
        if rows:
            blocks.append(rows)
    return blocks


FOOTPRINTS = {  # name -> (grouping, groups included)
    "ring0": ("rings", [0]), "ring<=1": ("rings", [0, 1]), "ring<=2": ("rings", [0, 1, 2]),
    "ring<=3": ("rings", [0, 1, 2, 3]), "footprint+downstream": ("route", [0, 1]),
    "ring<=1+downstream": ("combo", [0]),
}


def _signal(rows, grouping, groups) -> tuple[float, float, float, float, float, float]:
    """Pooled: captured signal sum(mean_F*mean_tot - cov/R), total signal sum(mean_tot^2 - var/R),
    paired noise sum(var_F), paired total noise sum(var_tot), unpaired noise of F and total."""
    cap = tot = vf = vt = uf = ut = 0.0
    for r in rows:
        x = r[grouping]
        xf, xt = x[:, groups].sum(1), x.sum(1)
        k = len(xt)
        cap += xf.mean() * xt.mean() - np.cov(xf, xt)[0, 1] / k
        tot += xt.mean() ** 2 - xt.var(ddof=1) / k
        vf += xf.var(ddof=1)
        vt += xt.var(ddof=1)
        for y in r["single_" + grouping]:
            uf += y[:, groups].sum(1).var(ddof=1) / 2
            ut += y.sum(1).var(ddof=1) / 2
    return cap, tot, vf, vt, uf, ut


def footprint_table(blocks) -> dict:
    rows = [r for b in blocks for r in b]
    out = {}
    for name, spec in FOOTPRINTS.items():
        cap, tot, vf, vt, uf, ut = _signal(rows, *spec)
        share = cap / tot if tot > 0 else float("nan")
        out[name] = {"signal_share": share, "paired_noise_ratio": vf / vt, "single_noise_ratio": uf / ut,
                     "snr_gain_paired": share ** 2 * vt / vf if vf > 0 else float("nan"),
                     "snr_gain_single": share ** 2 * ut / uf if uf > 0 else float("nan")}
    return out


def group_shares(blocks, grouping, names) -> dict:
    rows = [r for b in blocks for r in b]
    tot = sum(r[grouping].sum(1).mean() ** 2 - r[grouping].sum(1).var(ddof=1) / len(r[grouping]) for r in rows)
    out = {}
    for g, name in enumerate(names):
        cap = 0.0
        for r in rows:
            x = r[grouping]
            xf, xt = x[:, g], x.sum(1)
            cap += xf.mean() * xt.mean() - np.cov(xf, xt)[0, 1] / len(xt)
        out[name] = cap / tot if tot > 0 else float("nan")
    return out


def group_magnitudes(blocks, grouping, names) -> dict:
    """Noise-corrected mean square of each group's own difference, relative to the total's:
    large values with small shares mean effects that cancel (e.g. better here, worse downstream)."""
    rows = [r for b in blocks for r in b]
    tot = sum(r[grouping].sum(1).mean() ** 2 - r[grouping].sum(1).var(ddof=1) / len(r[grouping]) for r in rows)
    return {name: sum(r[grouping][:, g].mean() ** 2 - r[grouping][:, g].var(ddof=1) / len(r[grouping])
                      for r in rows) / tot if tot > 0 else float("nan") for g, name in enumerate(names)}


def bootstrap(blocks, fn, n=500, seed=0) -> list[float]:
    rng = np.random.default_rng(seed)
    vals = [fn([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]) for _ in range(n)]
    vals = [v for v in vals if np.isfinite(v)]
    return [float(x) for x in np.percentile(vals, [2.5, 97.5])] if vals else [float("nan")] * 2


def summarise(states: list[dict]) -> dict:
    blocks = pair_stats(states)
    out = {"states": len(states), "states_with_pairs": len(blocks), "pairs": sum(len(b) for b in blocks),
           "ring_shares": group_shares(blocks, "rings", RINGS),
           "route_shares": group_shares(blocks, "route", ROUTE),
           "ring_magnitudes": group_magnitudes(blocks, "rings", RINGS),
           "route_magnitudes": group_magnitudes(blocks, "route", ROUTE),
           "footprints": footprint_table(blocks)}
    for name in RINGS:
        out["ring_shares"][name + "_ci"] = bootstrap(blocks, lambda b: group_shares(b, "rings", RINGS)[name])
    for name in ROUTE:
        out["route_shares"][name + "_ci"] = bootstrap(blocks, lambda b: group_shares(b, "route", ROUTE)[name])
    for name in out["footprints"]:
        out["footprints"][name]["snr_gain_single_ci"] = bootstrap(
            blocks, lambda b: footprint_table(b)[name]["snr_gain_single"])
        out["footprints"][name]["signal_share_ci"] = bootstrap(
            blocks, lambda b: footprint_table(b)[name]["signal_share"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="one or more runs; their states are pooled in the summary")
    ap.add_argument("--states", type=int, default=48)
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--policy-plans", type=int, default=4)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--summary-only", action="store_true", help="pool existing outputs of the runs")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    all_states = []
    for run in a.runs:
        path = ROOT / "results/diagnostics" / f"d5_{Path(run).name}.json"
        if a.summary_only:
            all_states += json.loads(path.read_text())["states"]
            continue
        jobs = [(run, i, a.samples, a.policy_plans, a.reps) for i in range(a.states)]
        with Pool(a.workers) as pool:
            states = [s for s in pool.map(probe_state, jobs, chunksize=1) if s is not None]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"run": run, "summary": summarise(states), "states": states}))
        all_states += states
    summary = summarise(all_states)
    out = Path(a.out) if a.out else ROOT / "results/diagnostics" / "d5_summary.json"
    out.write_text(json.dumps({"runs": a.runs, "summary": summary}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
