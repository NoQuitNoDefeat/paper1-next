"""Diagnostic (no method change): does the critic miss the content of the plan being formed?

With the final method's ``set_summary = none`` the critic sees, at every micro state,
the cycle encoding, the mean embedding of the still-feasible candidates and four
counts (micro features) - but not *which* links were chosen.  Two partial plans
that leave a similar remaining set look alike to it even when one serves the long
queues and the other the short ones.

This probe restores a checkpoint (model, return scaler, training seed stream) into a
scratch run, collects training-style rollouts, and asks how much of the critic's
boundary TD error (where the cycle reward enters: delta = r + gamma V(next) - V(s_k),
s_k = complete plan) a ridge regression can explain out of sample from

* state features: sums / means of the candidate features over all candidates, and
* plan features: the same over the chosen links, and chosen / all ratios.

If adding the plan features raises the explained share clearly, the critic lacks
plan content.  Also reported: within-cycle explained variance of V (how well V
tracks the change of the lambda-return from micro state to micro state).

    .venv/bin/python tools/critic_probe.py results/e6/no_set_summary-s0 results/e5/A-lv20-s0
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from confirm import ROOT
from fanet_next.experiment.train import TrainingRun
from fanet_next.training.advantages import flatten_stream
from fanet_next.training.checkpoint import load_checkpoint, set_rng_state


def restore(run_dir: Path) -> TrainingRun:
    sel = json.loads((run_dir / "selection.json").read_text())["selected"]["checkpoint"]
    data = load_checkpoint(ROOT / sel)
    run = TrainingRun(data["config"], Path(tempfile.mkdtemp()))  # never writes into run_dir
    run.model.load_state_dict(data["model"])
    run.collector.load_state_dict(data["collector"], restore_envs=False)
    run.envs = run.collector.envs
    set_rng_state(data["rng"], run.policy.generator)
    return run


F = {n: i for i, n in enumerate(("queue", "hol", "route_next", "snr_margin", "distance", "distance_rate",
                                   "age", "survival", "quality_trend", "is_candidate", "reverse_queue"))}


def features(rec) -> tuple[np.ndarray, np.ndarray]:
    """Robust state and plan summaries from the candidate features (observation EDGE_FEATURES)."""
    x = np.asarray(rec.micro.graph.cand_x, dtype=np.float64)
    chosen = np.asarray(rec.micro.actions, dtype=np.int64)
    c, k = len(x), len(chosen)
    col = lambda name, rows=None: (x[:, F[name]] if rows is None else x[rows, F[name]])
    if c == 0:
        return np.zeros(6), np.zeros(8)
    state = np.array([col("queue").sum(), col("hol").mean(), c, np.log1p(c), col("route_next").sum(),
                      col("snr_margin").mean()])
    if k == 0:
        return state, np.zeros(8)
    share = lambda name: col(name, chosen).sum() / max(col(name).sum(), 1e-6)  # non-negative features
    plan = np.array([k, k / c, share("queue"), share("hol"), share("route_next"), share("reverse_queue"),
                     col("snr_margin", chosen).mean(), col("distance", chosen).mean()])
    return state, plan


def ridge_r2(xtr, ytr, xte, yte, alpha=10.0) -> float:
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-9
    a, b = (xtr - mu) / sd, (xte - mu) / sd
    a, b = np.c_[a, np.ones(len(a))], np.c_[b, np.ones(len(b))]
    w = np.linalg.solve(a.T @ a + alpha * np.eye(a.shape[1]), a.T @ ytr)
    return float(1 - ((yte - b @ w) ** 2).mean() / yte.var())


def probe(run_dir: Path, collects: int) -> dict:
    """Per cycle: T = r + gamma V(next) (the boundary target), V0 = V at the cycle start (no link
    chosen yet), Vk = V at the complete plan.  The critic's own refinement is how much Vk improves
    on V0 as a prediction of T; the plan-feature probe is how much of T - V0 a ridge regression on
    plan (+ state) features explains out of sample (split by episode)."""
    run = restore(run_dir)
    gamma = run.tcfg["gamma"]
    rows, steps = [], []
    for _ in range(collects):
        res = run.collector.collect()
        for stream in res.streams:
            if not stream:
                continue
            r, v, nv, g, bt, _ = map(np.asarray, flatten_stream(stream, gamma))
            target = r + g * bt * nv
            pos = 0
            for rec in stream:
                k = rec.num_actions
                pos += k + 1
                if k == 0:
                    continue
                st, pl = features(rec)
                vals = rec.micro.values
                rows.append((rec.episode, st, pl, float(target[pos - 1]), float(vals[0]), float(vals[k])))
                steps.extend(np.diff(vals[:k + 1]))
    ep = np.array([x[0] for x in rows])
    st = np.stack([x[1] for x in rows])
    pl = np.stack([x[2] for x in rows])
    t, v0, vk = (np.array([x[i] for x in rows]) for i in (3, 4, 5))
    s0, sk = t - v0, t - vk
    test = (ep % 2) == 1
    both = np.c_[st, pl]
    return {"run": str(run_dir), "cycles": len(rows),
            "surprise_std_cycle_start": float(s0.std()), "surprise_std_complete_plan": float(sk.std()),
            "critic_refinement_r2": float(1 - sk.var() / s0.var()),
            "probe_r2_state": ridge_r2(st[~test], s0[~test], st[test], s0[test]),
            "probe_r2_state_plan": ridge_r2(both[~test], s0[~test], both[test], s0[test]),
            "corr_dV_queue_share": float(np.corrcoef(vk - v0, pl[:, 2])[0, 1]),
            "corr_surprise_queue_share": float(np.corrcoef(s0, pl[:, 2])[0, 1]),
            "micro_step_value_change_std": float(np.std(steps)),
            "boundary_td_std": float(sk.std())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--collects", type=int, default=4, help="rollouts of num_envs x rollout_cycles cycles")
    a = ap.parse_args()
    out = [probe(ROOT / r, a.collects) for r in a.runs]
    for o in out:
        print(f"{o['run']}: {o['cycles']} cycles; surprise std at cycle start {o['surprise_std_cycle_start']:.4f}, "
              f"at complete plan {o['surprise_std_complete_plan']:.4f} -> critic refinement R2 "
              f"{o['critic_refinement_r2']:+.3f}; probe R2 state {o['probe_r2_state']:+.3f}, state+plan "
              f"{o['probe_r2_state_plan']:+.3f}; corr(Vk-V0, queue share) {o['corr_dV_queue_share']:+.2f}, "
              f"corr(T-V0, queue share) {o['corr_surprise_queue_share']:+.2f}; within-cycle step dV std "
              f"{o['micro_step_value_change_std']:.4f} vs boundary TD std {o['boundary_td_std']:.4f}")
    path = ROOT / "results/diagnostics/critic_probe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
