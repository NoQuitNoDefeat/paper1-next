"""Per-cycle optimality gap and decision time of any policy.

On every decision state a policy visits, the exact per-cycle max-weight optimum
(MILP; weights = next-hop queue lengths, the classical max-weight objective) is
solved on the same scheduling problem, and the policy's plan is scored with the
same weights.  The ratio plan/optimum is the approximation ratio used by the
link-scheduling literature (e.g. Zhao et al., TWC 2023).  It measures one-cycle
max-weight quality only: a policy that trades it for delay or delivery over
time (such as the learned method) is not "wrong" when it is below 1.
"""

from __future__ import annotations

import time

import numpy as np

from ..policy.base import Policy
from ..policy.classical import queue_weights
from ..scheduling.maxweight import max_weight_set
from .assemble import build_env
from .evaluate import eval_scenario, split_seeds


def probe(cfg: dict, policy: Policy, *, split: str = "dev", episodes: int = 4,
          seed_offset: int = 0, cycles: int | None = None, drain_cycles: int = 0,
          time_limit: float = 10.0) -> dict:
    scenario = eval_scenario(cfg, drain_cycles)
    env = build_env(cfg, run_id="optgap", build_graph=policy.needs_graph,
                    scenario_override=scenario)
    if hasattr(policy, "seed") and (not policy.learnable or policy.stochastic_eval):
        policy.seed(12345)
    ratios, decide_ms, solve_ms, statuses = [], [], [], []
    for i, seed in enumerate(split_seeds(split, episodes, seed_offset)):
        inp = env.reset(seed, episode=i)
        k = 0
        while True:
            w = queue_weights(inp)
            t0 = time.perf_counter()
            out = policy.act([inp], mode="greedy")[0]
            decide_ms.append((time.perf_counter() - t0) * 1e3)
            if inp.problem.num_candidates:
                sel, info = max_weight_set(inp.problem, w, time_limit=time_limit)
                solve_ms.append(info.seconds * 1e3)
                statuses.append(info.status)
                if info.objective > 0:
                    ratios.append(float(w[out.actions].sum()) / info.objective)
            tr = env.step(out.actions)
            k += 1
            if tr.end.value != "continue" or (cycles is not None and k >= cycles):
                break
            inp = tr.next_input
    r = np.asarray(ratios)
    return {"cycles": len(decide_ms), "scored_cycles": len(r),
            "ratio_mean": float(r.mean()) if len(r) else float("nan"),
            "ratio_p5": float(np.percentile(r, 5)) if len(r) else float("nan"),
            "ratio_min": float(r.min()) if len(r) else float("nan"),
            "optimal_frac": float(np.mean(r >= 1 - 1e-9)) if len(r) else float("nan"),
            "decision_ms_mean": float(np.mean(decide_ms)),
            "decision_ms_p99": float(np.percentile(decide_ms, 99)),
            "milp_ms_mean": float(np.mean(solve_ms)) if solve_ms else 0.0,
            "milp_ms_p99": float(np.percentile(solve_ms, 99)) if solve_ms else 0.0,
            "milp_not_optimal": int(sum(s != "optimal" for s in statuses)),
            "split": split, "episodes": episodes, "seed_offset": seed_offset,
            "cycles_per_episode": cycles, "drain_cycles": drain_cycles}
