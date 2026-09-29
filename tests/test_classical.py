"""Classical baselines: exact max-weight, local search, backpressure weights, spatial TDMA."""

from itertools import combinations

import numpy as np
import pytest

from fanet_next.config import load_config
from fanet_next.experiment.assemble import build_env, build_policy
from fanet_next.experiment.evaluate import eval_scenario
from fanet_next.policy.base import DecisionInput
from fanet_next.policy.classical import backpressure_weights
from fanet_next.scheduling.maxweight import (feasible, greedy_complete, local_search,
                                             max_weight_set)
from fanet_next.scheduling.problem import SchedulingProblem

from helpers import fixed_env, line_positions


def _random_problem(rng, n=8, c=11):
    pos = rng.uniform(0, 300, size=(n, 3))
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1) + np.eye(n)
    gain = 1e-4 * d ** -2.5
    np.fill_diagonal(gain, 0.0)
    pairs = [(s, r) for s in range(n) for r in range(n) if s != r]
    snr = 0.1 * gain / 1e-11
    ok = [p for p in pairs if snr[p] >= 3.0]
    links = np.array([ok[i] for i in rng.choice(len(ok), size=min(c, len(ok)), replace=False)])
    return SchedulingProblem(links=links, gain=gain, power=np.full(n, 0.1), noise=1e-11,
                             threshold=3.0)


def _brute_force(problem, w):
    best, arg = 0.0, []
    for k in range(1, problem.num_candidates + 1):
        for s in combinations(range(problem.num_candidates), k):
            if w[list(s)].sum() > best + 1e-12 and feasible(problem, s):
                best, arg = float(w[list(s)].sum()), list(s)
    return best, arg


@pytest.mark.parametrize("seed", range(6))
def test_milp_matches_brute_force_and_local_search_never_loses(seed):
    rng = np.random.default_rng(seed)
    p = _random_problem(rng)
    w = rng.integers(1, 20, size=p.num_candidates).astype(float)
    best, _ = _brute_force(p, w)
    sel, info = max_weight_set(p, w)
    assert info.status == "optimal" and feasible(p, sel)
    assert w[sel].sum() == pytest.approx(best)
    greedy = greedy_complete(p, w)
    ls = local_search(p, w, greedy)
    assert feasible(p, ls) and w[ls].sum() >= w[greedy].sum() - 1e-12
    assert w[ls].sum() <= best + 1e-9


def test_backpressure_weight_is_the_commodity_differential():
    # line 0 -> 1 -> 2: queue (0,1) holds packets for 2, queue (1,2) holds packets for 2
    births = [(0.001 * i, 0, 2) for i in range(1, 6)]
    env = fixed_env(line_positions(3), births, horizon=6)
    inp = env.reset(0)
    inp = env.step([]).next_input  # births admitted: 5 packets in (0,1)
    rep = inp.report
    q01 = int(np.nonzero((rep.queue_links == [0, 1]).all(axis=1))[0][0])
    assert rep.queue_dst[q01, 2] == 5 and rep.queue_dst.sum() == rep.queues.packets.sum()
    ctl_links = [tuple(l) for l in inp.problem.links]
    w = backpressure_weights(inp)
    assert w[ctl_links.index((0, 1))] == pytest.approx(5 + 5e-3)  # Q_0^2 - Q_1^2 = 5 - 0
    # serve (0,1) once: two packets move to node 1 (queue (1,2))
    act = [ctl_links.index((0, 1))]
    inp = env.step(act).next_input
    links = [tuple(l) for l in inp.problem.links]
    w = backpressure_weights(inp)
    assert w[links.index((0, 1))] == pytest.approx(3 - 2 + 3e-3)
    assert w[links.index((1, 2))] == pytest.approx(2 - 0 + 2e-3)


@pytest.mark.parametrize("policy", ["max_weight_opt", {"type": "max_weight_opt", "weights": "backpressure"},
                                    "backpressure", "lq_local_search", "spatial_tdma"])
def test_classical_baselines_run_feasible_plans(policy):
    cfg = load_config("configs/protocol_final.toml")
    sc = eval_scenario(cfg, 0)
    sc["horizon"] = 60
    pol = build_policy(cfg, policy)
    env = build_env(cfg, run_id="t", build_graph=False, scenario_override=sc)
    inp = env.reset(10_000_003)
    for _ in range(60):
        tr = env.step(pol.act([inp], mode="greedy")[0].actions)
        inp = tr.next_input
    m = env.metrics.summary()
    assert m["sinr_violation_cycle_frac"] == 0 and m["delivered"] > 0


def test_spatial_tdma_rotates_queue_oblivious_slots():
    cfg = load_config("configs/protocol_final.toml")
    sc = eval_scenario(cfg, 0)
    sc["horizon"] = 40
    pol = build_policy(cfg, {"type": "spatial_tdma", "rebuild_every": 25})
    env = build_env(cfg, run_id="t", build_graph=False, scenario_override=sc)
    inp = env.reset(10_000_004)
    plans = []
    for k in range(40):
        acts = pol.act([inp], mode="greedy")[0].actions
        _, start, frame = pol._frames["t"]
        slot = frame[(k - start) % len(frame)]
        plan = {tuple(int(x) for x in inp.problem.links[a]) for a in acts}
        assert plan <= slot  # only the current slot's links, no work-conserving fill
        plans.append(plan)
        inp = env.step(acts).next_input
    _, _, frame = pol._frames["t"]
    assert len(frame) > 1
    bare = build_policy(cfg, {"type": "spatial_tdma", "fill_slots": False, "links": "routed"})._frame(inp.report)
    routed = set().union(*bare)
    assert sum(len(s) for s in bare) == len(routed)  # coloring: each routed link in one slot
    filled = build_policy(cfg, {"type": "spatial_tdma", "links": "routed"})._frame(inp.report)
    assert len(filled) == len(bare) and set().union(*filled) == routed
    assert all(a <= b for a, b in zip(bare, filled)) and sum(map(len, filled)) > len(routed)
