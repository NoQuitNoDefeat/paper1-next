"""Lightweight backend semantics on hand-checkable scripted scenes."""

import numpy as np
import pytest

from fanet_next.contracts import EndType, ExecutionError, Plan

from helpers import fixed_env, line_positions

T = 0.02
BIT = 8 / 1e6  # seconds per byte at 1 Mbps


def q_of(report, link):
    ql = report.queue_links
    return int(np.nonzero((ql[:, 0] == link[0]) & (ql[:, 1] == link[1]))[0][0])


def test_line_relay_service_timing_snapshot_and_reward():
    # 0 -- 100 m -- 1 -- 100 m -- 2 ; 0 and 2 are out of range, route 0 -> 1 -> 2
    births = [(0.005, 0, 2), (0.006, 0, 2), (0.007, 0, 2)]
    env = fixed_env(line_positions(3), births, horizon=5)
    inp = env.reset(0)
    assert inp.report.next_hop[0, 2] == 1 and inp.report.next_hop[1, 2] == 2
    assert inp.problem.num_candidates == 0  # nothing queued yet

    # cycle 0: empty plan (no actor action); births are staged and admitted at t=0.02
    tr = env.step([])
    f = tr.facts
    assert f.births == 3 and f.risk_packets == 3
    assert f.post_service.packets.sum() == 0  # sampled before admission
    assert f.queued_end == 3
    rep = tr.next_input.report
    q01 = q_of(rep, (0, 1))
    assert rep.queues.packets[q01] == 3 and rep.queues.hol_wait[q01] == pytest.approx(0.0)

    # cycle 1: serve (0,1): two whole packets fit in 2500 B
    assert [tuple(x) for x in tr.next_input.problem.links] == [(0, 1)]
    tr = env.step([0])
    f = tr.facts
    assert f.served_packets[q01] == 2 and f.served_bytes[q01] == 2048
    assert f.post_service.packets[q01] == 1
    assert f.post_service.hol_wait[q01] == pytest.approx(2 * T - T)  # enqueued at 0.02, sampled 0.04
    q12 = q_of(rep, (1, 2))
    assert f.post_service.packets[q12] == 0  # relay arrivals are staged until the boundary
    # reward components over the 6 registered queues
    parts = tr.reward.parts
    assert parts["service"] == pytest.approx((2048 / 2500) / 6)
    assert parts["queue"] == pytest.approx((1 / 64) / 6)
    assert parts["delay"] == pytest.approx((1 / 64) * (0.02 / 0.2) / 6)
    assert parts["violation"] == 0.0

    # cycle 2: (0,1) and (1,2) share node 1 → only one of them; serve (1,2) → deliveries
    links = [tuple(x) for x in tr.next_input.problem.links]
    assert set(links) == {(0, 1), (1, 2)}
    tr = env.step([links.index((1, 2))])
    f = tr.facts
    assert sorted(f.delivered_ids) == [0, 1]
    t0 = 2 * T
    assert f.delivered_delays[0] == pytest.approx(t0 + 1024 * BIT - 0.005)
    assert f.delivered_delays[1] == pytest.approx(t0 + 2 * 1024 * BIT - 0.006)


def test_new_head_of_line_uses_its_own_enqueue_time():
    # one packet admitted at 0.02, a second at 0.04; with one-packet service the new head's
    # HOL after serving the first is measured from 0.04
    births = [(0.005, 0, 1), (0.025, 0, 1)]
    env = fixed_env(line_positions(2), births, horizon=5,
                    radio={"rate_bps": 0.5e6, "service_window_s": 0.02})  # 1250 B → 1 packet
    env.reset(0)
    env.step([])  # cycle 0: admit first
    tr = env.step([])  # cycle 1: admit second
    q = q_of(tr.next_input.report, (0, 1))
    assert tr.next_input.report.queues.hol_wait[q] == pytest.approx(0.02)
    tr = env.step([0])  # cycle 2: serve the older packet
    assert tr.facts.post_service.packets[q] == 1
    assert tr.facts.post_service.hol_wait[q] == pytest.approx(0.06 - 0.04)


def test_waiting_area_timeout_is_a_termination_and_counts_in_violation():
    # node 2 is unreachable: packets 0 -> 2 wait, then time out after max_wait
    pos = [[0, 0, 100], [100, 0, 100], [2000, 0, 100]]
    env = fixed_env(pos, [(0.005, 0, 2), (0.006, 0, 2)], horizon=6, waiting_max_wait=0.06)
    env.reset(0)
    tr = env.step([])
    assert tr.facts.moved_to_waiting == 2 and tr.facts.waiting_end == 2
    assert tr.next_input.report.waiting_packets[0] == 2
    assert tr.next_input.problem.num_candidates == 0  # waiting is not a schedulable link
    tr = env.step([])
    assert tr.next_input.report.waiting_oldest[0] == pytest.approx(0.02)
    tr = env.step([])
    assert tr.facts.terminations == {}
    tr = env.step([])  # waiting since 0.02, max wait 0.06 → expires at 0.08 (end of cycle 3)
    assert sorted(tr.facts.terminations["waiting_timeout"]) == [0, 1]
    assert tr.facts.risk_packets == 2
    assert tr.reward.parts["violation"] == pytest.approx(1.0)
    assert tr.facts.waiting_post_service == 2  # post-service sample precedes the expiry


def test_queue_overflow_and_conservation():
    births = [(0.001 * (i + 1), 0, 1) for i in range(5)]
    env = fixed_env(line_positions(2), births, horizon=3, queue_capacity=3)
    env.reset(0)
    tr = env.step([])
    assert sorted(tr.facts.terminations["queue_overflow"]) == [3, 4]  # FIFO by arrival time
    assert tr.facts.queued_end == 3
    b = env.backend
    assert b.totals["born"] == b.totals["delivered"] + b.totals["terminated"] + sum(b.in_system())


def _moving_scene(policy):
    # 0 -> 3 goes via 1 at cycle 0; at cycle 2 node 1 moves away and 2 takes over
    near = [[0, 0, 100], [100, 10, 100], [100, -400, 100], [200, 0, 100]]
    far = [[0, 0, 100], [100, 900, 100], [100, -10, 100], [200, 0, 100]]
    positions = [near, near, far, far, far]
    births = [(0.005, 0, 3), (0.006, 0, 3)]
    backend = {"type": "lightweight", "stale_queue_policy": policy,
               "routing": {"type": "min_hop", "update_every": 1}}
    return fixed_env(positions, births, horizon=4, backend=backend)


def test_route_change_rehomes_or_keeps_stale_queues():
    env = _moving_scene("rehome")
    env.reset(0)
    env.step([])
    tr = env.step([])  # boundary to cycle 2 recomputes routes with the far layout
    rep = tr.next_input.report
    assert rep.next_hop[0, 3] == 2
    assert tr.facts.rehomed == 2
    assert rep.queues.packets[q_of(rep, (0, 2))] == 2 and rep.queues.packets[q_of(rep, (0, 1))] == 0

    env = _moving_scene("keep")
    env.reset(0)
    env.step([])
    tr = env.step([])
    rep = tr.next_input.report
    assert rep.queues.packets[q_of(rep, (0, 1))] == 2
    assert tr.next_input.problem.num_candidates == 0  # stranded: (0,1) is no longer a route/link


def test_invalid_plans_are_execution_errors_and_horizon_truncates():
    env = fixed_env(line_positions(3), [(0.005, 0, 2)], horizon=2)
    inp = env.reset(0)
    with pytest.raises(ExecutionError):
        env.backend.execute(Plan(cycle=0, links=((0, 1), (1, 2))))  # node 1 twice
    with pytest.raises(ExecutionError):
        env.backend.execute(Plan(cycle=5, links=()))
    assert env.step([]).end is EndType.CONTINUE
    tr = env.step([])
    assert tr.end is EndType.TRUNCATED and tr.next_input.report.cycle == 2


def test_failed_transmission_keeps_packet_and_is_not_a_termination():
    # with the "none" control two links that break each other are both scheduled
    pos = [[0, 0, 100], [100, 0, 100], [110, 20, 100], [10, 20, 100]]
    births = [(0.005, 0, 1), (0.006, 2, 3)]
    env = fixed_env(pos, births, horizon=4, interference="none")
    env.reset(0)
    tr = env.step([])
    links = [tuple(x) for x in tr.next_input.problem.links]
    tr = env.step([links.index((0, 1)), links.index((2, 3))])
    assert set(tr.facts.failed) == {(0, 1), (2, 3)} and tr.facts.served_packets.sum() == 0
    assert tr.facts.terminations == {} and tr.facts.queued_end == 2


def test_violation_normalisation_primary_vs_fixed_reference_control():
    from fanet_next.reward.standard import REWARD

    pos = [[0, 0, 100], [100, 0, 100], [2000, 0, 100]]
    env = fixed_env(pos, [(0.005, 0, 2), (0.006, 0, 2), (0.007, 0, 1)], horizon=6,
                    waiting_max_wait=0.06)
    env.reset(0)
    for _ in range(4):
        tr = env.step([] if not env.current.problem.num_candidates else [0])
    facts = tr.facts
    assert len(facts.terminations["waiting_timeout"]) == 2
    primary = REWARD.build({"type": "standard"})(facts).parts["violation"]
    control = REWARD.build({"type": "fixed_violation_ref", "violation_ref_packets": 10})(facts)
    assert primary == pytest.approx(2 / facts.risk_packets)
    assert control.parts["violation"] == pytest.approx(2 / 10)


def test_ontime_ratio_counts_drops_as_late():
    from fanet_next.reward.metrics import MetricsAccumulator

    env = fixed_env(line_positions(2), [(0.005, 0, 1), (0.006, 0, 1), (0.007, 0, 1)], horizon=3,
                    queue_capacity=2)
    env.reset(0)
    for _ in range(3):
        env.step(list(range(env.current.problem.num_candidates))[:1])
    s = env.metrics.summary()
    assert s["born"] == 3 and s["delivered"] == 2 and s["term_queue_overflow"] == 1
    assert s["ontime_1s"] == pytest.approx(2 / 3)  # the dropped packet is never on time
    assert isinstance(MetricsAccumulator().summary()["ontime_2s"], float)


@pytest.mark.parametrize("reward", ["standard", "fixed_violation_ref"])
def test_reward_splits_exactly_over_nodes(reward):
    """terminated_nodes counts every terminated packet once; reward.by_node sums to the reward."""
    from fanet_next.config import load_config
    from fanet_next.experiment.assemble import build_env, build_policy

    cfg = load_config("configs/protocol_final.toml", ["scenario.flow_rate_pps=60.0", "scenario.horizon=80",
                                                       f"reward.type={reward}"])
    env = build_env(cfg, run_id="t", build_graph=False)
    policy = build_policy(cfg, "longest_queue", seed=0)
    inp = env.reset(1)
    dropped = 0
    for _ in range(80):
        tr = env.step(policy.act([inp])[0].actions)
        f = tr.facts
        assert sum(f.terminated_nodes.values()) == len(set(f.terminated_ids))
        split = env.reward.by_node(f, env.backend.qlinks, env.backend.sc.num_nodes)
        assert split.sum() == pytest.approx(tr.reward.total, abs=1e-12)
        dropped += len(f.terminated_ids)
        inp = tr.next_input
        if tr.end is not EndType.CONTINUE:
            break
    assert dropped > 0
