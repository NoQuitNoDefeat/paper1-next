"""Time-varying-discount GAE: micro vs boundary transitions, truncation, cut, termination."""

import numpy as np
import pytest

from fanet_next.model.runner import MicroRecord
from fanet_next.training.advantages import compute_stream_advantages, flatten_stream
from fanet_next.training.records import CycleRecord

G, LAM = 0.9, 0.8


def rec(values, reward, end, bootstrap=0.0):
    k = len(values) - 1
    micro = MicroRecord(graph=None, masks=np.zeros((k + 1, 0), bool),
                        actions=np.zeros(k, np.int64), logp=np.zeros(k, np.float32),
                        values=np.array(values, np.float32), micro=np.zeros((k + 1, 4), np.float32))
    return CycleRecord(micro=micro, reward=reward, raw_reward=reward, end=end, bootstrap=bootstrap)


def test_hand_computed_stream_with_truncation():
    v0, v1, w0, b = 1.0, 2.0, 0.5, 3.0
    r1, r2 = 0.3, -0.2
    stream = [rec([v0, v1], r1, "continue"), rec([w0], r2, "truncated", bootstrap=b)]
    compute_stream_advantages(stream, G, LAM)
    d2 = r2 + G * b - w0  # boundary, truncated: keep V(last valid state)
    d1 = r1 + G * w0 - v1  # boundary: physical discount once per cycle
    d0 = v1 - v0  # micro step: reward 0, discount 1
    a2 = d2
    a1 = d1 + G * LAM * a2
    a0 = d0 + 1.0 * LAM * a1  # lambda acts per micro transition
    assert stream[0].advantages == pytest.approx([a0, a1], rel=1e-6)
    assert stream[1].advantages == pytest.approx([a2], rel=1e-6)
    assert stream[0].returns == pytest.approx([a0 + v0, a1 + v1], rel=1e-6)


def test_termination_cancels_future_value_and_cut_stops_recursion():
    stream = [rec([1.0], 0.5, "terminated"), rec([2.0], 1.0, "cut", bootstrap=10.0)]
    compute_stream_advantages(stream, G, LAM)
    assert stream[0].advantages[0] == pytest.approx(0.5 - 1.0)  # no V(next)
    assert stream[1].advantages[0] == pytest.approx(1.0 + G * 10.0 - 2.0)
    # cut: the second record does not leak into the first
    s2 = [rec([1.0], 0.5, "cut", bootstrap=4.0), rec([2.0], 1.0, "truncated", bootstrap=0.0)]
    compute_stream_advantages(s2, G, LAM)
    assert s2[0].advantages[0] == pytest.approx(0.5 + G * 4.0 - 1.0)


def test_micro_step_count_does_not_change_physical_discount():
    # same cycle reward, different number of micro steps with constant values:
    # the boundary TD error sees gamma exactly once either way
    one = [rec([1.0, 1.0], 1.0, "continue"), rec([1.0], 0.0, "cut", bootstrap=1.0)]
    three = [rec([1.0, 1.0, 1.0, 1.0], 1.0, "continue"), rec([1.0], 0.0, "cut", bootstrap=1.0)]
    r1, _, n1, g1, _, _ = flatten_stream(one, G)
    r3, _, n3, g3, _, _ = flatten_stream(three, G)
    assert np.prod(g1) == pytest.approx(np.prod(g3)) == pytest.approx(G ** 2)
    assert sum(r1) == sum(r3) == 1.0


def test_stream_must_end_with_a_bootstrap_marker():
    with pytest.raises(ValueError):
        compute_stream_advantages([rec([1.0], 0.0, "continue")], G, LAM)


def test_cycle_credit_gives_every_micro_action_the_cycle_level_advantage():
    """credit = "cycle": GAE over cycle-start values only; value targets unchanged."""
    v0, v1, w0, b = 1.0, 2.0, 0.5, 3.0
    r1, r2 = 0.3, -0.2
    step = [rec([v0, v1], r1, "continue"), rec([w0], r2, "truncated", bootstrap=b)]
    cyc = [rec([v0, v1], r1, "continue"), rec([w0], r2, "truncated", bootstrap=b)]
    compute_stream_advantages(step, G, LAM)
    compute_stream_advantages(cyc, G, LAM, credit="cycle")
    a2 = r2 + G * b - w0
    a1 = r1 + G * w0 - v0 + G * LAM * a2  # within-cycle value step v1 - v0 does not enter
    assert cyc[0].advantages == pytest.approx([a1, a1], rel=1e-6)
    assert cyc[1].advantages == pytest.approx([a2], rel=1e-6)
    for s_, c_ in zip(step, cyc):
        assert np.array_equal(s_.returns, c_.returns)
    with pytest.raises(ValueError):
        compute_stream_advantages(cyc, G, LAM, credit="other")


def test_flat_credit_removes_within_cycle_value_changes_and_keeps_lambda_per_micro_step():
    v0, v1, w0, b = 1.0, 2.0, 0.5, 3.0
    r1, r2 = 0.3, -0.2
    flat = [rec([v0, v1], r1, "continue"), rec([w0], r2, "truncated", bootstrap=b)]
    compute_stream_advantages(flat, G, LAM, credit="flat")
    a2 = r2 + G * b - w0
    d1 = r1 + G * w0 - v0  # complete-plan value v1 replaced by the cycle-start value v0
    a1 = d1 + G * LAM * a2
    a0 = 0.0 + LAM * a1  # micro step: value change v1 - v0 removed, lambda still per micro step
    assert flat[0].advantages == pytest.approx([a0, a1], rel=1e-6)
    assert flat[1].advantages == pytest.approx([a2], rel=1e-6)
    step = [rec([v0, v1], r1, "continue"), rec([w0], r2, "truncated", bootstrap=b)]
    compute_stream_advantages(step, G, LAM)
    assert np.array_equal(flat[0].returns, step[0].returns)


def test_advantage_lambda_changes_only_the_advantages():
    """adv_lam (E18): the actor's advantages use their own lambda; value targets keep lam."""
    v0, v1, w0, b = 1.0, 2.0, 0.5, 3.0
    r1, r2 = 0.3, -0.2
    one = [rec([v0, v1], r1, "continue"), rec([w0], r2, "truncated", bootstrap=b)]
    compute_stream_advantages(one, G, LAM, adv_lam=1.0)
    a2 = r2 + G * b - w0
    a1 = r1 + G * w0 - v1 + G * 1.0 * a2
    a0 = (v1 - v0) + 1.0 * a1  # lambda = 1: within-chunk Monte Carlo minus the baseline v0
    assert one[0].advantages == pytest.approx([a0, a1], rel=1e-6)
    assert a0 == pytest.approx(r1 + G * (r2 + G * b) - v0)
    step = [rec([v0, v1], r1, "continue"), rec([w0], r2, "truncated", bootstrap=b)]
    same = [rec([v0, v1], r1, "continue"), rec([w0], r2, "truncated", bootstrap=b)]
    compute_stream_advantages(step, G, LAM)
    compute_stream_advantages(same, G, LAM, adv_lam=LAM)
    for s_, o_, m_ in zip(step, one, same):
        assert np.array_equal(s_.returns, o_.returns)
        assert np.array_equal(s_.advantages, m_.advantages) and np.array_equal(s_.returns, m_.returns)


def test_no_critic_credit_ignores_every_value():
    """credit = "none" (E18): discounted reward-to-go, no baseline, no bootstrap; targets unchanged."""
    v0, v1, w0, b = 1.0, 2.0, 0.5, 3.0
    r1, r2 = 0.3, -0.2
    none = [rec([v0, v1], r1, "continue"), rec([w0], r2, "truncated", bootstrap=b)]
    compute_stream_advantages(none, G, LAM, credit="none", adv_lam=1.0)
    assert none[1].advantages == pytest.approx([r2], rel=1e-6)
    assert none[0].advantages == pytest.approx([r1 + G * r2, r1 + G * r2], rel=1e-6)
    other = [rec([5.0, -4.0], r1, "continue"), rec([7.0], r2, "truncated", bootstrap=-9.0)]
    compute_stream_advantages(other, G, LAM, credit="none", adv_lam=1.0)
    for n_, o_ in zip(none, other):
        assert np.array_equal(n_.advantages, o_.advantages)
    step = [rec([v0, v1], r1, "continue"), rec([w0], r2, "truncated", bootstrap=b)]
    compute_stream_advantages(step, G, LAM)
    assert np.array_equal(none[0].returns, step[0].returns)
