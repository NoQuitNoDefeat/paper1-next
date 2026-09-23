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
