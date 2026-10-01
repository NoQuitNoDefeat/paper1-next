"""Trace-driven scenario source: splits by whole trace, interpolation, reproducibility."""

import numpy as np
import pytest

from fanet_next.scenario import SCENARIO
from fanet_next.seeds import DEV_SEED_BASE, TEST_SEED_BASE

from helpers import fixed_env  # noqa: F401  (registers implementations)


def write_trace(path, n=6, seconds=40.0, dt=0.2, speed=(3.0, -2.0, 0.5), offset=0.0):
    t = np.arange(0.0, seconds, dt)
    base = np.stack([np.array([20.0 * i + offset, 10.0 * i, 30.0]) for i in range(n)])
    pos = base[None] + t[:, None, None] * np.array(speed)[None, None]
    np.savez(path, t=t, pos=pos)


@pytest.fixture
def traces(tmp_path):
    for name, off in (("a", 0.0), ("b", 1000.0), ("c", 2000.0)):
        write_trace(tmp_path / f"{name}.npz", offset=off)
    return tmp_path


def source(traces, **kw):
    spec = {"type": "trace", "dir": str(traces), "train": ["a"], "dev": ["b"], "test": ["c"],
            "num_nodes": 4, "horizon": 50, "flow_rate_pps": 20.0}
    spec.update(kw)
    return SCENARIO.build(spec)


def test_seed_range_selects_the_trace_of_its_split(traces):
    src = source(traces)
    assert src.make(1_000_007).trace == "a"
    assert src.make(DEV_SEED_BASE + 3).trace == "b"
    assert src.make(TEST_SEED_BASE + 3).trace == "c"


def test_positions_follow_the_trace_and_velocities_its_motion(traces):
    sc = source(traces).make(5)
    assert sc.positions(0).shape == (4, 3) and sc.positions(sc.horizon).shape == (4, 3)
    np.testing.assert_allclose(sc.velocities(7), np.tile([3.0, -2.0, 0.5], (4, 1)), atol=1e-9)
    t0 = sc.start
    expected = np.stack([np.array([20.0 * i, 10.0 * i, 30.0]) for i in sc.nodes]) + \
        (t0 + 10 * sc.cycle_length) * np.array([3.0, -2.0, 0.5])
    np.testing.assert_allclose(sc.positions(10), expected, atol=1e-9)


def test_same_seed_same_scene_and_poisson_traffic(traces):
    src = source(traces)
    a, b = src.make(11), src.make(11)
    assert a.start == b.start and np.array_equal(a.nodes, b.nodes) and a.flows == b.flows
    assert sum(len(a.births(k)) for k in range(a.horizon)) > 0
    assert sorted(s for s, _ in a.flows) == list(range(4))


def test_splits_must_be_disjoint(traces):
    with pytest.raises(ValueError):
        source(traces, test=["a"])


def test_episode_longer_than_trace_is_rejected(traces):
    with pytest.raises(ValueError):
        source(traces, horizon=5000).make(1)
