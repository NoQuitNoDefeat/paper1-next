"""Mixture scenario source (E14) and per-episode area ranges of the random source."""

import numpy as np
import pytest

from fanet_next.config import load_config
from fanet_next.experiment.evaluate import eval_scenario
from fanet_next.scenario import SCENARIO

SLOW = {"speed_mps": [0.0, 5.0], "area_m": [150.0, 300.0], "num_nodes": [16, 30]}


def _mixture(weights=(0.5, 0.5), **shared):
    shared = {"num_nodes": 16, "area_m": 300.0, "speed_mps": [5.0, 30.0], "horizon": 30, **shared}
    return SCENARIO.build({"type": "mixture", "base_type": "random", "weights": list(weights),
                           "components": [{}, SLOW], **shared})


def _same(a, b):
    return (a.num_nodes == b.num_nodes and a.horizon == b.horizon and a.flows == b.flows
            and all(np.array_equal(a.positions(k), b.positions(k)) for k in range(a.horizon + 1))
            and all(a.births(k) == b.births(k) for k in range(a.horizon)))


def test_an_episode_is_exactly_its_components_episode():
    mix = _mixture()
    picked = set()
    for seed in range(40):
        i = mix.component_of(seed)
        picked.add(i)
        assert _same(mix.make(seed), mix.components[i].make(seed))
    assert picked == {0, 1}
    # the original component is the plain random source: same episodes for the same seeds
    plain = SCENARIO.build({"type": "random", "num_nodes": 16, "area_m": 300.0, "speed_mps": [5.0, 30.0],
                            "horizon": 30})
    seed = next(s for s in range(40) if mix.component_of(s) == 0)
    assert _same(mix.make(seed), plain.make(seed))


def test_components_are_picked_with_the_given_weights():
    mix = _mixture(weights=(0.25, 0.75))
    share = np.mean([mix.component_of(s) for s in range(4000)])
    assert share == pytest.approx(0.75, abs=4 * np.sqrt(0.75 * 0.25 / 4000))
    only = _mixture(weights=(1.0, 0.0))
    assert {only.component_of(s) for s in range(200)} == {0}


def test_component_overrides_and_shared_parameters():
    mix = _mixture()
    slow = [mix.make(s) for s in range(60) if mix.component_of(s) == 1]
    fast = [mix.make(s) for s in range(60) if mix.component_of(s) == 0]
    assert slow and fast
    assert all(0.0 <= sc.speed_mps <= 5.0 and 16 <= sc.num_nodes <= 30 for sc in slow)
    assert all(150.0 <= sc.motion_bounds()[1][0] <= 300.0 for sc in slow)
    assert all(5.0 <= sc.speed_mps <= 30.0 and sc.num_nodes == 16 for sc in fast)
    assert all(sc.motion_bounds()[1][0] == 300.0 for sc in fast)
    assert len({sc.motion_bounds()[1][0] for sc in slow}) == len(slow)  # area drawn per episode
    assert all(sc.horizon == 30 for sc in slow + fast)


def test_drain_evaluation_reaches_every_component():
    cfg = load_config("configs/explore/e14_broad.toml", ["scenario.horizon=40"])
    mix = SCENARIO.build(eval_scenario(cfg, drain_cycles=20))
    for seed in range(12):
        sc = mix.make(seed)
        assert sc.horizon == 60 and sc.traffic_done(40) and not any(sc.births(k) for k in range(40, 60))


def test_mixture_rejects_bad_weights_and_mixed_radios():
    with pytest.raises(ValueError):
        _mixture(weights=(1.0,))
    with pytest.raises(ValueError):
        _mixture(weights=(0.0, 0.0))
    with pytest.raises(ValueError):
        SCENARIO.build({"type": "mixture", "base_type": "random",
                        "components": [{}, {"radio": {"tx_power_dbm": 10.0}}]})
