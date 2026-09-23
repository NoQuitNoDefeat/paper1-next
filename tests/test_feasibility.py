"""Set interference, half duplex, controller masks and maximality (hand-checkable)."""

import itertools

import numpy as np
import pytest

from fanet_next.physics import set_sinr
from fanet_next.scheduling import ConstraintSet
from fanet_next.scheduling.constraints import INTERFERENCE, RESOURCE

from helpers import random_problem, toy_problem

# Three disjoint links A=(0,1), B=(2,3), C=(4,5); P = N0 = 1, signal gain 10.
LINKS = [(0, 1), (2, 3), (4, 5)]


def three_link_gain(cross: float, overrides: dict | None = None) -> np.ndarray:
    g = np.full((6, 6), 0.0)
    for (s, r) in LINKS:
        for (s2, r2) in LINKS:
            g[s, r2] = 10.0 if (s, r) == (s2, r2) else cross
    for (i, j), v in (overrides or {}).items():
        g[i, j] = v
    return g


def controller(problem, interference="full_sinr"):
    return ConstraintSet.from_config({"interference": interference}).start(problem)


def test_cumulative_interference_rejects_pairwise_compatible_triple():
    # pair SINR = 10/(1+4) = 2 >= 1.5, triple SINR = 10/(1+8) = 1.11 < 1.5
    p = toy_problem(LINKS, three_link_gain(4.0), threshold=1.5)
    assert np.allclose(set_sinr(np.array(LINKS[:2]), p.gain, p.power, p.noise), [2.0, 2.0])
    assert np.allclose(set_sinr(np.array(LINKS), p.gain, p.power, p.noise), 10 / 9)
    full = controller(p)
    full.step(0)
    full.step(1)
    assert not full.mask[2] and full.done
    pair = controller(p, "pairwise")
    pair.step(0)
    pair.step(1)
    assert pair.mask[2]  # control constraint accepts what full SINR rejects


def test_adding_a_link_must_keep_existing_links_feasible():
    # C barely hurts itself (0.1 from A) but kills A (8 from C): SINR_A = 10/9 < 1.5
    gain = three_link_gain(0.0, {(0, 5): 0.1, (4, 1): 8.0})
    p = toy_problem(LINKS, gain, threshold=1.5)
    full = controller(p)
    full.step(0)
    assert not full.mask[2]
    only_new = controller(p, "new_link_only")
    only_new.step(0)
    assert only_new.mask[2]


def test_half_duplex_blocks_shared_nodes_but_same_channel_is_not_a_conflict():
    links = [(0, 1), (1, 2), (2, 0), (3, 4)]
    gain = np.zeros((5, 5))
    for s, r in links:
        gain[s, r] = 10.0
    p = toy_problem(links, gain, threshold=1.0)
    c = controller(p)
    c.step(0)
    assert c.mask.tolist() == [False, False, False, True]  # node 0/1 busy; (3,4) is fine
    c2 = controller(p, "none")
    c2.step(3)
    assert c2.mask[:3].all()


def test_exact_threshold_is_feasible():
    p = toy_problem(LINKS[:2], three_link_gain(4.0), threshold=2.0)  # pair SINR exactly 2
    c = controller(p)
    c.step(0)
    assert c.mask[1]


@pytest.mark.parametrize("seed", range(20))
def test_incremental_mask_matches_brute_force_and_final_set_is_maximal(seed):
    rng = np.random.default_rng(seed)
    p = random_problem(rng)
    c = controller(p)
    while not c.done:
        sel = list(c.selected)
        for i in range(p.num_candidates):
            if i in sel:
                assert not c.mask[i]
                continue
            s = p.links[sel + [i]]
            nodes_ok = len(set(s.flatten())) == 2 * len(s)
            sinr_ok = np.all(set_sinr(s, p.gain, p.power, p.noise) >= p.threshold * (1 - 1e-9))
            assert c.mask[i] == (nodes_ok and sinr_ok), (sel, i)
        c.step(int(rng.choice(np.nonzero(c.mask)[0])))
    final = p.links[c.selected]
    if len(final):
        assert np.all(set_sinr(final, p.gain, p.power, p.noise) >= p.threshold * (1 - 1e-9))


def test_controller_rejects_infeasible_action_and_reports_micro_features():
    p = toy_problem(LINKS, three_link_gain(4.0), threshold=1.5)
    c = controller(p)
    f0 = c.micro_features()
    c.step(0)
    c.step(1)
    with pytest.raises(ValueError):
        c.step(2)
    assert f0[0] == 0 and c.micro_features()[0] == pytest.approx(2 / 3)
    assert c.micro_features()[2] == 1.0  # done flag


def test_every_registered_constraint_keeps_resources_and_terminates():
    rng = np.random.default_rng(0)
    for name in INTERFERENCE.names():
        for res in RESOURCE.names():
            for _ in range(5):
                p = random_problem(rng)
                c = ConstraintSet.from_config({"resource": res, "interference": name}).start(p)
                for _ in itertools.count():
                    if c.done:
                        break
                    c.step(int(np.nonzero(c.mask)[0][0]))
                nodes = p.links[c.selected].flatten()
                assert len(nodes) == len(set(nodes)), name
