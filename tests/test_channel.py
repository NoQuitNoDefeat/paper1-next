"""Air-to-air Rician channel (decisions.md §10): fading statistics, fade margin, planning
threshold, and ns-3 private frames that reproduce the lightweight backend's draws."""

import math

import numpy as np
import pytest

from fanet_next.backend import BACKEND
from fanet_next.backend.ns3.convert import build_episode, radio_settings
from fanet_next.contracts import Plan
from fanet_next.physics import RadioParams, db_to_lin, set_sinr
from fanet_next.scenario import CHANNEL, SCENARIO
from fanet_next.scenario.channel import rician_margin_db, rician_power
from fanet_next.scenario.routing import ROUTING

from helpers import line_positions

A2A = {"pathloss_exponent": 2.2, "tx_power_dbm": 13.379}  # same 161 m range as the base radio


def test_rician_power_is_reciprocal_unit_mean_with_rician_variance():
    rng = np.random.default_rng(0)
    k = 10.0
    p = np.stack([rician_power(k, 6, rng) for _ in range(4000)])
    assert np.array_equal(p, p.transpose(0, 2, 1)) and not p[:, range(6), range(6)].any()
    x = p[:, *np.triu_indices(6, 1)].ravel()
    assert x.mean() == pytest.approx(1.0, abs=0.01)
    assert x.var() == pytest.approx((1 + 2 * k) / (1 + k) ** 2, rel=0.05)


@pytest.mark.parametrize("k_db,outage", [(5.0, 0.1), (10.0, 0.1), (10.0, 0.01), (15.0, 0.1)])
def test_fade_margin_gives_the_target_outage(k_db, outage):
    margin = rician_margin_db(k_db, outage)
    rng = np.random.default_rng(1)
    x = np.concatenate([rician_power(db_to_lin(k_db), 200, rng)[np.triu_indices(200, 1)]
                        for _ in range(3)])
    sd = math.sqrt(outage * (1 - outage) / len(x))
    assert np.mean(x < db_to_lin(-margin)) == pytest.approx(outage, abs=4 * sd)


def test_margin_shrinks_with_a_stronger_line_of_sight():
    assert rician_margin_db(5, 0.1) > rician_margin_db(10, 0.1) > rician_margin_db(15, 0.1) > 0


def _pair_at_planning_boundary(channel: dict, horizon: int):
    """Two static nodes whose interference-free SNR equals the planning threshold."""
    radio = RadioParams(**A2A)
    margin = CHANNEL.build(channel).fade_margin_db
    budget = (radio.tx_power_dbm - radio.noise_dbm - radio.sinr_threshold_db - margin
              - radio.pathloss_ref_db)
    d = 10 ** (budget / (10 * radio.pathloss_exponent))
    sc = SCENARIO.build({"type": "fixed", "positions": [[0, 0, 100], [d, 0, 100]],
                         "births": [], "horizon": horizon, "radio": A2A}).make(0)
    return sc, radio, margin


def test_scheduler_plans_with_the_margin_and_boundary_links_fail_at_the_outage_rate():
    channel = {"type": "rician", "k_factor_db": 10.0, "outage": 0.1}
    sc, radio, margin = _pair_at_planning_boundary(channel, horizon=3000)
    be = BACKEND.build({"type": "lightweight", "channel": channel})
    rep = be.reset(sc, seed=4)
    assert rep.threshold == pytest.approx(radio.threshold * db_to_lin(margin))
    snr = rep.tx_power[0] * rep.gain[0, 1] / rep.noise
    assert snr == pytest.approx(rep.threshold)  # planned exactly at the boundary
    failed = sum(len(be.execute(Plan(cycle=k, links=((0, 1),))).facts.failed)
                 for k in range(sc.horizon))
    assert failed / sc.horizon == pytest.approx(0.1, abs=4 * math.sqrt(0.09 / sc.horizon))


def test_ideal_channel_keeps_the_decoding_threshold():
    sc = SCENARIO.build({"type": "fixed", "positions": line_positions(3), "births": [],
                         "horizon": 2}).make(0)
    rep = BACKEND.build({"type": "lightweight"}).reset(sc, seed=0)
    assert rep.threshold == RadioParams().threshold


def test_shadowing_is_reciprocal_fixed_per_episode_and_known_to_the_scheduler():
    ch = CHANNEL.build({"type": "rician", "shadowing_db": 2.0})
    radio = RadioParams(**A2A)
    pos = np.array(line_positions(5, spacing=50.0), dtype=float)
    ch.reset(5, np.random.default_rng(7))
    g1, g2 = ch.estimate(pos, radio, None), ch.estimate(pos, radio, None)
    assert np.array_equal(g1, g2) and np.allclose(g1, g1.T)
    assert np.array_equal(ch.true_gain(pos, radio), g1)
    ch.reset(5, np.random.default_rng(8))
    assert not np.allclose(ch.estimate(pos, radio, None), g1)


def test_shadowing_leaves_the_fading_draws_unchanged():
    """Same seed with and without shadowing: identical fading per pair and cycle, so a
    shadowing on/off comparison differs in the shadowing only."""
    radio = RadioParams(**A2A)
    pos = np.array(line_positions(6, spacing=60.0), dtype=float)
    fades = []
    for shadowing_db in (0.0, 5.0):
        ch = CHANNEL.build({"type": "rician", "shadowing_db": shadowing_db})
        rng = np.random.default_rng(3)
        ch.reset(6, rng)
        mean = ch.estimate(pos, radio, rng)
        fades.append([ch.execution_at(pos, radio, rng)[np.triu_indices(6, 1)] / mean[np.triu_indices(6, 1)]
                      for _ in range(5)])
    assert np.allclose(fades[0], fades[1], rtol=1e-12)


@pytest.mark.parametrize("shadowing_db", [0.0, 2.0])
def test_ns3_private_frames_reproduce_the_lightweight_draws(shadowing_db):
    """Static nodes (no mid-window shift): the PHY's private gains are exactly the gains the
    lightweight backend executes with, cycle by cycle, from the same seed."""
    channel = {"type": "rician", "k_factor_db": 10.0, "shadowing_db": shadowing_db}
    pos = line_positions(6, spacing=70.0)
    sc = SCENARIO.build({"type": "fixed", "positions": pos, "births": [], "horizon": 12,
                         "radio": A2A}).make(0)
    plans = [((0, 1), (3, 2)), ((1, 2), (4, 5)), ((5, 4),)] * 4
    be = BACKEND.build({"type": "lightweight", "channel": channel})
    rep, seed = be.reset(sc, seed=11), 11
    light = [be.execute(Plan(cycle=k, links=p)).facts.exec_sinr for k, p in enumerate(plans)]
    ep = build_episode(sc, ROUTING.build("min_hop"), wireless=radio_settings(sc, motion=False),
                       channel=CHANNEL.build(channel), seed=seed)
    assert ep.plan_threshold == pytest.approx(rep.threshold)
    assert np.array_equal(ep.gains[0], rep.gain)
    for k, p in enumerate(plans):
        ns3 = set_sinr(np.array(p), ep.exec_gains[k], ep.power, ep.noise)
        assert np.array_equal(ns3, light[k]), k
    frames = ep.payload["wireless"]["frames"]
    assert len(frames) == sc.horizon + 1
    assert ([l["key"] for l in frames[3]["links"]]
            == [l["key"] for l in ep.payload["physical_inputs"][3]["links"]])


def test_ns3_rejects_channels_it_cannot_reproduce():
    with pytest.raises(ValueError):
        BACKEND.build({"type": "ns3", "channel": "lognormal"})
    with pytest.raises(ValueError):
        BACKEND.build({"type": "ns3", "channel": "rician", "execution": "ledger"})
    BACKEND.build({"type": "ns3", "channel": "rician"})  # PHY execution: accepted


def test_routes_only_use_links_the_scheduler_can_use():
    """With a fade margin the planning threshold exceeds the decoding threshold: every
    next hop chosen at a route update must reach the planning threshold on its own, in
    both backends (otherwise packets wait on links that can never be scheduled)."""
    channel = {"type": "rician", "k_factor_db": 10.0, "outage": 0.01}  # 6.2 dB > 6 dB routing margin
    src = SCENARIO.build({"type": "random", "num_nodes": 16, "area_m": 300.0, "horizon": 30,
                          "radio": {**A2A, "rate_bps": 4e6}})
    routing = {"type": "min_hop", "update_every": 25, "snr_margin_db": 6.0}
    for seed in (1, 2, 3):
        sc = src.make(seed)
        be = BACKEND.build({"type": "lightweight", "channel": channel, "routing": routing})
        rep = be.reset(sc, seed=seed)
        snr = rep.tx_power[:, None] * rep.gain / rep.noise
        hops = {(u, int(rep.next_hop[u, d])) for u in range(sc.num_nodes) for d in range(sc.num_nodes)
                if rep.next_hop[u, d] >= 0}
        assert hops and all(snr[u, v] >= rep.threshold * (1 - 1e-9) for u, v in hops)
        ep = build_episode(sc, ROUTING.build(routing), wireless=radio_settings(sc, motion=False),
                           channel=CHANNEL.build(channel), seed=seed)
        assert np.array_equal(ep.next_hops[0], rep.next_hop)
