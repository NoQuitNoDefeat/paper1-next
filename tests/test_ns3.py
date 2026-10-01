"""ns-3 bridge: live round trips, lockstep ledger alignment and PHY execution checks.

Skipped unless this project's ns-3 tree is built (tools/ns3/setup.sh).
"""

import sys

import pytest

from fanet_next.backend.ns3.process import DEFAULT_NS3_ROOT

BUILT = (DEFAULT_NS3_ROOT / "build/contrib/fanet-scheduler/examples/ns3.48-fanet-bridge-default").is_file()
pytestmark = [pytest.mark.ns3, pytest.mark.skipif(not BUILT, reason="ns-3 tree not built")]


def _line_relay(execution):
    from fanet_next.backend import BACKEND
    from fanet_next.contracts import EndType, Plan
    from fanet_next.scenario import SCENARIO

    from helpers import line_positions

    sc = SCENARIO.build({"type": "fixed", "positions": line_positions(3), "horizon": 4,
                         "births": [(0.005, 0, 2), (0.006, 0, 2), (0.007, 0, 2)],
                         "radio": {"rate_bps": 1e6, "service_window_s": 0.02}}).make(0)
    be = BACKEND.build({"type": "ns3", "execution": execution,
                        "routing": {"type": "min_hop", "update_every": 1}})
    rep = be.reset(sc, run_id="t", episode=0, seed=0)
    assert rep.next_hop[0, 2] == 1 and rep.queues.packets.sum() == 0
    plans = [(), ((0, 1),), ((1, 2),), ()]
    outs = [be.execute(Plan(cycle=k, links=p)) for k, p in enumerate(plans)]
    assert outs[1].facts.served_packets.sum() == 2 and outs[1].facts.post_service.packets.sum() == 1
    assert outs[-1].end is EndType.TRUNCATED and be._sup is None  # closed cleanly
    return outs


def test_ledger_round_trip_matches_hand_computed_relay():
    f = _line_relay("ledger")[2].facts
    assert sorted(f.delivered_ids) == [0, 1]
    assert f.delivered_delays == pytest.approx([0.04 + 1024 * 8e-6 - 0.005,
                                                 0.04 + 2 * 1024 * 8e-6 - 0.006], abs=2e-9)


def test_phy_round_trip_adds_frame_overhead_and_propagation():
    f = _line_relay("phy")[2].facts
    frame = (1024 + 24) * 8e-6 + 1e-6  # identity header + 1 us propagation per packet
    assert sorted(f.delivered_ids) == [0, 1]
    assert f.delivered_delays == pytest.approx([0.04 + frame - 0.005, 0.04 + 2 * frame - 0.006],
                                               abs=2e-9)
    assert f.radio_events["rx_ok"] == 2 and "rx_error" not in f.radio_events


def _phy_run(overrides, seed, horizon=40):
    from fanet_next.config import deep_merge, load_config
    from fanet_next.experiment.assemble import build_env, build_policy

    cfg = deep_merge(load_config("configs/protocol_final.toml"),
                     {"scenario": {"horizon": horizon},
                      "backend": {"type": "ns3", "motion": False}, **overrides})
    pol = build_policy(cfg, "longest_queue")
    env = build_env(cfg, run_id="phy", build_graph=False)
    inp, facts = env.reset(seed), []
    while True:
        tr = env.step(pol.act([inp], mode="greedy")[0].actions)
        facts.append(tr.facts)
        if tr.end.value != "continue":
            return facts
        inp = tr.next_input


@pytest.mark.parametrize("interference", ["full_sinr", "pairwise"])
def test_static_phy_fails_exactly_the_links_below_threshold(interference):
    """Without motion every planned DATA head sees the whole planned set, so the PHY fails a
    link iff its cumulative SINR on the frozen gains is below threshold; followers of a
    successful head only lose interferers.  full_sinr plans therefore never fail."""
    import numpy as np

    facts = _phy_run({"constraints": {"interference": interference}}, 10_000_018)
    th = 10 ** (4.771212547196624 / 10)
    failed_total = 0
    for f in facts:
        below = {l for l, s in zip(f.planned, f.exec_sinr) if s < th * (1 - 1e-9)}
        assert set(f.failed) == below, f.cycle
        failed_total += len(f.failed)
    if interference == "full_sinr":
        assert failed_total == 0
    else:  # the positive control: ignoring cumulative interference is caught by the PHY
        assert failed_total > 0 and sum(f.radio_events.get("rx_error", 0) for f in facts) > 0
    assert sum(len(f.delivered_ids) for f in facts) > 0


def test_static_phy_with_rician_fading_fails_exactly_the_faded_links_below_threshold():
    """Rician channel (decisions.md §10): the PHY executes the private faded gains, so a
    link fails iff its cumulative SINR on those gains is below the decoding threshold;
    the scheduler planned with the fade margin, and fading still fails some links."""
    facts = _phy_run({"backend": {"type": "ns3", "motion": False,
                                  "channel": {"type": "rician", "k_factor_db": 10.0}},
                      "scenario": {"horizon": 80,
                                   "radio": {"pathloss_exponent": 2.2, "tx_power_dbm": 13.379}}},
                     10_000_019, horizon=80)
    th = 10 ** (4.771212547196624 / 10)
    failed_total = 0
    for f in facts:
        below = {l for l, s in zip(f.planned, f.exec_sinr) if s < th * (1 - 1e-9)}
        assert set(f.failed) == below, f.cycle
        failed_total += len(f.failed)
    assert failed_total > 0 and sum(len(f.delivered_ids) for f in facts) > 0


@pytest.mark.parametrize("policy,seed", [("longest_queue", 10_000_016), ("random", 10_000_017)])
def test_lockstep_alignment_with_lightweight_backend(policy, seed):
    from fanet_next.backend.ns3.alignment import align_episode
    from fanet_next.config import load_config

    r = align_episode(load_config("configs/protocol_final.toml"), seed, policy, horizon=60)
    assert r["aligned"], r["mismatches"]
    assert r["cycles"] == 60 and r["served"] > 0 and r["delivered"] > 0


def test_replayed_trajectory_runs_on_the_phy_with_motion(tmp_path):
    """A trace scene (velocity changing every cycle, positions continuous) is accepted by the
    moving PHY and delivers like the lightweight backend within a small difference."""
    import numpy as np

    from fanet_next.config import deep_merge, load_config
    from fanet_next.experiment.assemble import build_env, build_policy

    t = np.arange(0.0, 60.0, 0.2)
    pos = np.stack([np.stack([60.0 * i + 15.0 * np.sin(0.3 * t + i), 30.0 * np.cos(0.2 * t + i),
                              100.0 + 2.0 * np.sin(0.1 * t)], 1) for i in range(8)], 1)
    np.savez(tmp_path / "wavy.npz", t=t, pos=pos)
    delivered = {}
    for backend in ("lightweight", "ns3"):
        cfg = load_config("configs/protocol_final.toml")
        cfg = deep_merge(cfg, {"scenario": {"type": "trace", "dir": str(tmp_path), "train": ["wavy"],
                                            "num_nodes": 8, "horizon": 60, "flow_rate_pps": 20.0,
                                            "radio": cfg["scenario"]["radio"]}})
        if backend == "ns3":
            from fanet_next.experiment.assemble import switch_backend
            cfg = switch_backend(cfg, "ns3")
        pol = build_policy(cfg, "longest_queue")
        env = build_env(cfg, run_id="trace", build_graph=False)
        inp, n = env.reset(5), 0
        while True:
            tr = env.step(pol.act([inp], mode="greedy")[0].actions)
            n += len(tr.facts.delivered_ids)
            if tr.end.value != "continue":
                break
            inp = tr.next_input
        delivered[backend] = n
    assert delivered["ns3"] > 0 and abs(delivered["ns3"] - delivered["lightweight"]) <= 0.1 * delivered["lightweight"]
