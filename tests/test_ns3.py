"""ns-3 bridge: live round trip and lockstep alignment with the lightweight backend.

Skipped unless this project's ns-3 tree is built (tools/ns3/setup.sh).
"""

import sys

import pytest

from fanet_next.backend.ns3.process import DEFAULT_NS3_ROOT

BUILT = (DEFAULT_NS3_ROOT / "build/contrib/fanet-scheduler/examples/ns3.48-fanet-bridge-default").is_file()
pytestmark = [pytest.mark.ns3, pytest.mark.skipif(not BUILT, reason="ns-3 tree not built")]


def test_live_round_trip_matches_hand_computed_relay():
    from fanet_next.backend import BACKEND
    from fanet_next.contracts import EndType, Plan
    from fanet_next.scenario import SCENARIO

    from helpers import line_positions

    sc = SCENARIO.build({"type": "fixed", "positions": line_positions(3), "horizon": 4,
                         "births": [(0.005, 0, 2), (0.006, 0, 2), (0.007, 0, 2)],
                         "radio": {"rate_bps": 1e6, "service_window_s": 0.02}}).make(0)
    be = BACKEND.build({"type": "ns3", "routing": {"type": "min_hop", "update_every": 1}})
    rep = be.reset(sc, run_id="t", episode=0, seed=0)
    assert rep.next_hop[0, 2] == 1 and rep.queues.packets.sum() == 0
    plans = [(), ((0, 1),), ((1, 2),), ()]
    outs = [be.execute(Plan(cycle=k, links=p)) for k, p in enumerate(plans)]
    assert outs[1].facts.served_packets.sum() == 2 and outs[1].facts.post_service.packets.sum() == 1
    f = outs[2].facts
    assert sorted(f.delivered_ids) == [0, 1]
    assert f.delivered_delays == pytest.approx([0.04 + 1024 * 8e-6 - 0.005,
                                                 0.04 + 2 * 1024 * 8e-6 - 0.006], abs=2e-9)
    assert outs[-1].end is EndType.TRUNCATED and be._sup is None  # closed cleanly


@pytest.mark.parametrize("policy,seed", [("longest_queue", 10_000_016), ("random", 10_000_017)])
def test_lockstep_alignment_with_lightweight_backend(policy, seed):
    from fanet_next.backend.ns3.alignment import align_episode
    from fanet_next.config import load_config

    r = align_episode(load_config("configs/protocol_final.toml"), seed, policy, horizon=60)
    assert r["aligned"], r["mismatches"]
    assert r["cycles"] == 60 and r["served"] > 0 and r["delivered"] > 0
