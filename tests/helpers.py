"""Shared builders for tests."""

from __future__ import annotations

import numpy as np

from fanet_next.backend import BACKEND
from fanet_next.experiment import assemble  # noqa: F401  (registers all implementations)
from fanet_next.loop import SchedulingEnv
from fanet_next.observation import OBSERVATION
from fanet_next.reward.standard import REWARD
from fanet_next.scenario import SCENARIO
from fanet_next.scheduling import CANDIDATES, ConstraintSet, SchedulingProblem

# 1 Mbps x 20 ms = 2500 B -> exactly two 1024 B packets per scheduled link and cycle
TWO_PACKET_RADIO = {"rate_bps": 1e6, "service_window_s": 0.02}


def line_positions(n: int, spacing: float = 100.0) -> list[list[float]]:
    return [[spacing * i, 0.0, 100.0] for i in range(n)]


def fixed_env(positions, births, *, horizon=10, backend=None, interference="full_sinr",
              radio=None, build_graph=True, waiting_max_wait=1.0, waiting_capacity=16,
              queue_capacity=64, reward=None, candidates=None) -> SchedulingEnv:
    scenario = {"type": "fixed", "positions": positions, "births": births, "horizon": horizon,
                "radio": radio or TWO_PACKET_RADIO, "waiting_max_wait": waiting_max_wait,
                "waiting_capacity": waiting_capacity, "queue_capacity": queue_capacity}
    return SchedulingEnv(
        scenario_source=SCENARIO.build(scenario),
        backend=BACKEND.build(backend or {"type": "lightweight", "routing": {"type": "min_hop", "update_every": 1}}),
        candidates=CANDIDATES.build(candidates or "standard"),
        constraints=ConstraintSet.from_config({"interference": interference}),
        observation=OBSERVATION.build("standard"),
        reward=REWARD.build(reward or "standard"), build_graph=build_graph, run_id="test")


def toy_problem(links, gain, power=1.0, noise=1.0, threshold=1.5) -> SchedulingProblem:
    gain = np.asarray(gain, dtype=float)
    n = len(gain)
    return SchedulingProblem(links=np.asarray(links), gain=gain, power=np.full(n, power),
                             noise=noise, threshold=threshold)


def random_problem(rng: np.random.Generator, n_nodes=10, n_links=14, threshold=3.0) -> SchedulingProblem:
    pos = rng.uniform(0, 300, size=(n_nodes, 3))
    from fanet_next.physics import RadioParams, path_gain

    radio = RadioParams()
    gain = path_gain(pos, radio)
    pairs = [(u, v) for u in range(n_nodes) for v in range(n_nodes) if u != v]
    snr = np.array([radio.tx_power_w * gain[u, v] / radio.noise_w for u, v in pairs])
    ok = [p for p, s in zip(pairs, snr) if s >= threshold]
    rng.shuffle(ok)
    links = sorted(ok[:n_links])
    return SchedulingProblem(links=np.array(links).reshape(-1, 2), gain=gain,
                             power=np.full(n_nodes, radio.tx_power_w), noise=radio.noise_w,
                             threshold=threshold)
