"""Training-only (privileged) critic input, E21: the future exogenous load of every node.

The arrivals of the training scenarios are drawn in advance and do not depend on the schedule,
so a critic that sees them is a valid input-dependent baseline (it never changes the expected
policy gradient), while the actor keeps the decision-time observation.  Per node and per bin of
future cycles [k + lo, k + hi): the arrivals it originates and the arrivals routed through it as a
relay along the current next-hop table, both per cycle.  Cycles beyond the scenario horizon (the
training episode's end, not a traffic stop) are filled with their expectation: each flow's rate
times the cycle length.
"""

from __future__ import annotations

import numpy as np

DEFAULT_BINS = (10, 30, 100)


def _relays(next_hop: np.ndarray, src: int, dst: int) -> list[int]:
    out, x, n = [], src, len(next_hop)
    for _ in range(n):
        nh = int(next_hop[x, dst])
        if nh < 0 or nh == dst:
            break
        out.append(nh)
        x = nh
    return out


def future_load(scenario, report, bins=DEFAULT_BINS, placebo: bool = False) -> np.ndarray:
    """(num_nodes, 2 * len(bins)) float32: per bin, originated and relayed arrivals per cycle."""
    n, k = report.num_nodes, int(report.cycle)
    out = np.zeros((n, 2 * len(bins)), dtype=np.float32)
    if placebo:
        return out
    nh = report.next_hop
    horizon = int(getattr(scenario, "horizon", 0))
    rate = float(getattr(scenario, "flow_rate_pps", 0.0)) * float(report.cycle_length)
    flows = list(getattr(scenario, "flows", []))
    route = {}
    lo = 0
    for b, hi in enumerate(bins):
        real_end = min(k + hi, horizon)
        for c in range(k + lo, real_end):
            for birth in scenario.births(c):
                s, d = int(birth.src), int(birth.dst)
                out[s, 2 * b] += 1.0
                key = (s, d)
                if key not in route:
                    route[key] = _relays(nh, s, d)
                for r in route[key]:
                    out[r, 2 * b + 1] += 1.0
        beyond = (k + hi) - max(k + lo, horizon)  # cycles of the bin past the episode end
        if beyond > 0 and rate > 0:
            for s, d in flows:
                out[s, 2 * b] += rate * beyond
                key = (int(s), int(d))
                if key not in route:
                    route[key] = _relays(nh, int(s), int(d))
                for r in route[key]:
                    out[r, 2 * b + 1] += rate * beyond
        out[:, 2 * b: 2 * b + 2] /= float(hi - lo)
        lo = hi
    return out
