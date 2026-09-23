"""TD / GAE with time-varying discount (research-method §5).

``delta_n = r_n + Gamma_n * B_n * V_old(next_n) - V_old(n)``
``A_n     = delta_n + Gamma_n * lambda * C_n * A_{n+1}``

Micro transitions: r = 0, Gamma = 1, B = C = 1, next = following micro state.
Boundary transition (after the last micro state of a cycle): r = cycle reward,
Gamma = gamma; next = first micro state of the next cycle.  Truncation / slice
cut: B = 1 with the stored bootstrap value, C = 0.  Termination: B = C = 0.
lambda acts per micro transition; the physical discount only at boundaries.
"""

from __future__ import annotations

import numpy as np

from .records import CONTINUE, CUT, TERMINATED, TRUNCATED, CycleRecord


def gae(rewards, values, next_values, gammas, bootstrap, cont, lam: float) -> np.ndarray:
    rewards, values, next_values = map(np.asarray, (rewards, values, next_values))
    gammas, bootstrap, cont = map(np.asarray, (gammas, bootstrap, cont))
    delta = rewards + gammas * bootstrap * next_values - values
    adv = np.zeros(len(rewards), dtype=np.float64)
    running = 0.0
    for n in range(len(rewards) - 1, -1, -1):
        running = delta[n] + gammas[n] * lam * cont[n] * running
        adv[n] = running
    return adv


def flatten_stream(records: list[CycleRecord], gamma: float):
    """Per-transition arrays for one environment's records in time order."""
    rewards, values, next_values, gammas, boot, cont = [], [], [], [], [], []
    for i, rec in enumerate(records):
        v = rec.micro.values.astype(np.float64)
        k = rec.num_actions
        for j in range(k):  # micro transitions
            rewards.append(0.0)
            values.append(v[j])
            next_values.append(v[j + 1])
            gammas.append(1.0)
            boot.append(1.0)
            cont.append(1.0)
        rewards.append(rec.reward)
        values.append(v[k])
        gammas.append(gamma)
        if rec.end == CONTINUE:
            if i + 1 >= len(records):
                raise ValueError("stream ends with a 'continue' record; mark it 'cut'")
            next_values.append(float(records[i + 1].micro.values[0]))
            boot.append(1.0)
            cont.append(1.0)
        elif rec.end in (TRUNCATED, CUT):
            next_values.append(rec.bootstrap)
            boot.append(1.0)
            cont.append(0.0)
        elif rec.end == TERMINATED:
            next_values.append(0.0)
            boot.append(0.0)
            cont.append(0.0)
        else:
            raise ValueError(f"unknown end type {rec.end!r}")
    return rewards, values, next_values, gammas, boot, cont


def compute_stream_advantages(records: list[CycleRecord], gamma: float, lam: float) -> None:
    """Fill ``advantages`` and ``returns`` (= A + V_old) of every record in place."""
    if not records:
        return
    r, v, nv, g, b, c = flatten_stream(records, gamma)
    adv = gae(r, v, nv, g, b, c, lam)
    ret = adv + np.asarray(v)
    pos = 0
    for rec in records:
        n = rec.num_actions + 1
        rec.advantages = adv[pos:pos + n].astype(np.float32)
        rec.returns = ret[pos:pos + n].astype(np.float32)
        pos += n
