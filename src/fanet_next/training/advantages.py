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


def cycle_advantages(records: list[CycleRecord], gamma: float, lam: float) -> np.ndarray:
    """GAE over cycles with the cycle-start values only (one advantage per cycle):
    ``delta_t = r_t + gamma * B_t * V(start of next cycle) - V(start of cycle t)``."""
    v0 = np.array([float(rec.micro.values[0]) for rec in records])
    adv = np.zeros(len(records))
    running = 0.0
    for i in range(len(records) - 1, -1, -1):
        rec = records[i]
        if rec.end == CONTINUE:
            if i + 1 >= len(records):
                raise ValueError("stream ends with a 'continue' record; mark it 'cut'")
            nxt, boot, cont = v0[i + 1], 1.0, 1.0
        elif rec.end in (TRUNCATED, CUT):
            nxt, boot, cont = rec.bootstrap, 1.0, 0.0
        elif rec.end == TERMINATED:
            nxt, boot, cont = 0.0, 0.0, 0.0
        else:
            raise ValueError(f"unknown end type {rec.end!r}")
        running = rec.reward + gamma * boot * nxt - v0[i] + gamma * lam * cont * running
        adv[i] = running
    return adv


def compute_stream_advantages(records: list[CycleRecord], gamma: float, lam: float,
                              credit: str = "step") -> None:
    """Fill ``advantages`` and ``returns`` (= A + V_old) of every record in place.

    ``credit = "step"``: micro-step GAE (the method's default).  ``credit = "cycle"``: every
    micro action of a cycle gets the cycle-level advantage (``cycle_advantages``), so the
    critic's within-cycle value changes do not enter the policy update (E17 arm A); the
    value targets (``returns``) stay the micro-step lambda-returns either way."""
    if credit not in ("step", "cycle"):
        raise ValueError(f"unknown credit {credit!r}")
    if not records:
        return
    r, v, nv, g, b, c = flatten_stream(records, gamma)
    adv = gae(r, v, nv, g, b, c, lam)
    ret = adv + np.asarray(v)
    per_cycle = cycle_advantages(records, gamma, lam) if credit == "cycle" else None
    pos = 0
    for i, rec in enumerate(records):
        n = rec.num_actions + 1
        rec.advantages = (adv[pos:pos + n] if per_cycle is None else np.full(n, per_cycle[i])).astype(np.float32)
        rec.returns = ret[pos:pos + n].astype(np.float32)
        pos += n
