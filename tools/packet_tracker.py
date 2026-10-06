"""Per-packet bookkeeping for stage-1 diagnostics (X0 loss ledger, X2 hindsight probe, S1e).

Wrap every ``env.step`` with ``before(env)`` / ``after(env, tr)``.  For each packet id it keeps the
birth cycle, the route length at birth, successful transmissions, the remaining time at its first
departure from its source and at each arrival at a relay, the cycle it first became hopeless under
the current routes (it cannot make the deadline even if it moves one hop at the start of every
remaining cycle: remaining < (hops - 1) * T + one packet airtime), and its fate (delivered with delay, or the
termination reason and where it was: source, relay, waiting area, late at the destination).

The steady-state window counts packets born in cycles [W, stop - D/T) with W = max(100, D/T)
(docs/next-method.md section 3.3): they face a loaded network for their whole life, and their
deadline falls before traffic stops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PacketLog:
    born_cycle: int
    hops0: int
    tx: int = 0
    depart_remaining: float | None = None  # remaining time when it first left its source
    arrive_remaining: list = field(default_factory=list)  # remaining time at each relay arrival
    last_tx_time: float | None = None
    last_tx_remaining: float | None = None  # remaining time when it last moved one hop
    hopeless_cycle: int | None = None
    tx_after_hopeless: int = 0
    fate: str | None = None  # "delivered" or "<reason>@<place>"
    delay: float | None = None


def steady_window(deadline: float, cycle: float, stop: int) -> tuple[int, int]:
    d_cycles = int(math.ceil(deadline / cycle))
    return max(100, d_cycles), stop - d_cycles


class PacketTracker:
    def __init__(self, deadline: float | None):
        self.deadline = deadline
        self.log: dict[int, PacketLog] = {}
        self.total_tx = 0

    # ------------------------------------------------------------------ per cycle
    def before(self, env) -> None:
        be = self.be = env.backend
        sc = be.sc
        self.cycle = be.cycle
        t = be.cycle * sc.cycle_length
        self.np0 = be.next_pid
        self.pre = {p.pid: (p, p.node) for q in be.queues for p in q}
        self.pre_wait = {p.pid for w in be.waiting for p in w}
        if self.deadline is None:
            return
        air = sc.packet_size * 8.0 / sc.radio.rate_bps
        for pid, (p, node) in self.pre.items():
            lg = self.log.get(pid)
            if lg is None or lg.hopeless_cycle is not None:
                continue
            hops = 0 if node == p.dst else be._route_len(node, p.dst)
            if hops > 0 and p.born + self.deadline - t < (hops - 1) * sc.cycle_length + air - 1e-12:
                lg.hopeless_cycle = be.cycle

    def after(self, env, tr) -> None:
        be, f = self.be, tr.facts
        for i, pid in enumerate(range(self.np0, be.next_pid)):
            self.log[pid] = PacketLog(born_cycle=self.cycle, hops0=int(f.born_hops0[i]) if i < len(f.born_hops0) else -1)
        delivered = dict(zip(f.delivered_ids, f.delivered_delays))
        for pid, (p, node0) in self.pre.items():
            if pid in delivered or p.node != node0:  # transmitted successfully this cycle
                lg = self.log.get(pid)
                self.total_tx += 1
                if lg is None:
                    continue
                lg.tx += 1
                if lg.hopeless_cycle is not None:
                    lg.tx_after_hopeless += 1
                t_done = p.node_arrived if pid not in delivered else p.born + delivered[pid]
                lg.last_tx_time = t_done
                if self.deadline is not None:
                    rem = self.deadline - (t_done - p.born)
                    lg.last_tx_remaining = rem
                    if node0 == p.src and lg.depart_remaining is None:
                        lg.depart_remaining = rem
                    if pid not in delivered and p.node != p.dst:
                        lg.arrive_remaining.append(rem)
        for pid, dl in delivered.items():
            if pid in self.log:
                self.log[pid].fate, self.log[pid].delay = "delivered", float(dl)
        for reason, ids in f.terminations.items():
            for pid in ids:
                lg = self.log.get(pid)
                if lg is None:
                    continue
                if pid in self.pre_wait:
                    place = "waiting"
                elif pid in self.pre:
                    p = self.pre[pid][0]
                    place = "late" if p.node == p.dst else ("source" if p.node == p.src else "relay")
                else:
                    place = "new" if lg.born_cycle == self.cycle else "moved"  # born or re-homed/arrived this cycle
                lg.fate = f"{reason}@{place}"

    def step(self, env, actions):
        self.before(env)
        tr = env.step(actions)
        self.after(env, tr)
        return tr

    # ------------------------------------------------------------------ summaries
    def summary(self, window: tuple[int, int] | None = None) -> dict:
        logs = [lg for lg in self.log.values() if window is None or window[0] <= lg.born_cycle < window[1]]
        n = max(len(logs), 1)
        fates: dict[str, int] = {}
        for lg in logs:
            fates[lg.fate or "in_system"] = fates.get(lg.fate or "in_system", 0) + 1
        tx = sum(lg.tx for lg in logs)
        lost = [lg for lg in logs if lg.fate not in ("delivered", None)]
        left = [lg for lg in logs if lg.tx > 0]
        left_multi = [lg for lg in left if lg.hops0 >= 2]
        q = lambda xs: [float(np.quantile(xs, p)) for p in (0.1, 0.5, 0.9)] if xs else None  # noqa: E731
        exp_relay = [lg for lg in logs if lg.fate == "deadline@relay"]
        out = {
            "born": len(logs),
            "delivered": fates.get("delivered", 0) / n,
            "fates": {k: v / n for k, v in sorted(fates.items())},
            "tx_per_born": tx / n,
            "tx_wasted_share": sum(lg.tx for lg in lost) / max(tx, 1),
            "tx_after_hopeless_share": sum(lg.tx_after_hopeless for lg in logs) / max(tx, 1),
            "hopeless_share": sum(lg.hopeless_cycle is not None for lg in logs) / n,
            "pipeline_efficiency": sum(lg.fate == "delivered" for lg in left) / max(len(left), 1),
            "pipeline_efficiency_multihop": sum(lg.fate == "delivered" for lg in left_multi) / max(len(left_multi), 1),
            "offered_hop_load_over_tx": sum(max(lg.hops0, 0) for lg in logs) / max(tx, 1),
            "delay_mean": float(np.mean([lg.delay for lg in logs if lg.delay is not None])) if fates.get("delivered") else None,
        }
        if self.deadline is not None:
            multi = [lg for lg in logs if lg.hops0 >= 2]
            out["depart_remaining_q10_50_90_multihop"] = q([lg.depart_remaining for lg in multi if lg.depart_remaining is not None])
            out["arrive_remaining_q10_50_90_by_relay"] = {
                str(h + 1): q([lg.arrive_remaining[h] for lg in multi if len(lg.arrive_remaining) > h]) for h in range(3)}
            out["last_tx_remaining_q10_50_90_expired_at_relay"] = q(
                [lg.last_tx_remaining for lg in exp_relay if lg.last_tx_remaining is not None])
        return out
