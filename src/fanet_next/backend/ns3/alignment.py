"""Lockstep semantic alignment: lightweight backend vs ns-3 bridge on identical inputs.

Both environments are built from one config (same scenario seed, routing,
observation, reward); the policy decides from the lightweight input and the
same ordered plan is executed by both backends.  Every report and every cycle's
facts are compared field by field.  Times coming from ns-3 are integer
nanoseconds, the lightweight backend uses float seconds, so time-valued fields
use an absolute tolerance.
"""

from __future__ import annotations

import numpy as np

from ...config import deep_merge
from ...experiment.assemble import build_env, build_policy

TIME_TOL = 2e-9  # seconds; birth times are rounded to integer ns on the ns-3 side


def _cmp(prefix, a, b, out, tol=None):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        out.append(f"{prefix}: shape {a.shape} != {b.shape}")
    elif tol is None and not np.array_equal(a, b):
        idx = np.argwhere(a != b)[:3].tolist()
        out.append(f"{prefix}: differ at {idx}")
    elif tol is not None and a.size and np.max(np.abs(a - b)) > tol:
        out.append(f"{prefix}: max |diff| {np.max(np.abs(a - b)):.3g} > {tol}")


def compare_reports(a, b, where: str) -> list[str]:
    out: list[str] = []
    _cmp(f"{where} queue packets", a.queues.packets, b.queues.packets, out)
    _cmp(f"{where} queue bytes", a.queues.bytes, b.queues.bytes, out)
    _cmp(f"{where} queue capacity", a.queues.capacity, b.queues.capacity, out)
    _cmp(f"{where} hol", a.queues.hol_wait, b.queues.hol_wait, out, TIME_TOL)
    _cmp(f"{where} waiting packets", a.waiting_packets, b.waiting_packets, out)
    _cmp(f"{where} waiting oldest", a.waiting_oldest, b.waiting_oldest, out, TIME_TOL)
    _cmp(f"{where} next_hop", a.next_hop, b.next_hop, out)
    _cmp(f"{where} route_next", a.route_next, b.route_next, out)
    _cmp(f"{where} gain", a.gain, b.gain, out)
    return out


def compare_facts(a, b, where: str) -> list[str]:
    out: list[str] = []
    for f in ("planned", "succeeded", "failed"):
        if tuple(getattr(a, f)) != tuple(getattr(b, f)):
            out.append(f"{where} {f}: {getattr(a, f)} != {getattr(b, f)}")
    _cmp(f"{where} served packets", a.served_packets, b.served_packets, out)
    _cmp(f"{where} served bytes", a.served_bytes, b.served_bytes, out)
    _cmp(f"{where} post packets", a.post_service.packets, b.post_service.packets, out)
    _cmp(f"{where} post hol", a.post_service.hol_wait, b.post_service.hol_wait, out, TIME_TOL)
    for f in ("births", "risk_packets", "queued_end", "waiting_end", "waiting_post_service",
              "moved_to_waiting", "restored_from_waiting", "relay_terminated", "delivered_bytes"):
        if getattr(a, f) != getattr(b, f):
            out.append(f"{where} {f}: {getattr(a, f)} != {getattr(b, f)}")
    da = dict(zip(a.delivered_ids, a.delivered_delays))
    db = dict(zip(b.delivered_ids, b.delivered_delays))
    if set(da) != set(db):
        out.append(f"{where} delivered ids differ: {sorted(set(da) ^ set(db))[:5]}")
    else:
        _cmp(f"{where} e2e delay", [da[i] for i in sorted(da)], [db[i] for i in sorted(db)], out,
             TIME_TOL)
    ta = {k: sorted(v) for k, v in a.terminations.items() if v}
    tb = {k: sorted(v) for k, v in b.terminations.items() if v}
    if ta != tb:
        out.append(f"{where} terminations: {ta} != {tb}")
    return out


def align_episode(cfg: dict, seed: int, policy: str | dict = "longest_queue", *,
                  horizon: int | None = None, model_state=None, max_report: int = 20) -> dict:
    """Run one episode on both backends in lockstep; return counts and the first mismatches."""
    backend = cfg.get("backend", {})
    common = {k: backend[k] for k in ("routing",) if k in backend}
    light = deep_merge(cfg, {"backend": {"type": "lightweight", "channel": "ideal",
                                         "stale_queue_policy": "keep", "waiting_restore": "drop",
                                         **common}})
    ns3 = deep_merge(cfg, {"backend": {"type": "ns3", **common}})
    if horizon is not None:
        light = deep_merge(light, {"scenario": {"horizon": horizon}})
        ns3 = deep_merge(ns3, {"scenario": {"horizon": horizon}})
    pol = build_policy(light, policy)
    if model_state is not None:
        pol.model.load_state_dict(model_state)
    env_a = build_env(light, run_id="align-lw", build_graph=pol.needs_graph)
    env_b = build_env(ns3, run_id="align-ns3", build_graph=pol.needs_graph)
    mism: list[str] = []
    try:
        inp_a, inp_b = env_a.reset(seed), env_b.reset(seed)
        mism += compare_reports(inp_a.report, inp_b.report, "boundary 0")
        cycles = served = delivered = terminated = 0
        reward_diff = 0.0
        while True:
            if not np.array_equal(inp_a.problem.links, inp_b.problem.links):
                mism.append(f"cycle {cycles}: candidate sets differ")
                break
            actions = pol.act([inp_a], mode="greedy")[0].actions
            ta, tb = env_a.step(actions), env_b.step(actions)
            mism += compare_facts(ta.facts, tb.facts, f"cycle {cycles}")
            mism += compare_reports(ta.next_input.report, tb.next_input.report,
                                    f"boundary {cycles + 1}")
            reward_diff = max(reward_diff, abs(ta.reward.total - tb.reward.total))
            cycles += 1
            served += int(ta.facts.served_packets.sum())
            delivered += len(ta.facts.delivered_ids)
            terminated += len(ta.facts.terminated_ids)
            if ta.end != tb.end:
                mism.append(f"cycle {cycles}: end {ta.end} != {tb.end}")
            if ta.end.value != "continue" or len(mism) >= max_report:
                break
            inp_a, inp_b = ta.next_input, tb.next_input
    finally:
        env_b.backend.close()
    return {"seed": seed, "cycles": cycles, "served": served, "delivered": delivered,
            "terminated": terminated, "max_reward_diff": reward_diff,
            "mismatches": mism[:max_report], "aligned": not mism}
