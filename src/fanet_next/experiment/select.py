"""Checkpoint selection with a rule fixed before looking at results.

Reliability first: a checkpoint is eligible when its final (drained) delivery
ratio on the selection episodes is at most ``delta`` below the reference
heuristic (paired mean over the same seeds).  Among eligible checkpoints the
lowest mean end-to-end delay wins (ties: lower p95).  If none is eligible the
highest delivery ratio wins.  Confirmation must use other seeds
(``eval --seed-offset``), never the selection episodes.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..training.checkpoint import load_checkpoint
from .assemble import build_model, build_policy, code_version, feature_schema
from .evaluate import evaluate, paired

RULE = ("eligible: paired mean delivery_ratio - reference >= -delta; "
        "pick min e2e_delay_mean_s (tie: e2e_delay_p95_s); none eligible: max delivery_ratio")


def candidates(run_dir: Path) -> list[Path]:
    ckpts = sorted((run_dir / "checkpoints").glob("iter_*.pt"))
    latest = run_dir / "checkpoints" / "latest.pt"
    if latest.exists():
        it = load_checkpoint(latest)["progress"]["iteration"]
        if not any(p.name == f"iter_{it:05d}.pt" for p in ckpts):
            ckpts.append(latest)
    return ckpts


def select_checkpoint(run_dir: str | Path, *, reference: str = "longest_queue", episodes: int = 16,
                      drain: int = 250, delta: float = 0.002, num_envs: int = 8) -> dict:
    run_dir = Path(run_dir)
    cands = candidates(run_dir)
    if not cands:
        raise SystemExit(f"no checkpoints in {run_dir}")
    cfg = load_checkpoint(cands[0])["config"]
    ref = evaluate(cfg, build_policy(cfg, reference), split="dev", episodes=episodes,
                   drain_cycles=drain, num_envs=num_envs)
    rows = []
    for path in cands:
        ck = load_checkpoint(path)
        model = build_model(cfg, feature_schema(cfg))
        model.load_state_dict(ck["model"])
        res = evaluate(cfg, build_policy(cfg, "ppo", model=model), split="dev", episodes=episodes,
                       drain_cycles=drain, num_envs=num_envs)
        d = paired(res["rows"], ref["rows"], "delivery_ratio")
        m = res["mean"]
        rows.append({"checkpoint": str(path), "iteration": ck["progress"]["iteration"],
                     "delivery_ratio": m["delivery_ratio"], "delivery_diff": d,
                     "e2e_delay_mean_s": m["e2e_delay_mean_s"], "e2e_delay_p95_s": m["e2e_delay_p95_s"],
                     "ontime_1s": m.get("ontime_1s"), "ontime_2s": m.get("ontime_2s"),
                     "termination_ratio": m["termination_ratio"],
                     "eligible": d["mean"] >= -delta})
        print(f"  iter {rows[-1]['iteration']:4d}: dr={m['delivery_ratio']:.4f} "
              f"(diff {d['mean']:+.4f}) delay={m['e2e_delay_mean_s']:.3f}s "
              f"p95={m['e2e_delay_p95_s']:.3f}s eligible={rows[-1]['eligible']}", flush=True)
    eligible = [r for r in rows if r["eligible"]]
    if eligible:
        best = min(eligible, key=lambda r: (r["e2e_delay_mean_s"], r["e2e_delay_p95_s"]))
    else:
        best = max(rows, key=lambda r: r["delivery_ratio"])
    out = {"rule": RULE, "delta": delta, "reference": reference, "episodes": episodes,
           "drain": drain, "seeds": ref["seeds"], "reference_mean": ref["mean"],
           "candidates": rows, "selected": best, "any_eligible": bool(eligible),
           "code": code_version()}
    (run_dir / "selection.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print(f"selected iteration {best['iteration']} ({'eligible' if eligible else 'none eligible'})")
    return out
