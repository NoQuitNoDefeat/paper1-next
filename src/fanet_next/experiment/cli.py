"""Command line: train / resume / eval / components.

Examples::

    fanet-next train --config configs/base.toml --run-dir results/base-s0
    fanet-next resume --run-dir results/base-s0 --iterations 300
    fanet-next eval --config configs/base.toml --policies longest_queue random --episodes 8
    fanet-next eval --run-dir results/base-s0 --policies ppo longest_queue --split test
    fanet-next components
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..config import load_config
from ..registry import describe
from ..training.checkpoint import load_checkpoint
from .assemble import build_model, build_policy, code_version, feature_schema
from .evaluate import evaluate, paired
from .train import TrainingRun

PAIRED_KEYS = ("delivery_ratio", "termination_ratio", "e2e_delay_mean_s", "e2e_delay_p95_s",
               "ontime_1s", "ontime_2s")
KEYS = ("delivery_ratio", "ontime_1s", "ontime_2s", "e2e_delay_mean_s", "e2e_delay_p95_s",
        "throughput_bps",
        "queue_backlog_mean", "waiting_backlog_mean", "termination_ratio", "plan_size_mean",
        "exec_failed_link_frac", "sinr_violation_cycle_frac", "reward_mean", "decision_ms_mean")


def _cmd_train(args) -> None:
    cfg = load_config(args.config, args.set)
    run_dir = Path(args.run_dir or f"results/{cfg.get('name', 'run')}-s{cfg.get('seed', 0)}-"
                                   f"{time.strftime('%Y%m%d-%H%M%S')}")
    if (run_dir / "checkpoints" / "latest.pt").exists():
        raise SystemExit(f"{run_dir} already has a checkpoint; use `resume`")
    print(f"run dir: {run_dir}")
    TrainingRun(cfg, run_dir).train(args.iterations)


def _cmd_resume(args) -> None:
    overrides = {"iterations": args.iterations} if args.iterations else None
    run = TrainingRun.resume(args.run_dir, args.checkpoint, fresh_envs=args.fresh_envs,
                             overrides=overrides)
    print(f"resumed at iteration {run.progress['iteration']}")
    run.train(args.iterations)


def _cmd_eval(args) -> None:
    from ..config import apply_override
    from .assemble import switch_backend
    if args.run_dir:
        ckpt = load_checkpoint(args.checkpoint or Path(args.run_dir) / "checkpoints" / "latest.pt")
        cfg = ckpt["config"]
    else:
        ckpt = None
        cfg = load_config(args.config)
    if args.backend:
        cfg = switch_backend(cfg, args.backend)
    for item in args.set or []:
        apply_override(cfg, item)
    table = {}
    for name in args.policies:
        if name == "ppo":
            if ckpt is None:
                raise SystemExit("evaluating 'ppo' needs --run-dir (a trained checkpoint)")
            model = build_model(cfg, feature_schema(cfg))
            model.load_state_dict(ckpt["model"])
            policy = build_policy(cfg, "ppo", model=model)
        else:
            policy = build_policy(cfg, name)
        def progress(row, done, total, name=name, t0=time.time()):
            print(f"episode {done}/{total} {name} seed={row['seed']} "
                  f"delivery={row['delivery_ratio']:.4f} delay={row['e2e_delay_mean_s']:.3f} "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)

        res = evaluate(cfg, policy, split=args.split, episodes=args.episodes, mode=args.mode,
                       num_envs=args.num_envs, drain_cycles=args.drain,
                       seed_offset=args.seed_offset, on_episode=progress)
        table[name] = res
        m = res["mean"]
        print(f"{name:>16}: " + "  ".join(f"{k.replace('_mean', '')}={m[k]:.4g}" for k in KEYS if k in m),
              flush=True)
    comparisons = {}
    ref = args.reference or next((n for n in args.policies if n != "ppo"), None)
    if ref in table and len(table) > 1:
        print(f"paired differences vs {ref} (mean [95% CI], wins/n):")
        for name, res in table.items():
            if name == ref:
                continue
            comparisons[name] = {k: paired(res["rows"], table[ref]["rows"], k) for k in PAIRED_KEYS}
            print(f"{name:>16}: " + "  ".join(
                f"{k}={c['mean']:+.4g} [{c['ci95'][0]:+.3g},{c['ci95'][1]:+.3g}] {c['wins']}/{c['n']}"
                for k, c in comparisons[name].items() if "ci95" in c))
    if args.out:
        out = {"config": cfg, "comparisons": comparisons, "reference": ref, "code": code_version(), "results": table,
               "checkpoint": str(args.checkpoint or args.run_dir or "")}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
        print(f"wrote {args.out}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="fanet-next")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--config", required=True)
    t.add_argument("--set", action="append", default=[], help="override, e.g. training.lr=1e-4")
    t.add_argument("--run-dir")
    t.add_argument("--iterations", type=int)
    r = sub.add_parser("resume")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--checkpoint")
    r.add_argument("--iterations", type=int, help="new total iteration count")
    r.add_argument("--fresh-envs", action="store_true", help="restart episodes instead of restoring")
    e = sub.add_parser("eval")
    e.add_argument("--config")
    e.add_argument("--run-dir")
    e.add_argument("--checkpoint")
    e.add_argument("--set", action="append", default=[])
    e.add_argument("--backend", help="execute on this backend with the same environment "
                   "parameters (e.g. ns3), applied before --set")
    e.add_argument("--policies", nargs="+", default=["longest_queue", "random"])
    e.add_argument("--split", default="dev", choices=["dev", "test"])
    e.add_argument("--episodes", type=int, default=8)
    e.add_argument("--mode", default="greedy", choices=["greedy", "sample"])
    e.add_argument("--num-envs", type=int, default=8)
    e.add_argument("--drain", type=int, default=0,
                   help="stop traffic at the horizon and run this many extra cycles")
    e.add_argument("--reference", help="policy for paired differences (default: first baseline)")
    e.add_argument("--seed-offset", type=int, default=0,
                   help="skip the first N seeds of the split (e.g. those used for selection)")
    e.add_argument("--out")
    sel = sub.add_parser("select", help="pick a checkpoint by the fixed reliability-first rule")
    sel.add_argument("--run-dir", required=True)
    sel.add_argument("--reference", default="longest_queue")
    sel.add_argument("--episodes", type=int, default=16)
    sel.add_argument("--drain", type=int, default=250)
    sel.add_argument("--delta", type=float, default=0.002)
    sel.add_argument("--num-envs", type=int, default=8)
    st = sub.add_parser("status", help="read-only dashboard of training runs")
    st.add_argument("--root", default="results")
    st.add_argument("--watch", type=float, default=0.0, help="refresh every N seconds")
    sub.add_parser("components")
    for sp in (e, sel):
        sp.add_argument("--threads", type=int, default=4, help="torch CPU threads")
    args = p.parse_args(argv)
    torch.set_num_threads(getattr(args, "threads", 4))
    if args.cmd == "train":
        _cmd_train(args)
    elif args.cmd == "resume":
        _cmd_resume(args)
    elif args.cmd == "status":
        from .status import main as status_main
        status_main(args.root, args.watch)
    elif args.cmd == "select":
        from .select import select_checkpoint
        select_checkpoint(args.run_dir, reference=args.reference, episodes=args.episodes,
                          drain=args.drain, delta=args.delta, num_envs=args.num_envs)
    elif args.cmd == "eval":
        if not (args.config or args.run_dir):
            raise SystemExit("eval needs --config or --run-dir")
        _cmd_eval(args)
    else:
        from . import assemble  # noqa: F401  (registers everything)
        print(describe())


if __name__ == "__main__":
    main()
