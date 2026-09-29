"""Training runs of learned baselines (E11): same budget and bookkeeping as the primary method.

Config: any protocol config plus ``[baseline]`` with ``policy`` (a registered learned
baseline spec) and optional ``learner`` settings.  The run uses the primary method's
``[training]`` budget (iterations x num_envs x rollout_cycles environment cycles),
the same training seed stream, dev evaluation and kept checkpoints, so
``fanet-next select`` applies the identical selection rule.  (No exact resume: a
baseline run restarts from scratch.)
"""

from __future__ import annotations

import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from ..baselines import make_learner
from ..config import dump_json
from ..reward.metrics import merge_summaries
from ..training.checkpoint import save_checkpoint
from .assemble import TRAIN_SEED_BASE, build_env, build_policy, check_compatibility, code_version
from .evaluate import evaluate
from .train import TRAINING_DEFAULTS, seed_everything


class BaselineRun:
    def __init__(self, cfg: dict, run_dir: str | Path):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.tcfg = {**TRAINING_DEFAULTS, **cfg.get("training", {})}
        t = self.tcfg
        torch.set_num_threads(int(t["threads"]))
        self.seed = int(cfg.get("seed", 0))
        seed_everything(self.seed)
        self.spec = dict(cfg["baseline"]["policy"])
        self.policy = build_policy(cfg, self.spec, seed=self.seed + 100_000)
        self.envs = [build_env(cfg, run_id=f"train-env{e}", build_graph=self.policy.needs_graph)
                     for e in range(int(t["num_envs"]))]
        self.manifest = check_compatibility(self.envs[0], self.policy)
        self.seed_base = TRAIN_SEED_BASE + 100_000 * self.seed
        self.learner = make_learner(self.spec["type"], self.policy, self.envs, cfg, self.seed_base,
                                    self.seed + 200_000)
        self.progress = {"iteration": 0, "cycles": 0, "episodes": 0, "wall_seconds": 0.0}
        self.code = code_version()

    def _log(self, name: str, entry: dict) -> None:
        with (self.run_dir / name).open("a") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=float) + "\n")

    def _write_meta(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        dump_json(self.cfg, self.run_dir / "config.json")
        snapshot = self.run_dir / "code_snapshot"
        if not snapshot.exists():
            root = Path(__file__).resolve().parents[3]
            for sub in ("src", "configs"):
                shutil.copytree(root / sub, snapshot / sub,
                                ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        params = sum(p.numel() for p in self.policy.net.parameters())
        meta = {"manifest": {**self.manifest, "model_parameters": int(params)}, "code": self.code,
                "seed": self.seed, "policy_spec": self.spec, "python": platform.python_version(),
                "torch": torch.__version__, "numpy": np.__version__,
                "started": time.strftime("%Y-%m-%d %H:%M:%S"), "train_seed_base": self.seed_base,
                "training": self.tcfg, "learner": self.cfg["baseline"].get("learner", {})}
        (self.run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    def save(self, path: Path | None = None) -> Path:
        path = path or self.run_dir / "checkpoints" / "latest.pt"
        save_checkpoint(path, {"config": self.cfg, "policy_spec": self.spec,
                               "model": self.policy.state_dict(), "learner": self.learner.state_dict(),
                               "progress": dict(self.progress), "code": self.code,
                               "manifest": self.manifest})
        return path

    def evaluate_dev(self) -> dict:
        t = self.tcfg
        result = evaluate(self.cfg, self.policy, split="dev", episodes=int(t["eval_episodes"]),
                          mode="greedy", num_envs=int(t["num_envs"]),
                          drain_cycles=int(t["eval_drain_cycles"]))
        entry = {"iteration": self.progress["iteration"],
                 **{f"dev/{k}": v for k, v in result["mean"].items()}}
        self._log("eval_log.jsonl", entry)
        print(f"   dev: dr={result['mean']['delivery_ratio']:.3f} "
              f"delay={result['mean']['e2e_delay_mean_s']:.3f}s "
              f"plan={result['mean']['plan_size_mean']:.2f}", flush=True)
        return result

    def train(self, iterations: int | None = None) -> None:
        t = self.tcfg
        total = int(iterations if iterations is not None else t["iterations"])
        self._write_meta()
        setup = self.learner.setup(self)
        if setup:
            self._log("train_log.jsonl", setup)
            print(f"[setup] {setup}", flush=True)
        while self.progress["iteration"] < total:
            it = self.progress["iteration"]
            t0 = time.perf_counter()
            stats = self.learner.iteration(it, total)
            episodes = stats.pop("episodes", [])
            self.progress["iteration"] = it + 1
            self.progress["cycles"] += int(stats.pop("cycles"))
            self.progress["episodes"] += len(episodes)
            self.progress["wall_seconds"] += time.perf_counter() - t0
            entry = {"iteration": it + 1, **stats, "seconds": time.perf_counter() - t0,
                     "progress": dict(self.progress)}
            if episodes:
                ep = merge_summaries(episodes)
                entry.update({f"train/{k}": ep[k] for k in (
                    "delivery_ratio", "e2e_delay_mean_s", "plan_size_mean", "termination_ratio")
                    if k in ep})
            self._log("train_log.jsonl", entry)
            print(f"[it {it + 1:4d}] " + " ".join(
                f"{k}={v:.4g}" for k, v in entry.items()
                if isinstance(v, float) and k not in ("seconds",)) + f" t={entry['seconds']:.1f}s",
                flush=True)
            if t["eval_every"] and (it + 1) % t["eval_every"] == 0:
                self.evaluate_dev()
            if t["checkpoint_every"] and (it + 1) % t["checkpoint_every"] == 0:
                self.save()
                if t["keep_every"] and (it + 1) % t["keep_every"] == 0:
                    self.save(self.run_dir / "checkpoints" / f"iter_{it + 1:05d}.pt")
        self.save()
