"""Training runs: assemble, collect, update, evaluate, log, checkpoint, resume."""

from __future__ import annotations

import json
import platform
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from ..config import dump_json
from ..policy.learned import LearnedPolicy
from ..reward.metrics import merge_summaries
from ..training.advantages import compute_stream_advantages
from ..training.checkpoint import load_checkpoint, rng_state, save_checkpoint, set_rng_state
from ..training.collector import RolloutCollector, SeedStream
from ..training.normalize import ReturnScaler
from ..training.ppo import TRAINER
from .assemble import (TRAIN_SEED_BASE, build_env, build_model, check_compatibility, code_version,
                       feature_schema)
from .evaluate import evaluate

TRAINING_DEFAULTS = dict(
    iterations=200, num_envs=8, rollout_cycles=128, gamma=0.99, gae_lambda=0.95,
    reward_norm=True, lr=3e-4, lr_final=None, eval_every=10, eval_episodes=8,
    checkpoint_every=10, keep_every=50, threads=1, device="cpu", eval_drain_cycles=0)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TrainingRun:
    def __init__(self, cfg: dict, run_dir: str | Path):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.tcfg = {**TRAINING_DEFAULTS, **cfg.get("training", {})}
        t = self.tcfg
        torch.set_num_threads(int(t["threads"]))
        self.seed = int(cfg.get("seed", 0))
        seed_everything(self.seed)
        schema = feature_schema(cfg)
        self.model = build_model(cfg, schema).to(t["device"])
        self.policy = LearnedPolicy(self.model, device=t["device"], seed=self.seed + 100_000)
        self.envs = [build_env(cfg, run_id=f"train-env{e}") for e in range(int(t["num_envs"]))]
        self.manifest = check_compatibility(self.envs[0], self.policy)
        self.scaler = ReturnScaler(len(self.envs), t["gamma"], enabled=bool(t["reward_norm"]))
        self.collector = RolloutCollector(
            self.envs, self.policy, seeds=SeedStream(TRAIN_SEED_BASE + 100_000 * self.seed),
            scaler=self.scaler, rollout_cycles=int(t["rollout_cycles"]))
        trainer_spec = dict(cfg.get("trainer", {"type": "ppo"}))
        trainer_spec.setdefault("type", "ppo")
        trainer_spec.setdefault("lr", t["lr"])
        self.trainer = TRAINER.build(trainer_spec, model=self.model, seed=self.seed + 200_000,
                                     device=t["device"])
        self.progress = {"iteration": 0, "cycles": 0, "actions": 0, "episodes": 0,
                         "wall_seconds": 0.0, "errors": 0}
        self.code = code_version()  # fixed at start; later edits on disk do not change it

    # ------------------------------------------------------------- files
    def _write_meta(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        dump_json(self.cfg, self.run_dir / "config.json")
        snapshot = self.run_dir / "code_snapshot"
        if not snapshot.exists():
            root = Path(__file__).resolve().parents[3]
            for sub in ("src", "configs"):
                shutil.copytree(root / sub, snapshot / sub,
                                ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        meta = {"manifest": self.manifest, "code": self.code, "seed": self.seed,
                "python": platform.python_version(), "torch": torch.__version__,
                "numpy": np.__version__, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "train_seed_base": self.collector.seeds.base, "training": self.tcfg}
        (self.run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    def _log(self, name: str, entry: dict) -> None:
        with (self.run_dir / name).open("a") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ------------------------------------------------------- checkpoints
    def save(self, path: Path | None = None) -> Path:
        path = path or self.run_dir / "checkpoints" / "latest.pt"
        save_checkpoint(path, {
            "config": self.cfg, "model": self.model.state_dict(),
            "trainer": self.trainer.state_dict(), "progress": dict(self.progress),
            "collector": self.collector.state_dict(), "rng": rng_state(self.policy.generator),
            "code": self.code, "manifest": self.manifest})
        return path

    @classmethod
    def resume(cls, run_dir: str | Path, checkpoint: str | Path | None = None,
               fresh_envs: bool = False, overrides: dict | None = None) -> "TrainingRun":
        run_dir = Path(run_dir)
        data = load_checkpoint(checkpoint or run_dir / "checkpoints" / "latest.pt")
        cfg = data["config"]
        if overrides:
            cfg = {**cfg, "training": {**cfg.get("training", {}), **overrides}}
        run = cls(cfg, run_dir)
        run.model.load_state_dict(data["model"])
        run.trainer.load_state_dict(data["trainer"])
        run.progress = dict(data["progress"])
        restore_envs = not fresh_envs and run.envs[0].backend.supports_state
        run.collector.load_state_dict(data["collector"], restore_envs=restore_envs)
        run.envs = run.collector.envs
        set_rng_state(data["rng"], run.policy.generator)
        if data.get("code", {}).get("source_sha256_16") != run.code["source_sha256_16"]:
            print("warning: source code differs from the checkpoint's; resume is not exact")
        run._log("train_log.jsonl", {"event": "resume", "iteration": run.progress["iteration"],
                                     "restored_envs": restore_envs, "code": run.code,
                                     "checkpoint_code": data.get("code")})
        return run

    # ------------------------------------------------------------ train
    def _lr(self, it: int, total: int) -> float:
        t = self.tcfg
        if t["lr_final"] is None:
            return t["lr"]
        frac = min(it / max(total, 1), 1.0)
        return t["lr"] + frac * (t["lr_final"] - t["lr"])

    def train(self, iterations: int | None = None, verbose: bool = True) -> None:
        t = self.tcfg
        total = int(iterations if iterations is not None else t["iterations"])
        if self.progress["iteration"] == 0:
            self._write_meta()
            if self.cfg.get("imitation") and not self.progress.get("imitated"):
                self.imitate()
        while self.progress["iteration"] < total:
            it = self.progress["iteration"]
            t0 = time.perf_counter()
            self.trainer.set_lr(self._lr(it, total))
            res = self.collector.collect()
            for stream in res.streams:
                compute_stream_advantages(stream, t["gamma"], t["gae_lambda"])
            records = res.records
            t1 = time.perf_counter()
            stats = self.trainer.update(records)
            t2 = time.perf_counter()
            self.progress["iteration"] = it + 1
            self.progress["cycles"] += len(records)
            self.progress["actions"] += stats["actions"]
            self.progress["episodes"] += len(res.episodes)
            self.progress["errors"] += len(res.errors)
            self.progress["wall_seconds"] += t2 - t0
            raw = np.array([r.raw_reward for r in records])
            entry = {"iteration": it + 1, **{f"ppo/{k}": v for k, v in stats.items()},
                     "reward_raw_mean": float(raw.mean()), "reward_scale": self.scaler.scale,
                     "actions_per_cycle": stats["actions"] / max(len(records), 1),
                     "collect_s": t1 - t0, "update_s": t2 - t1, "errors": res.errors[:5],
                     "progress": dict(self.progress)}
            if res.episodes:
                ep = merge_summaries(res.episodes)
                entry.update({f"train/{k}": ep[k] for k in (
                    "delivery_ratio", "e2e_delay_mean_s", "queue_backlog_mean", "plan_size_mean",
                    "termination_ratio", "reward_mean") if k in ep})
            self._log("train_log.jsonl", entry)
            if verbose:
                print(f"[it {it + 1:4d}] r={entry['reward_raw_mean']:+.4f} "
                      f"ev={stats['explained_var']:+.2f}/{stats['explained_var_boundary']:+.2f} "
                      f"ent={stats['entropy']:.3f} kl={stats['approx_kl']:.4f} "
                      f"dr={entry.get('train/delivery_ratio', float('nan')):.3f} "
                      f"t={t2 - t0:.1f}s", flush=True)
            if t["eval_every"] and (it + 1) % t["eval_every"] == 0:
                self.evaluate_dev()
            if t["checkpoint_every"] and (it + 1) % t["checkpoint_every"] == 0:
                self.save()
                if t["keep_every"] and (it + 1) % t["keep_every"] == 0:
                    self.save(self.run_dir / "checkpoints" / f"iter_{it + 1:05d}.pt")
        self.save()

    def imitate(self) -> dict:
        """Warm start: fit actor (and critic) to a teacher policy before PPO."""
        from ..training.imitation import fit, teacher_records
        from .assemble import build_policy

        icfg = {"teacher": "longest_queue", "cycles": 500, "epochs": 6, "lr": 1e-3,
                "minibatch": 256, **self.cfg["imitation"]}
        t0 = time.perf_counter()
        teacher = build_policy(self.cfg, icfg["teacher"], seed=self.seed)
        envs = [build_env(self.cfg, run_id=f"imitate-env{e}") for e in range(len(self.envs))]
        base = TRAIN_SEED_BASE + 100_000 * self.seed + 50_000
        records = teacher_records(envs, teacher, [base + e for e in range(len(envs))],
                                  int(icfg["cycles"]), self.tcfg["gamma"])
        boundary = np.array([r.returns[-1] for r in records])
        scale = float(boundary.std()) if self.scaler.enabled and boundary.std() > 0 else 1.0
        stats = fit(self.model, records, epochs=int(icfg["epochs"]), minibatch=int(icfg["minibatch"]),
                    lr=float(icfg["lr"]), reward_scale=scale, seed=self.seed + 300_000,
                    device=self.tcfg["device"])
        if self.scaler.enabled:  # PPO starts with the return scale the critic was fitted to
            self.scaler.rms.var, self.scaler.rms.count = scale ** 2, 1e3
        self.progress["imitated"] = True
        entry = {"event": "imitation", "teacher": icfg["teacher"], "records": len(records),
                 "return_scale": scale, "seconds": time.perf_counter() - t0, **stats}
        self._log("train_log.jsonl", entry)
        print(f"[imitation] teacher={icfg['teacher']} nll={stats['nll']:.3f} "
              f"p>0.5={stats['p_teacher_gt_half']:.3f} v={stats['value_loss']:.3f}", flush=True)
        return entry

    def evaluate_dev(self) -> dict:
        t = self.tcfg
        # evaluation must not disturb training randomness
        state = rng_state(self.policy.generator)
        result = evaluate(self.cfg, self.policy, split="dev", episodes=int(t["eval_episodes"]),
                          mode="greedy", num_envs=int(t["num_envs"]),
                          drain_cycles=int(t["eval_drain_cycles"]))
        set_rng_state(state, self.policy.generator)
        entry = {"iteration": self.progress["iteration"],
                 **{f"dev/{k}": v for k, v in result["mean"].items()}}
        self._log("eval_log.jsonl", entry)
        print(f"   dev: dr={result['mean']['delivery_ratio']:.3f} "
              f"delay={result['mean']['e2e_delay_mean_s']:.3f}s "
              f"r={result['mean']['reward_mean']:+.4f} plan={result['mean']['plan_size_mean']:.2f}",
              flush=True)
        return result
