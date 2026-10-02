"""Train, then select checkpoints for, the runs listed in training specs (E13b onwards).

A spec (configs/experiments/*_train.json) crosses methods ("kinds", each with its own
training command, seeds, a rough relative cost and peak memory) with training scenarios:

    {"name": "...",
     "runs": "results/e14/runs/{scenario}-{kind}-s{seed}",
     "scenarios": {"broad": {"scenario": "configs/explore/e14_broad.toml", "set": []}},
     "scenarios_from": "configs/experiments/e13_flock.json",   # alternative: an eval spec's scenarios
     "kinds": {"main": {"cmd": ["train", "--config", "configs/protocol_final.toml"],
                        "seeds": [0, 1, 2, 3, 4], "cost": 2.0, "mem_gb": 1.5, "priority": 0}, ...}}

Every run is trained from scratch with the method's own recipe; the scenario section
is replaced by the scenario file (`--scenario`) and the overrides follow.  A run whose
checkpoint selection is done is skipped; one that was interrupted is resumed
(`fanet-next resume`); a finished training leaves `<run>/TRAINED`.  Checkpoints are
chosen on the scenario's dev split by the fixed rule (`fanet-next select`) as soon as
a run's training ends.

All specs given share one queue: lower ``priority`` first, then longest first.  A job
starts (the first in queue order that fits) only while at most
``--parallel`` jobs run and their memory estimates stay within ``--mem-budget`` (GB):
oversubscribing the 16 GB machine makes macOS swap and every job crawl.

    .venv/bin/python tools/train_runs.py configs/experiments/e13b_train.json \
        configs/experiments/e14_train.json --parallel 6 --mem-budget 9
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from confirm import CLI, ROOT

SELECT_MEM_GB = 1.0


def runs(spec: dict) -> list[dict]:
    scenarios = spec.get("scenarios")
    if scenarios is None:
        scenarios = json.loads((ROOT / spec["scenarios_from"]).read_text())["scenarios"]
    out = []
    for scen, s in scenarios.items():
        for kind, k in spec["kinds"].items():
            for seed in k["seeds"]:
                run = ROOT / spec["runs"].format(scenario=scen, kind=kind, seed=seed)
                sets = [a for o in s.get("set", []) + [f"seed={seed}"] for a in ("--set", o)]
                scen_args = ["--scenario", s["scenario"]] if s.get("scenario") else []
                iters = k["cmd"][k["cmd"].index("--iterations"):][:2] if "--iterations" in k["cmd"] else []
                out.append({"dir": run, "cost": float(k.get("cost", 1.0)), "mem": float(k.get("mem_gb", 1.0)),
                            "priority": int(k.get("priority", 0)),
                            "train": [CLI, *k["cmd"], *scen_args, *sets, "--run-dir", str(run)],
                            "resume": [CLI, "resume", "--run-dir", str(run), *iters]})
    return out


def trained(run: dict) -> bool:
    return (run["dir"] / "TRAINED").exists()


def selected(run: dict) -> bool:
    return (run["dir"] / "selection.json").exists()


class Queue:
    def __init__(self, parallel: int, mem_budget: float, gap: float, log):
        self.parallel, self.mem_budget, self.gap, self.log = parallel, mem_budget, gap, log
        self.waiting: list[dict] = []  # {"run", "step": "train" | "select", "attempt"}
        self.running: dict[subprocess.Popen, dict] = {}
        self.last = 0.0

    def mem(self, job: dict) -> float:
        return job["run"]["mem"] if job["step"] == "train" else SELECT_MEM_GB

    def command(self, job: dict) -> list[str]:
        run = job["run"]
        if job["step"] == "select":
            return [CLI, "select", "--run-dir", str(run["dir"])]
        return run["resume"] if (run["dir"] / "checkpoints" / "latest.pt").exists() else run["train"]

    def start(self, job: dict) -> None:
        cmd = self.command(job)
        log_file = f"{job['run']['dir']}.{job['step']}.log"
        fh = open(log_file, "a")
        fh.write(f"\n# {time.strftime('%Y-%m-%d %H:%M:%S')} attempt {job['attempt'] + 1}: {' '.join(cmd)}\n")
        fh.flush()
        self.running[subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)] = job
        self.last = time.time()
        self.log(f"started {job['step']} {job['run']['dir'].name} ({cmd[1]}, ~{self.mem(job):g} GB)")

    def reap(self) -> None:
        for proc, job in list(self.running.items()):
            if proc.poll() is None:
                continue
            del self.running[proc]
            name = f"{job['step']} {job['run']['dir'].name}"
            if proc.returncode == 0:
                self.log(f"finished {name}")
                if job["step"] == "train":
                    (job["run"]["dir"] / "TRAINED").touch()
                    self.waiting.append({"run": job["run"], "step": "select", "attempt": 0})
                continue
            self.log(f"FAILED (exit {proc.returncode}, attempt {job['attempt'] + 1}) {name}")
            if job["attempt"] == 0:
                self.waiting.append({**job, "attempt": 1})

    def run(self) -> None:
        while self.waiting or self.running:
            self.reap()
            used = sum(self.mem(j) for j in self.running.values())
            if len(self.running) < self.parallel and time.time() - self.last >= self.gap:
                for i, job in enumerate(self.waiting):
                    if not self.running or used + self.mem(job) <= self.mem_budget:
                        self.start(self.waiting.pop(i))
                        break
            time.sleep(2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("specs", nargs="+")
    p.add_argument("--parallel", type=int, default=6)
    p.add_argument("--mem-budget", type=float, default=9.0, help="GB for all running jobs together")
    p.add_argument("--log", default=None, help="driver log (default: next to the first spec's runs)")
    a = p.parse_args()
    specs = [json.loads((ROOT / s).read_text()) for s in a.specs]
    all_runs = [r for s in specs for r in runs(s)]
    for r in all_runs:
        r["dir"].parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(a.log) if a.log else all_runs[0]["dir"].parent.parent / "train.log"

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    q = Queue(a.parallel, a.mem_budget, 5.0, log)
    for r in sorted(all_runs, key=lambda r: (r["priority"], -r["cost"])):
        if not trained(r):
            q.waiting.append({"run": r, "step": "train", "attempt": 0})
        elif not selected(r):
            q.waiting.append({"run": r, "step": "select", "attempt": 0})
    n_train = sum(j["step"] == "train" for j in q.waiting)
    log(f"{n_train} training and {len(q.waiting) - n_train} selection jobs queued "
        f"({', '.join(s['name'] for s in specs)}); parallel {a.parallel}, memory budget {a.mem_budget:g} GB")
    q.run()
    log("done")


if __name__ == "__main__":
    main()
