"""E13 retraining on the data-driven scenes, then checkpoint selection on their dev split.

Main method (configs/protocol_final.toml, seeds 0-4) and Zhao-GCN
(configs/baselines/zhao_gcn.toml, seeds 0-2), each trained from scratch with its
original recipe on:

* flock30, one model per fold: trained on two of the three test flights, selected on
  the dev flight (6mps_obstacles), tested on the third (configs/experiments/e13_flock.json);
* BonnMotion RPGM and Gauss-Markov: trained on the train traces, selected on the dev
  traces, tested on the test traces (configs/experiments/e13_bonnmotion.json).

Run dirs: results/e13/runs/{flock|bm}-{scenario}-{main|zhao}-s{seed}.

    .venv/bin/python tools/e13_train.py --parallel 7
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import e10_ns3 as base
from confirm import CLI, ROOT

KINDS = {"main": (["train", "--config", "configs/protocol_final.toml"], range(5)),
         "zhao": (["train-baseline", "--config", "configs/baselines/zhao_gcn.toml"], range(3))}
SPECS = {"flock": "configs/experiments/e13_flock.json", "bm": "configs/experiments/e13_bonnmotion.json"}


def jobs() -> tuple[list[list[str]], list[list[str]]]:
    train, select = [], []
    for prefix, path in SPECS.items():
        spec = json.loads((ROOT / path).read_text())
        for scen, s in spec["scenarios"].items():
            for kind, (cmd, seeds) in KINDS.items():
                for seed in seeds:
                    run = ROOT / f"results/e13/runs/{prefix}-{scen}-{kind}-s{seed}"
                    sets = [a for o in s.get("set", []) + [f"seed={seed}"] for a in ("--set", o)]
                    if not (run / "checkpoints" / "latest.pt").exists():
                        train.append([CLI, *cmd, "--scenario", s["scenario"], *sets, "--run-dir", str(run),
                                      "--out", str(run) + ".train.json"])
                    if not (run / "selection.json").exists():
                        select.append([CLI, "select", "--run-dir", str(run), "--out", str(run) + ".select.json"])
    return train, select


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parallel", type=int, default=7)
    a = p.parse_args()
    log_path = ROOT / "results/e13/train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    train, _ = jobs()
    log(f"{len(train)} training jobs")
    # launch() needs an "--out" for its logs; strip it before the command runs
    base.launch([_wrap(c) for c in train], a.parallel, 5.0, log)
    _, select = jobs()
    log(f"{len(select)} selection jobs")
    base.launch([_wrap(c) for c in select], a.parallel, 2.0, log)
    log("done")


def _wrap(cmd: list[str]) -> list[str]:
    """Shell wrapper so the driver's '--out <file>' names the log while the CLI never sees it."""
    out = cmd[cmd.index("--out") + 1]
    real = cmd[:cmd.index("--out")]
    quoted = " ".join("'" + c.replace("'", "'\\''") + "'" for c in real)
    return ["bash", "-c", f"exec {quoted}", "--out", out]


if __name__ == "__main__":
    main()
