"""Train, then select checkpoints for, the runs listed in training specs (E13b onwards).

A spec (configs/experiments/*_train.json) crosses methods ("kinds", each with its own
training command, seeds and a rough relative cost) with training scenarios:

    {"name": "...",
     "runs": "results/e14/runs/{scenario}-{kind}-s{seed}",
     "scenarios": {"broad": {"scenario": "configs/explore/e14_broad.toml", "set": []}},
     "scenarios_from": "configs/experiments/e13_flock.json",   # alternative: an eval spec's scenarios
     "kinds": {"main": {"cmd": ["train", "--config", "configs/protocol_final.toml"],
                        "seeds": [0, 1, 2, 3, 4], "cost": 2.0}, ...}}

Every run is trained from scratch with the method's own recipe; the scenario section
is replaced by the scenario file (`--scenario`) and the overrides follow.  Checkpoints
are then chosen on the scenario's dev split by the fixed rule (`fanet-next select`).
All specs given share one job queue, longest first; rerunning skips finished steps.

    .venv/bin/python tools/train_runs.py configs/experiments/e13b_train.json \
        configs/experiments/e14_train.json --parallel 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import e10_ns3 as base
from confirm import CLI, ROOT


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
                out.append({"dir": run, "cost": float(k.get("cost", 1.0)),
                            "train": [CLI, *k["cmd"], *scen_args, *sets, "--run-dir", str(run)]})
    return out


def jobs(specs: list[dict]) -> tuple[list[list[str]], list[list[str]]]:
    train, select = [], []
    for r in sorted((r for s in specs for r in runs(s)), key=lambda r: -r["cost"]):
        if not (r["dir"] / "checkpoints" / "latest.pt").exists():
            train.append(r["train"] + ["--out", str(r["dir"]) + ".train.json"])
        if not (r["dir"] / "selection.json").exists():
            select.append([CLI, "select", "--run-dir", str(r["dir"]), "--out", str(r["dir"]) + ".select.json"])
    return train, select


def _wrap(cmd: list[str]) -> list[str]:
    """Shell wrapper so the launcher's '--out <file>' names the log while the CLI never sees it."""
    out = cmd[cmd.index("--out") + 1]
    real = cmd[:cmd.index("--out")]
    quoted = " ".join("'" + c.replace("'", "'\\''") + "'" for c in real)
    return ["bash", "-c", f"exec {quoted}", "--out", out]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("specs", nargs="+")
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--log", default=None, help="driver log (default: next to the first spec's runs)")
    a = p.parse_args()
    specs = [json.loads((ROOT / s).read_text()) for s in a.specs]
    for s in specs:
        for r in runs(s):
            r["dir"].parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(a.log) if a.log else runs(specs[0])[0]["dir"].parent.parent / "train.log"

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    train, _ = jobs(specs)
    log(f"{len(train)} training jobs ({', '.join(s['name'] for s in specs)})")
    base.launch([_wrap(c) for c in train], a.parallel, 5.0, log)
    _, select = jobs(specs)
    log(f"{len(select)} selection jobs")
    base.launch([_wrap(c) for c in select], a.parallel, 2.0, log)
    log("done")


if __name__ == "__main__":
    main()
