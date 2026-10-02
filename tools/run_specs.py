"""Evaluate several comparison specs from one job queue (longest jobs first), then write
each spec's summary as tools/e11_compare.py does.  One queue keeps every core busy until
the end, where one driver per spec would leave cores idle while its last long jobs run.

    .venv/bin/python tools/run_specs.py configs/experiments/e14_synthetic.json \
        configs/experiments/e14_flock.json --parallel 9
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import e10_ns3 as base
from confirm import ROOT, resolve
from e11_compare import summarise

WEIGHT = {"grlinq": 4.0, "max_weight_opt": 2.0, "backpressure_opt": 2.0, "lq_local_search": 1.5}


def cost(cmd: list[str], spec: dict) -> float:
    """Rough relative run time: episodes x policy weight x 3 for scenes with up to 30 nodes."""
    out = Path(cmd[cmd.index("--out") + 1])
    s = spec["scenarios"][out.parts[-3]]
    sets = s if isinstance(s, list) else s.get("set", [])
    big = any("num_nodes=30" in x or "num_nodes=[16, 30]" in x for x in sets)
    w = next((v for k, v in WEIGHT.items() if k in out.name), 1.0)
    return int(cmd[cmd.index("--episodes") + 1]) * w * (3.0 if big else 1.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("specs", nargs="+")
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--gap", type=float, default=3.0)
    p.add_argument("--log", default="results/run_specs.log")
    a = p.parse_args()
    specs = [json.loads((ROOT / s).read_text()) for s in a.specs]

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with (ROOT / a.log).open("a") as fh:
            fh.write(line + "\n")

    jobs = sorted(((cost(j, s), j) for s in specs for j in base.build_jobs(s)), key=lambda x: -x[0])
    log(f"{len(jobs)} evaluation jobs ({', '.join(s['name'] for s in specs)})")
    base.launch([resolve(j) for _, j in jobs], a.parallel, a.gap, log)
    for s in specs:
        out = ROOT / s["out"]
        (out / "summary.md").write_text(summarise(s) + "\n")
        log(f"wrote {out / 'summary.md'} ({len(base.build_jobs(s))} jobs still missing)")


if __name__ == "__main__":
    main()
