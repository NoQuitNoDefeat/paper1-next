"""Run queued `fanet-next train` jobs, keeping at most N training processes alive.

Queue file: one job per line, the arguments of `fanet-next train` (must contain
``--run-dir``); ``#`` starts a comment.  Training processes started elsewhere
count towards the limit.  Each job logs to ``<run-dir>.log``; queue events go
to ``<queue file>.log``.

    .venv/bin/python tools/run_queue.py --queue results/e5/queue.txt --max 3
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / ".venv" / "bin" / "fanet-next"


def running_trains() -> int:
    out = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True).stdout
    return sum(1 for line in out.splitlines()
               if "fanet-next train" in line and "Python" in line.split(" ", 1)[0])


def parse_queue(path: Path) -> list[list[str]]:
    jobs = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            args = shlex.split(line)
            if "--run-dir" not in args:
                raise SystemExit(f"queue line without --run-dir: {line}")
            jobs.append(args)
    return jobs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--queue", required=True)
    p.add_argument("--max", type=int, default=3)
    p.add_argument("--poll", type=float, default=20.0)
    a = p.parse_args()
    queue = Path(a.queue)
    log = queue.with_suffix(queue.suffix + ".log")

    def note(msg: str) -> None:
        with log.open("a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")

    note(f"queue started: {len(parse_queue(queue))} jobs, max {a.max} concurrent")
    children = []
    for args in parse_queue(queue):
        run_dir = Path(args[args.index("--run-dir") + 1])
        if run_dir.exists():
            note(f"SKIP {run_dir} (already exists)")
            continue
        while running_trains() >= a.max:
            time.sleep(a.poll)
        out = open(f"{run_dir}.log", "w")
        children.append(subprocess.Popen([str(CLI), "train", *args], cwd=ROOT, stdout=out,
                                         stderr=subprocess.STDOUT))
        note(f"START {run_dir} pid={children[-1].pid}")
        time.sleep(a.poll)  # let it register before counting again
    for child in children:
        code = child.wait()
        note(f"EXIT pid={child.pid} code={code}")
    note("queue finished")


if __name__ == "__main__":
    sys.exit(main())
