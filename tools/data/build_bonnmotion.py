"""Third-party mobility traces from BonnMotion (GPL-2.0, github.com/sys-uos/BonnMotion, v3.0.1)
for the ``trace`` scenario source, plus their dataset configs.

Only the horizontal mobility model is changed relative to the synthetic training scenes:
16 nodes, 300 m x 300 m, speeds 5-30 m/s, no pauses.  BonnMotion generates 2-D motion
(its 3-D output is unusable here: RPGM keeps the group reference at the floor and
Gauss-Markov ignores depth), and each node gets a fixed altitude drawn uniformly from
80-120 m as in the synthetic scenes (numpy, seeded by the trace seed).
BonnMotion defaults are kept except these scale settings (its defaults are pedestrian:
0.5-1.5 m/s, 60 s pauses, 2.5 m around the group centre):

* RPGM (reference point group mobility): average group size 4 (default 3), at most
  30 m from the group centre (real flocking: nearest neighbours 11-27 m), group size
  s.d. 2 and group change probability 0.01 (defaults);
* Gauss-Markov: uniform initial speed in 5-30 m/s, speed s.d. 2 m/s (default 0.5 at
  1.5 m/s top speed), bouncing at the borders; angle s.d. pi/8 and 2.5 s updates (defaults).

Each trace: 600 s after skipping 600 s (initial transient), BonnMotion seed per trace;
train seeds 1-20, dev 101-104, test 201-208.  RPGM members may leave the box slightly
(group deviation); the trace keeps them.

    .venv/bin/python tools/data/build_bonnmotion.py
"""

from __future__ import annotations

import gzip
import subprocess
import tempfile
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
BM = PROJECT / "third_party/BonnMotion/bin/bm"
OUT = PROJECT / "data/processed/bonnmotion"
COMMON = ["-n", "16", "-d", "600", "-i", "600", "-x", "300", "-y", "300", "-J", "2D"]
MODELS = {
    "rpgm": ["RPGM", "-l", "5", "-h", "30", "-p", "0", "-a", "4", "-r", "30"],
    "gauss_markov": ["GaussMarkov", "-m", "5", "-h", "30", "-s", "2", "-b", "-u"],
}
SPLITS = {"train": range(1, 21), "dev": range(101, 105), "test": range(201, 209)}
ALTITUDE_M = (80.0, 120.0)
DT = 0.2


def parse_movements(path: Path, duration: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    lines = [l for l in gzip.open(path, "rt").read().splitlines() if l.strip() and not l.startswith("#")]
    grid = np.arange(0.0, duration + 1e-9, DT)
    pos = np.empty((len(grid), len(lines), 3))
    for i, line in enumerate(lines):
        w = np.array(line.split(), dtype=float).reshape(-1, 3)  # t x y waypoints
        for j in range(2):
            pos[:, i, j] = np.interp(grid, w[:, 0], w[:, j + 1])
    pos[:, :, 2] = np.random.default_rng(seed).uniform(*ALTITUDE_M, size=len(lines))[None, :]
    return grid, pos


def main() -> None:
    version = subprocess.run([str(BM)], capture_output=True, text=True).stdout.splitlines()[0]
    for model, args in MODELS.items():
        out = OUT / model
        out.mkdir(parents=True, exist_ok=True)
        names = {}
        for split, seeds in SPLITS.items():
            names[split] = []
            for seed in seeds:
                with tempfile.TemporaryDirectory() as tmp:
                    cmd = [str(BM), "-f", "s", *args[:1], *COMMON, "-R", str(seed), *args[1:]]
                    subprocess.run(cmd, cwd=tmp, check=True, capture_output=True)
                    t, pos = parse_movements(Path(tmp) / "s.movements.gz", 600.0, seed)
                    params = (Path(tmp) / "s.params").read_text()
                name = f"{split}_{seed:03d}"
                np.savez_compressed(out / f"{name}.npz", t=t, pos=pos, params=params, generator=version)
                names[split].append(name)
        cfg = PROJECT / f"configs/datasets/bonnmotion_{model}.toml"
        cfg.write_text(DATASET.format(model=model, title=TITLES[model],
                                      train=names["train"], dev=names["dev"], test=names["test"]))
        print(f"{model}: {sum(len(v) for v in names.values())} traces -> {out}, config {cfg.name}")


TITLES = {"rpgm": "参考点组群移动（RPGM）", "gauss_markov": "高斯-马尔可夫"}
DATASET = """# 第三方移动模型：BonnMotion v3.0.1 的{title}（GPL-2.0，github.com/sys-uos/BonnMotion）。
# 由 tools/data/build_bonnmotion.py 生成：轨迹在 data/processed/bonnmotion/{model}/（不进 git）。
# 只换移动模型，其余与合成训练场景一致（16 节点，300 m，80-120 m，5-30 m/s）。
extends = "../protocol_final.toml"
name = "bonnmotion-{model}"

[scenario]
type = "trace"
dir = "data/processed/bonnmotion/{model}"
train = {train}
dev = {dev}
test = {test}
num_nodes = 16
flow_rate_pps = [10.0, 30.0]
cycle_length = 0.02
horizon = 500
packet_size = 1024
queue_capacity = 64
waiting_capacity = 16
waiting_max_wait = 1.0

[scenario.radio]
tx_power_dbm = 20.0
noise_dbm = -80.0
pathloss_ref_db = 40.05
pathloss_exponent = 2.5
sinr_threshold_db = 4.771212547196624
rate_bps = 4e6
service_window_s = 0.02
"""

if __name__ == "__main__":
    main()
