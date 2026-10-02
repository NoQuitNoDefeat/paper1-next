"""Trace files from the Verdoucq et al. 2025 swarm flights (Zenodo 17902132, CC BY 4.0) -> E16.

Each raw file (data/raw/verdoucq2025/extracted/data/raw_Fig*.txt) has one row per ~1 s:
time, the experiment's control parameter, then x y z (m) of 10 swarm drones and 1 intruder.
For the network only the 10 swarm drones count (the intruder is not part of the swarm).

* airborne part: the longest stretch in which all 10 swarm drones fly at >= 4 m (the
  mission altitude is 5-15 m; take-off and landing are dropped);
* resampled onto a uniform 1 s grid (linear), time starting at 0;
* scaled as a whole by one factor k for all flights (geometric similarity).  The swarm
  is ~20 m across, so unscaled every drone would be one hop from every other.  k is chosen,
  before any policy runs, so that the mean fraction of drone pairs within the single-hop
  range of the flock30 main condition (66 m) equals that of the synthetic training scenes
  (0.54, results/e14/topology.md); positions and speeds scale by k, the rate of topology
  change relative to the range does not.

    .venv/bin/python tools/data/build_verdoucq.py   # -> data/processed/verdoucq2025/*.npz, build.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/verdoucq2025/extracted/data"
OUT = ROOT / "data/processed/verdoucq2025"
SWARM = 10
MIN_Z = 4.0
RANGE_M = 66.0
TARGET_IN_RANGE = 0.54


def airborne(t: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = (pos[:, :, 2] >= MIN_Z).all(axis=1)
    best, start = (0, 0), None
    for i, v in enumerate(np.append(ok, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            best = max(best, (start, i), key=lambda ab: ab[1] - ab[0])
            start = None
    a, b = best
    grid = np.arange(np.ceil(t[a]), np.floor(t[b - 1]) + 1e-9, 1.0)
    res = np.stack([np.stack([np.interp(grid, t[a:b], pos[a:b, i, j]) for j in range(3)], 1)
                    for i in range(pos.shape[1])], 1)
    return grid - grid[0], res


def load() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for f in sorted(RAW.glob("raw_Fig*.txt")):
        a = np.loadtxt(f)
        pos = a[:, 2:].reshape(len(a), -1, 3)[:, :SWARM]
        out[f.stem.split("_")[1].lower()] = airborne(a[:, 0], pos)  # e.g. "fig4"
    return out


def in_range_fraction(flights: dict, k: float) -> float:
    fr = []
    iu = np.triu_indices(SWARM, 1)
    for _, pos in flights.values():
        d = np.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[:, iu[0], iu[1]] * k
        fr.append((d <= RANGE_M).mean())
    return float(np.mean(fr))


def topology(t: np.ndarray, pos: np.ndarray, rng: np.random.Generator) -> dict:
    """Method-independent statistics as in tools/e14_topology.py (16 random 10 s windows, 66 m)."""
    iu = np.triu_indices(SWARM, 1)
    rows = []
    for _ in range(16):
        times = rng.uniform(t[0], t[-1] - 10.02) + 0.02 * np.arange(501)
        p = np.stack([np.stack([np.interp(times, t, pos[:, i, j]) for j in range(3)], 1) for i in range(SWARM)], 1)
        d = np.linalg.norm(p[:, :, None, :] - p[:, None, :, :], axis=-1)[:, iu[0], iu[1]]
        inr = d <= RANGE_M
        rate = np.abs(np.diff(d, axis=0)).mean() / 0.02
        rows.append([inr.mean(), inr.sum(1).mean() * 2 / SWARM, (inr[1:] != inr[:-1]).mean() / 0.02, rate / RANGE_M])
    m = np.mean(rows, 0)
    return {"in_range": float(m[0]), "degree": float(m[1]), "flips_per_pair_s": float(m[2]),
            "rate_over_range_per_s": float(m[3])}


def main() -> None:
    flights = load()
    lo, hi = 1.0, 50.0  # fraction within range falls as k grows
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if in_range_fraction(flights, mid) > TARGET_IN_RANGE else (lo, mid)
    k = round(0.5 * (lo + hi), 2)
    OUT.mkdir(parents=True, exist_ok=True)
    info = {"scale": k, "target_in_range": TARGET_IN_RANGE, "range_m": RANGE_M, "min_altitude_m": MIN_Z,
            "in_range_fraction": in_range_fraction(flights, k), "flights": {}}
    rng = np.random.default_rng(0)
    for name, (t, pos) in flights.items():
        np.savez(OUT / f"{name}.npz", t=t, pos=pos * k)
        info.setdefault("topology", {})[name] = topology(t, pos * k, rng)
        speed = np.linalg.norm(np.diff(pos, axis=0), axis=-1).mean()
        info["flights"][name] = {"seconds": float(t[-1]), "samples": len(t), "mean_speed_mps_raw": float(speed),
                                 "altitude_m_scaled": [float(pos[:, :, 2].min() * k), float(pos[:, :, 2].max() * k)]}
    (OUT / "build.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
