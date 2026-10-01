"""Vásárhelyi 2018 flocking logs -> trace files for the ``trace`` scenario source.

For each flight: the window in which all 30 drones are airborne (> 10 m above their
take-off point), sampled every 0.2 s (the native GPX rate), positions in metres east /
north of the flight's mean position and height above each drone's take-off point.

    .venv/bin/python tools/data/build_vasarhelyi.py   # -> data/processed/vasarhelyi2018/<flight>.npz
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from inspect_vasarhelyi import R_EARTH, ROOT, load

OUT = ROOT.parents[2] / "processed/vasarhelyi2018"
AIRBORNE_M = 10.0
DT = 0.2


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    zip_sha = hashlib.sha256((ROOT.parent / "flight_logs.zip").read_bytes()).hexdigest()
    for flight in sorted(ROOT.iterdir()):
        files = sorted(flight.glob("*.gpx"))
        drones = {p.stem: load(p) for p in files}
        lat0 = float(np.mean([d[:, 1].mean() for d in drones.values()]))
        lon0 = float(np.mean([d[:, 2].mean() for d in drones.values()]))
        tracks, windows = [], []
        for d in drones.values():
            x = np.radians(d[:, 2] - lon0) * R_EARTH * np.cos(np.radians(lat0))
            y = np.radians(d[:, 1] - lat0) * R_EARTH
            z = d[:, 3] - np.median(d[:20, 3])
            up = d[z > AIRBORNE_M, 0]
            windows.append((up.min(), up.max()))
            tracks.append((d[:, 0], np.stack([x, y, z], 1)))
        t0, t1 = max(w[0] for w in windows), min(w[1] for w in windows)
        grid = np.arange(t0, t1, DT)
        pos = np.stack([np.stack([np.interp(grid, t, p[:, j]) for j in range(3)], 1) for t, p in tracks], 1)
        np.savez_compressed(OUT / f"{flight.name}.npz", t=grid - grid[0], pos=pos,
                            drones=np.array([p.stem for p in files]), origin=np.array([lat0, lon0]),
                            start_unix=grid[0], source_sha256=zip_sha)
        print(f"{flight.name}: {len(drones)} drones, {len(grid)} samples, {(grid[-1] - grid[0]) / 60:.1f} min")


if __name__ == "__main__":
    main()
