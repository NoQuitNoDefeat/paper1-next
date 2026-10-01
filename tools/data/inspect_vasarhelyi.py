"""Check the open A-class conditions of the Vásárhelyi 2018 flocking logs (data/raw/vasarhelyi2018).

Per flight: drones, sampling interval, common airborne window, altitude above the
take-off point, horizontal speed, and inter-drone distances during that window.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "data/raw/vasarhelyi2018/extracted"
PT = re.compile(r'<trkpt lat="([-\d.]+)" lon="([-\d.]+)">\s*<ele>([-\d.]+)</ele>\s*<time>([^<]+)</time>')
R_EARTH = 6_371_000.0


def load(path: Path) -> np.ndarray:
    rows = [(datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp(), float(la), float(lo), float(el))
            for la, lo, el, t in PT.findall(path.read_text())]
    return np.array(rows)


def main(airborne_m: float = 10.0) -> None:
    for flight in sorted(ROOT.iterdir()):
        drones = {p.stem: load(p) for p in sorted(flight.glob("*.gpx"))}
        lat0 = np.mean([d[:, 1].mean() for d in drones.values()])
        lon0 = np.mean([d[:, 2].mean() for d in drones.values()])
        dts = np.concatenate([np.diff(d[:, 0]) for d in drones.values()])
        # common time grid at 0.2 s over the window where every drone is airborne
        windows = []
        enu = {}
        for k, d in drones.items():
            x = np.radians(d[:, 2] - lon0) * R_EARTH * np.cos(np.radians(lat0))
            y = np.radians(d[:, 1] - lat0) * R_EARTH
            agl = d[:, 3] - np.percentile(d[:20, 3], 50)  # height above the take-off point
            enu[k] = (d[:, 0], np.stack([x, y, agl], 1))
            up = d[agl > airborne_m, 0]
            windows.append((up.min(), up.max()) if len(up) else (np.inf, -np.inf))
        t0, t1 = max(w[0] for w in windows), min(w[1] for w in windows)
        print(f"== {flight.name}: {len(drones)} drones, median dt {np.median(dts):.3f} s "
              f"(p95 {np.percentile(dts, 95):.3f}), files span "
              f"{min(d[0, 0] for d in drones.values()):.0f}-{max(d[-1, 0] for d in drones.values()):.0f}")
        if t1 <= t0:
            print("   no common airborne window"); continue
        grid = np.arange(t0, t1, 0.2)
        pos = np.stack([np.stack([np.interp(grid, t, p[:, j]) for j in range(3)], 1)
                        for t, p in enu.values()], 1)  # (T, N, 3)
        vel = np.diff(pos, axis=0) / 0.2
        hs = np.linalg.norm(vel[:, :, :2], axis=2)
        diff = pos[:, :, None, :] - pos[:, None, :, :]
        dist = np.linalg.norm(diff, axis=3)
        n = pos.shape[1]
        iu = np.triu_indices(n, 1)
        pair = dist[:, iu[0], iu[1]]
        nn = np.where(np.eye(n, dtype=bool)[None], np.inf, dist).min(axis=2)
        print(f"   common airborne window {(t1 - t0) / 60:.1f} min; AGL median {np.median(pos[:, :, 2]):.1f} m "
              f"[p5 {np.percentile(pos[:, :, 2], 5):.1f}, p95 {np.percentile(pos[:, :, 2], 95):.1f}]")
        print(f"   horizontal speed median {np.median(hs):.2f} m/s [p5 {np.percentile(hs, 5):.2f}, "
              f"p95 {np.percentile(hs, 95):.2f}]")
        print(f"   nearest-neighbour distance median {np.median(nn):.1f} m [p5 {np.percentile(nn, 5):.1f}, "
              f"p95 {np.percentile(nn, 95):.1f}]")
        print(f"   pairwise distance median {np.median(pair):.1f} m [p5 {np.percentile(pair, 5):.1f}, "
              f"p95 {np.percentile(pair, 95):.1f}, max {pair.max():.1f}]; flock diameter median "
              f"{np.median(pair.max(axis=1)):.1f} m")
        ext = pos[:, :, :2].reshape(-1, 2)
        print(f"   horizontal extent {np.ptp(ext[:, 0]):.0f} x {np.ptp(ext[:, 1]):.0f} m")


if __name__ == "__main__":
    main(*(float(a) for a in sys.argv[1:]))
