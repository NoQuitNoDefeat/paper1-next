"""E15 design basis: how much of the UCSB link variability is slow (known to a scheduler)?

The reception model of E13 part 4 cannot tell slow from fast variability: every packet
is judged on its own.  The receivers also log the RSSI of every received packet, so the
received power itself can be split (data/raw/ucsb_802154, primary configuration of part
4: horizontal transmitter without obstruction -> horizontal receiver):

* RSSI (dBm) = P0 - 10 n log10 d + residual, least squares on received packets at
  distances where nearly every packet arrives (so that losses below the sensitivity do
  not truncate the residuals; the cut-off is reported);
* residual = per-link offset (one value per run, transmitter and receiver: placement,
  antennas, terrain, i.e. slow) + within-run part;
* autocorrelation of the within-run part between packets of one link at lags of
  0.5-20 s: what persists over a few cycles can be measured and is "known", what is gone
  after 0.5 s is fast fading.

    .venv/bin/python tools/data/ucsb_rssi.py      # writes results/e15/ucsb_rssi.json
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from ucsb_linkmodel import RAW, RUN, SEND_S, enu, ts

OUT = Path(__file__).resolve().parents[2] / "results/e15/ucsb_rssi.json"
LAGS = (1, 2, 4, 10, 20, 40)  # packets (0.5 s each)


def received() -> list[dict]:
    """One record per received packet (first copy of a sequence number) with its 3-D distance."""
    notes = {(round(float(r["Lat"]), 6), round(float(r["Lon"]), 6)): r["Note"]
             for r in csv.DictReader(open(RAW / "static_transmitters.csv", newline=""))}
    first: dict[tuple, dict] = {}
    times = defaultdict(list)
    for r in csv.DictReader(open(RAW / "signal.csv", newline="")):
        run = tuple(r[k] for k in RUN)
        tx, seq, rx = r["Transmitter_Id"], int(r["Transmitter_Packet_Seq"]), r["Device"]
        times[(run, tx)].append((seq, ts(r["Time"])))
        key = (run, tx, rx, seq)
        if key not in first:
            lat, lon = float(r["Transmitter_Real_Lat"]), float(r["Transmitter_Real_Lon"])
            first[key] = {"rssi": -float(r["Rssi"]), "tx_lat": lat, "tx_lon": lon,
                          "tx_alt": float(r["Transmitter_Real_Altitude"]),
                          "note": notes.get((round(lat, 6), round(lon, 6)), "?")}
    drone = defaultdict(list)
    for r in csv.DictReader(open(RAW / "drone.csv", newline="")):
        drone[tuple(r[k] for k in RUN)].append((ts(r["Time"]), float(r["Lat"]), float(r["Lon"]),
                                                float(r["Altitude"])))
    offset = {k: float(np.median(np.array(v)[:, 1] - SEND_S * np.array(v)[:, 0])) for k, v in times.items()}
    out = []
    for (run, tx, rx, seq), p in first.items():
        if run not in drone:
            continue
        d = np.array(drone[run])
        t = offset[(run, tx)] + SEND_S * seq
        if not d[0, 0] <= t <= d[-1, 0]:
            continue
        x, y = enu(np.interp(t, d[:, 0], d[:, 1]), np.interp(t, d[:, 0], d[:, 2]), p["tx_lat"], p["tx_lon"])
        z = np.interp(t, d[:, 0], d[:, 3]) - p["tx_alt"]
        out.append({"link": (run, tx, rx), "seq": seq, "rx": rx, "note": p["note"],
                    "dist": float(math.sqrt(x * x + y * y + z * z)), "rssi": p["rssi"]})
    return out


def analyse(rows: list[dict], d_max: float) -> dict:
    rows = [r for r in rows if 1.0 <= r["dist"] <= d_max]
    d = np.array([r["dist"] for r in rows])
    p = np.array([r["rssi"] for r in rows])
    a = np.stack([np.ones_like(d), -10 * np.log10(d)], 1)
    coef, *_ = np.linalg.lstsq(a, p, rcond=None)
    res = p - a @ coef
    links = defaultdict(list)
    for r, e in zip(rows, res):
        links[r["link"]].append((r["seq"], e))
    offsets, within = [], {}
    for k, v in links.items():
        if len(v) < 20:
            continue
        v = np.array(sorted(v))
        offsets.append(v[:, 1].mean())
        within[k] = (v[:, 0].astype(int), v[:, 1] - v[:, 1].mean())
    w_all = np.concatenate([w for _, w in within.values()])
    acf = {}
    for lag in LAGS:
        x, y = [], []
        for seq, w in within.values():
            pos = {s: i for i, s in enumerate(seq)}
            for i, s in enumerate(seq):
                j = pos.get(s + lag)
                if j is not None:
                    x.append(w[i])
                    y.append(w[j])
        acf[f"{lag * SEND_S:g}s"] = {"pairs": len(x), "corr": float(np.corrcoef(x, y)[0, 1]) if len(x) > 50 else None}
    return {"d_max_m": d_max, "packets": len(rows), "links": len(within),
            "p0_dbm": float(coef[0]), "exponent": float(coef[1]),
            "sigma_total_db": float(res.std()), "sigma_link_offset_db": float(np.std(offsets)),
            "sigma_within_db": float(w_all.std()), "within_autocorr": acf}


def main() -> None:
    rows = received()
    primary = [r for r in rows if r["rx"] == "xbee_h" and r["note"].startswith("Horizontal, no obstruction")]
    out = {"note": "RSSI resolution 1 dB; residuals of received packets only",
           "primary": [analyse(primary, d) for d in (60.0, 100.0)],
           "pooled": [analyse(rows, d) for d in (60.0, 100.0)]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    for label in ("primary", "pooled"):
        for r in out[label]:
            print(f"== {label}, d <= {r['d_max_m']:.0f} m: {r['packets']} packets, {r['links']} links, "
                  f"n {r['exponent']:.2f}, sigma total {r['sigma_total_db']:.2f} dB = link offsets "
                  f"{r['sigma_link_offset_db']:.2f} + within {r['sigma_within_db']:.2f}")
            print("   within-link autocorrelation: " + ", ".join(
                f"{k} {v['corr']:.2f}" for k, v in r["within_autocorr"].items() if v["corr"] is not None))


if __name__ == "__main__":
    main()
