"""Packet reception ratio (PRR) of the UCSB 802.15.4 aerial measurements, computed two ways.

Each transmitter sends one packet every 0.5 s from a single sequence counter; the
``Transmitter_Rate`` labels (500, 1000, 15000, ... ms) split those packets into
interleaved logical streams (odd sequence numbers carry "500", even ones the others),
so a sequence span counts packets of every label.  XBee repeats broadcasts, so
the same sequence number can be logged more than once (2.5 % of rows overall, far
more close to a transmitter): receptions are counted once per sequence number.

1. Sequence method, per (run, transmitter, receiver): received distinct sequence
   numbers (all labels) / (last - first + 1).
2. Sector method of the paper (Nekrasov et al., DroNet'19, Fig. 5 / Table 1): in
   10 m horizontal-displacement sectors, distinct received packets / (time the UAS
   spent in the sector / 0.5 s), here for altitudes 150-250 ft, horizontal transmitter
   (static note "Horizontal, no obstruction") and horizontal receiver, against the
   paper's ERR50 = 130 m (PRR > 50 % up to 130 m) and ERR25 = 180 m.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

RAW = Path(__file__).resolve().parents[2] / "data/raw/ucsb_802154"
RUN = ("Source", "Location", "Flight_Group", "Scenario", "Run")
SEND_INTERVAL_S = 0.5
R_EARTH = 6_371_000.0


def ts(text: str) -> float:
    head, _, frac = text.strip().partition(".")
    return datetime.fromisoformat(f"{head}.{frac[:6]}" if frac else head).timestamp()


def horiz(lat1, lon1, lat2, lon2):
    x = math.radians(lon2 - lon1) * R_EARTH * math.cos(math.radians((lat1 + lat2) / 2))
    y = math.radians(lat2 - lat1) * R_EARTH
    return math.hypot(x, y)


def main() -> None:
    sig = list(csv.DictReader(open(RAW / "signal.csv", newline="")))
    # 1. sequence method
    seqs = defaultdict(set)
    for r in sig:
        seqs[(tuple(r[k] for k in RUN), r["Transmitter_Id"], r["Device"])].add(int(r["Transmitter_Packet_Seq"]))
    prr = np.array([len(s) / (max(s) - min(s) + 1) for s in seqs.values() if len(s) > 1])
    print(f"sequence method: {len(prr)} (run, tx, rx) streams; PRR median {np.median(prr):.3f} "
          f"(p10 {np.percentile(prr, 10):.3f}, p90 {np.percentile(prr, 90):.3f}); pooled "
          f"{sum(len(s) for s in seqs.values()) / sum(max(s) - min(s) + 1 for s in seqs.values()):.3f}")
    # 2. sector method (paper), horizontal tx -> horizontal rx, 150-250 ft.  The transmitter's
    # position is the per-packet Transmitter_Real_Lat/Lon: static_transmitters.csv keyed by
    # Source does not match it for many runs (offsets of 150-260 m, one of 8.3 km), so the
    # orientation note is matched by position instead.
    notes = {}
    for r in csv.DictReader(open(RAW / "static_transmitters.csv", newline="")):
        notes[(round(float(r["Lat"]), 6), round(float(r["Lon"]), 6))] = r["Note"]
    tags = {"150ft", "200ft", "250ft"}
    runs_tx = defaultdict(lambda: defaultdict(float))
    recv = defaultdict(int)
    good_runs = {}
    seen = set()  # XBee repeats broadcasts: count each (run, tx, rx, seq) once, at its first reception
    unmatched = set()
    for r in sorted(sig, key=lambda r: ts(r["Time"])):
        if r["Tag"].strip() not in tags or r["Device"] != "xbee_h":
            continue
        pos = (round(float(r["Transmitter_Real_Lat"]), 6), round(float(r["Transmitter_Real_Lon"]), 6))
        note = notes.get(pos)
        if note is None:
            unmatched.add(pos)
            continue
        key = (tuple(r[k] for k in RUN), r["Transmitter_Id"], r["Device"], r["Transmitter_Packet_Seq"])
        if note.startswith("Horizontal, no obstruction") and key not in seen:
            seen.add(key)
            recv[int(float(r["Real_DistanceXY"]) // 10)] += 1
            good_runs[(tuple(r[k] for k in RUN), r["Transmitter_Id"])] = pos
    print(f"  transmitter positions without a static note: {len(unmatched)}")
    want = defaultdict(list)
    for (run, tx), pos in good_runs.items():
        want[run].append(pos)
    last = {}
    with open(RAW / "drone.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            run = tuple(r[k] for k in RUN)
            if run not in want or r["Tag"].strip() not in tags:
                continue
            t = ts(r["Time"])
            dt = min(t - last.get(run, t), 0.1)  # gaps > 0.1 s are logging breaks
            last[run] = t
            lat, lon = float(r["Lat"]), float(r["Lon"])
            for tlat, tlon in want[run]:
                runs_tx[0][int(horiz(lat, lon, tlat, tlon) // 10)] += dt
    sent = {s: sec / SEND_INTERVAL_S for s, sec in runs_tx[0].items()}
    print("sector method (150-250 ft, horizontal tx -> horizontal rx):")
    err = {}
    for s in sorted(sent):
        if sent[s] < 5 or s > 30:
            continue
        p = recv.get(s, 0) / sent[s]
        print(f"  {s * 10:3d}-{s * 10 + 10:3d} m: sent {sent[s]:7.0f}, received {recv.get(s, 0):6d}, PRR {p:.3f}")
        err[s] = p
    for n in (50, 33, 25):
        ok = [s for s in sorted(err) if all(err[x] > n / 100 for x in sorted(err) if x <= s)]
        print(f"  ERR{n}: {((max(ok) + 1) * 10) if ok else 0} m (paper: {dict(((50, 130), (33, 150), (25, 180)))[n]} m)")


if __name__ == "__main__":
    main()
