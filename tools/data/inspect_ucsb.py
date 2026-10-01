"""Check the open B-class conditions of the UCSB 802.15.4 aerial measurements (data/raw/ucsb_802154)."""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

RAW = Path(__file__).resolve().parents[2] / "data/raw/ucsb_802154"
RUN = ("Source", "Location", "Flight_Group", "Scenario", "Run")


def ts(text: str) -> float:
    text = text.strip()
    if "." in text:  # nanosecond fractions: keep microseconds
        head, frac = text.split(".")
        text = f"{head}.{frac[:6]}"
    return datetime.fromisoformat(text).timestamp()


def main() -> None:
    with open(RAW / "signal.csv", newline="") as fh:
        sig = list(csv.DictReader(fh))
    print("signal rows", len(sig))
    print("receivers", {k: sum(r["Device"] == k for r in sig) for k in {r["Device"] for r in sig}})
    print("frequency", {r["Frequency"] for r in sig}, "rates (ms)", sorted({int(r["Transmitter_Rate"]) for r in sig}))
    dist = np.array([float(r["Real_Distance"]) for r in sig])
    rssi = np.array([float(r["Rssi"]) for r in sig])
    print("3-D distance m: min %.0f median %.0f max %.0f" % tuple(np.percentile(dist, [0, 50, 100])))
    print("RSSI magnitude: min %.0f median %.0f max %.0f" % (rssi.min(), np.median(rssi), rssi.max()))
    print("altitude tags", sorted({r["Tag"].strip() for r in sig}, key=lambda x: int(x[:-2])))
    runs = defaultdict(list)
    for r in sig:
        runs[tuple(r[k] for k in RUN)].append(r)
    ntx = [len({r["Transmitter_Id"] for r in rs}) for rs in runs.values()]
    print("runs", len(runs), "transmitters per run min/median/max", min(ntx), int(np.median(ntx)), max(ntx))
    # reception ratio from sequence numbers: 500 ms streams per (run, transmitter, receiver)
    streams = defaultdict(set)
    for r in sig:
        if r["Transmitter_Rate"] == "500":
            streams[(tuple(r[k] for k in RUN), r["Transmitter_Id"], r["Device"])].add(int(r["Transmitter_Packet_Seq"]))
    prr = np.array([len(s) / (max(s) - min(s) + 1) for s in streams.values() if len(s) > 1])
    print("500 ms streams", len(prr), "reception ratio (received / sequence span) median %.3f p10 %.3f p90 %.3f"
          % tuple(np.percentile(prr, [50, 10, 90])))
    # time alignment: do packet times fall inside the drone log of the same run?
    span = {}
    with open(RAW / "drone.csv", newline="") as fh:
        rd = csv.DictReader(fh)
        n, dts, last = 0, [], None
        for r in rd:
            k = tuple(r[c] for c in RUN)
            t = ts(r["Time"])
            lo, hi = span.get(k, (t, t))
            span[k] = (min(lo, t), max(hi, t))
            if last is not None and n < 5000:
                dts.append(t - last)
            last, n = t, n + 1
    print("drone rows", n, "median dt %.3f s" % np.median(dts))
    inside = [all(span.get(k, (np.inf, -np.inf))[0] - 1 <= ts(r["Time"]) <= span.get(k, (np.inf, -np.inf))[1] + 1
                  for r in rs[:50]) for k, rs in runs.items()]
    print("runs whose (first 50) packet times lie within the drone log span of the same run: %d / %d"
          % (sum(inside), len(inside)))


if __name__ == "__main__":
    main()
