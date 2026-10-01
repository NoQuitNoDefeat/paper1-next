"""E13 part 4: does the simulator's link model reproduce real packet reception?  (B-class check)

Replay of the UCSB 802.15.4 aerial measurements (data/raw/ucsb_802154):

* every transmitter sends one packet every 0.5 s with one sequence counter, so a
  packet's send time follows from its sequence number (line fitted per run and
  transmitter on its receptions); all packets sent while the UAS log runs count,
  received ones once per (run, transmitter, receiver, sequence);
* UAS position at the send time (50 Hz log), transmitter position from the
  per-packet ``Transmitter_Real_*`` columns -> 3-D distance d.

Runs are split in two halves with a fixed seed: parameters are fitted by maximum
likelihood on one half and judged on the other.  All models are
P(received | d) = rho * S(d), rho a distance-free delivery cap (MAC/collision losses):

* step:      S = 1 if d <= D else 0 (floor 1e-3) - threshold decoding without fading;
* rician:    S = P(|h|^2 >= 10^(-(c - 10 n log10 d)/10)), |h|^2 unit-mean Rician with K;
* lognormal: S = Phi((c - 10 n log10 d) / sigma).

Metrics on the held-out half: mean log-likelihood per packet and the packet-weighted
mean absolute error of the reception ratio in 10 m distance sectors.  Every model
depends on d only, so likelihoods are computed on 1 m distance bins (bin centre,
received / lost counts) - an approximation well below the GPS error.  Primary
configuration: horizontal transmitter without obstruction -> horizontal receiver
(as the paper's main case); secondary: all configurations pooled.

    .venv/bin/python tools/data/ucsb_linkmodel.py
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import ncx2, norm

RAW = Path(__file__).resolve().parents[2] / "data/raw/ucsb_802154"
OUT = Path(__file__).resolve().parents[2] / "results/e13/ucsb_linkmodel.json"
RUN = ("Source", "Location", "Flight_Group", "Scenario", "Run")
R_EARTH = 6_371_000.0
SEND_S = 0.5
FLOOR = 1e-3


def ts(text: str) -> float:
    head, _, frac = text.strip().partition(".")
    return datetime.fromisoformat(f"{head}.{frac[:6]}" if frac else head).timestamp()


def enu(lat, lon, lat0, lon0):
    return (np.radians(np.asarray(lon) - lon0) * R_EARTH * math.cos(math.radians(lat0)),
            np.radians(np.asarray(lat) - lat0) * R_EARTH)


def build_packets() -> list[dict]:
    """One record per (run, transmitter, receiver) with distance and outcome of every sent packet."""
    notes = {(round(float(r["Lat"]), 6), round(float(r["Lon"]), 6)): r["Note"]
             for r in csv.DictReader(open(RAW / "static_transmitters.csv", newline=""))}
    recv = defaultdict(set)              # (run, tx, rx) -> received sequence numbers
    times = defaultdict(list)            # (run, tx) -> (seq, reception time)
    txpos = {}
    for r in csv.DictReader(open(RAW / "signal.csv", newline="")):
        run = tuple(r[k] for k in RUN)
        tx, seq = r["Transmitter_Id"], int(r["Transmitter_Packet_Seq"])
        recv[(run, tx, r["Device"])].add(seq)
        times[(run, tx)].append((seq, ts(r["Time"])))
        lat, lon = float(r["Transmitter_Real_Lat"]), float(r["Transmitter_Real_Lon"])
        txpos[(run, tx)] = (lat, lon, float(r["Transmitter_Real_Altitude"]),
                            notes.get((round(lat, 6), round(lon, 6)), "?"))
    drone = defaultdict(list)
    for r in csv.DictReader(open(RAW / "drone.csv", newline="")):
        drone[tuple(r[k] for k in RUN)].append((ts(r["Time"]), float(r["Lat"]), float(r["Lon"]),
                                                float(r["Altitude"])))
    out = []
    for (run, tx), st in times.items():
        if run not in drone:
            continue
        d = np.array(drone[run])
        seq_t = np.array(st)
        offset = float(np.median(seq_t[:, 1] - SEND_S * seq_t[:, 0]))  # t_send = offset + 0.5 seq
        lo = int(math.ceil((d[0, 0] - offset) / SEND_S))
        hi = int(math.floor((d[-1, 0] - offset) / SEND_S))
        seqs = np.arange(lo, hi + 1)
        tsend = offset + SEND_S * seqs
        lat0, lon0, alt_tx, note = txpos[(run, tx)]
        lat_u = np.interp(tsend, d[:, 0], d[:, 1])
        lon_u = np.interp(tsend, d[:, 0], d[:, 2])
        x, y = enu(lat_u, lon_u, lat0, lon0)
        z = np.interp(tsend, d[:, 0], d[:, 3]) - alt_tx
        dist = np.sqrt(x * x + y * y + z * z)
        for rx in ("xbee_h", "xbee_v"):
            got = recv.get((run, tx, rx), set())
            out.append({"run": run, "tx": tx, "rx": rx, "note": note, "dist": dist,
                        "ok": np.isin(seqs, list(got)).astype(float)})
    return out


# ------------------------------------------------------------------ models
def rician_success(margin_db: np.ndarray, k_db: float) -> np.ndarray:
    """P(|h|^2 >= 10^(-margin/10)) for unit-mean Rician power with K (dB)."""
    k = 10 ** (k_db / 10)
    x = 10 ** (-margin_db / 10)
    return ncx2.sf(2 * (k + 1) * x, 2, 2 * k)


def prob(model: str, p: np.ndarray, d: np.ndarray) -> np.ndarray:
    rho = 1 / (1 + np.exp(-p[0]))
    if model == "step":
        s = np.where(d <= np.exp(p[1]), 1.0, 0.0)
    else:
        margin = p[1] - 10 * p[2] * np.log10(np.maximum(d, 1.0))
        s = rician_success(margin, p[3]) if model == "rician" else norm.cdf(margin / np.exp(p[3]))
    return np.clip(rho * s, FLOOR, 1 - FLOOR)


def bins(d: np.ndarray, ok: np.ndarray, width: float = 1.0):
    """(bin centre, received, lost) over occupied 1 m distance bins."""
    idx = (d // width).astype(int)
    n = np.bincount(idx)
    k = np.bincount(idx, weights=ok)
    keep = n > 0
    return (np.flatnonzero(keep) + 0.5) * width, k[keep], (n - k)[keep]


def nll(model, p, b):
    centre, got, lost = b
    q = prob(model, p, centre)
    return -(got @ np.log(q) + lost @ np.log(1 - q)) / (got.sum() + lost.sum())


def fit(model: str, d: np.ndarray, ok: np.ndarray) -> np.ndarray:
    b = bins(d, ok)
    if model == "step":  # one-dimensional range search, then rho
        best = None
        for D in np.arange(10, 400, 1.0):
            inside = d <= D
            rho = min(max(ok[inside].mean() if inside.any() else 0.5, 1e-3), 1 - 1e-3)
            p = np.array([math.log(rho / (1 - rho)), math.log(D)])
            v = nll(model, p, b)
            if best is None or v < best[0]:
                best = (v, p)
        return best[1]
    starts = [np.array([2.0, c, n, k]) for c in (40.0, 60.0) for n in (2.0, 3.0)
              for k in ((3.0, 10.0) if model == "rician" else (math.log(3.0), math.log(8.0)))]
    res = [minimize(lambda p: nll(model, p, b), s, method="Nelder-Mead",
                    options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-7}) for s in starts]
    return min(res, key=lambda r: r.fun).x


def sector_mae(model, p, d, ok, width=10.0):
    sec = (d // width).astype(int)
    err, w = 0.0, 0
    for s in np.unique(sec):
        m = sec == s
        if m.sum() < 20:
            continue
        err += m.sum() * abs(ok[m].mean() - prob(model, p, d[m]).mean())
        w += m.sum()
    return err / w


def describe(model, p):
    rho = 1 / (1 + math.exp(-p[0]))
    if model == "step":
        return {"rho": rho, "range_m": math.exp(p[1])}
    extra = {"k_db": p[3]} if model == "rician" else {"sigma_db": math.exp(p[3])}
    return {"rho": rho, "c_db": p[1], "exponent": p[2], **extra}


def evaluate(packets, select, label):
    runs = sorted({r["run"] for r in packets})
    rng = np.random.default_rng(0)
    fit_runs = set(map(tuple, np.array(runs, dtype=object)[rng.permutation(len(runs))[: len(runs) // 2]]))
    rows = [r for r in packets if select(r)]
    take = lambda half: (np.concatenate([r["dist"] for r in rows if (r["run"] in fit_runs) == half]),
                         np.concatenate([r["ok"] for r in rows if (r["run"] in fit_runs) == half]))
    (d_fit, ok_fit), (d_test, ok_test) = take(True), take(False)
    keep_fit, keep_test = d_fit < 1000, d_test < 1000  # drops the transmitter with a longitude typo
    d_fit, ok_fit, d_test, ok_test = d_fit[keep_fit], ok_fit[keep_fit], d_test[keep_test], ok_test[keep_test]
    result = {"label": label, "packets_fit": int(len(d_fit)), "packets_test": int(len(d_test)),
              "prr_test": float(ok_test.mean()), "models": {}}
    for model in ("step", "rician", "lognormal"):
        p = fit(model, d_fit, ok_fit)
        result["models"][model] = {"params": describe(model, p),
                                   "test_loglik": float(-nll(model, p, bins(d_test, ok_test))),
                                   "test_sector_mae": float(sector_mae(model, p, d_test, ok_test))}
    return result


def main() -> None:
    packets = build_packets()
    out = [evaluate(packets, lambda r: r["rx"] == "xbee_h" and r["note"].startswith("Horizontal, no obstruction"),
                    "horizontal tx (no obstruction) -> horizontal rx"),
           evaluate(packets, lambda r: True, "all configurations pooled")]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    for r in out:
        print(f"== {r['label']}: fit {r['packets_fit']} packets, test {r['packets_test']} (PRR {r['prr_test']:.3f})")
        for m, v in r["models"].items():
            print(f"   {m:9s} loglik {v['test_loglik']:.4f}  sector MAE {v['test_sector_mae']:.4f}  "
                  + "  ".join(f"{k} {x:.3g}" for k, x in v["params"].items()))


if __name__ == "__main__":
    main()
