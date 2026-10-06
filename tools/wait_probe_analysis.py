"""Pre-registered analysis of S1d (docs/experiments.md, "S1d 预先登记"): when does holding back the
lowest-priority link of the rule plan pay off, and can it be predicted from observable features?

Input: results/s1d/<family>_D1.json (tools/rollout_probe.py --score cohort --log-decisions).  One
row per decision with two or more candidates; label y = 1 when dropping the base plan's last link
scores higher than the base plan (cohort score), else 0; rows without that candidate are skipped.

1. Descriptive: per feature of the dropped link, the mean when waiting wins and when it does not,
   and the standardised difference.
2. Predictability: logistic regression on standardised features (and log-transformed counts),
   evaluated by leave-one-episode-out over all families pooled and per family; ROC AUC on the
   held-out episodes.  AUC >= 0.75 reads as "the information for a wait gate is observable".
3. Value at stake: mean gain of the best wait over the base plan where waiting wins (cohort score,
   packets).

    .venv/bin/python tools/wait_probe_analysis.py
"""

from __future__ import annotations

import json

import numpy as np

from confirm import ROOT

FAMILIES = ("default", "load_high", "nodes24")
FEATURES = ("plan_size", "n_pk", "share_last_hop", "rem_hops_mean", "rem_hops_max", "age_mean", "age_max",
            "share_hopeless", "share_tight", "share_at_source", "rx_queued", "rx_room_min",
            "dist_m", "net_queued")
LOG = {"n_pk", "rx_queued", "net_queued"}
AUC_FLAG = 0.75


def rows() -> list[dict]:
    out = []
    for fam in FAMILIES:
        p = ROOT / "results/s1d" / f"{fam}_D1.json"
        if not p.exists():
            continue
        for ep, r in enumerate(json.loads(p.read_text())["rows"]):
            for d in r["rollout"].get("decisions", []) if "decisions" in r["rollout"] else r.get("decisions", []):
                if "last" not in d or not np.isfinite(d["gain1"]):
                    continue
                best_wait = max(g for g in (d["gain1"], d["gain2"]) if np.isfinite(g))
                out.append({"family": fam, "episode": f"{fam}:{ep}", "y": int(d["gain1"] > 0),
                            "gain_best_wait": best_wait, **{k: float(d["last"][k]) for k in FEATURES}})
    return out


def matrix(rs: list[dict]) -> np.ndarray:
    x = np.array([[np.log1p(r[k]) if k in LOG else r[k] for k in FEATURES] for r in rs], dtype=float)
    return x


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 50) -> np.ndarray:
    """Newton's method with a small ridge; x already standardised, intercept appended."""
    xb = np.hstack([x, np.ones((len(x), 1))])
    w = np.zeros(xb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(xb @ w, -30, 30)))
        g = xb.T @ (p - y) + l2 * np.r_[w[:-1], 0.0]
        h = (xb * (p * (1 - p))[:, None]).T @ xb + l2 * np.diag(np.r_[np.ones(len(w) - 1), 0.0])
        w -= np.linalg.solve(h, g)
    return w


def auc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.r_[pos, neg], kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    allv = np.r_[pos, neg]
    for v in np.unique(allv):  # average ranks for ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def loeo_auc(rs: list[dict]) -> dict:
    x, y = matrix(rs), np.array([r["y"] for r in rs], dtype=float)
    eps = np.array([r["episode"] for r in rs])
    scores = np.full(len(y), np.nan)
    for e in np.unique(eps):
        tr, te = eps != e, eps == e
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-9
        w = fit_logistic((x[tr] - mu) / sd, y[tr])
        scores[te] = np.hstack([(x[te] - mu) / sd, np.ones((te.sum(), 1))]) @ w
    mu, sd = x.mean(0), x.std(0) + 1e-9
    w = fit_logistic((x - mu) / sd, y)
    coef = dict(sorted(zip(FEATURES, w[:-1].tolist()), key=lambda kv: -abs(kv[1])))
    return {"n": int(len(y)), "positives": int(y.sum()), "auc_loeo": auc(scores, y), "coef_standardised": coef}


def main() -> None:
    rs = rows()
    if not rs:
        print("no S1d results")
        return
    y = np.array([r["y"] for r in rs])
    desc = {}
    for k in FEATURES:
        v = np.array([r[k] for r in rs], dtype=float)
        a, b = v[y == 1], v[y == 0]
        desc[k] = {"wait_wins": float(a.mean()) if len(a) else None, "base_wins": float(b.mean()),
                   "std_diff": float((a.mean() - b.mean()) / (v.std() + 1e-9)) if len(a) else None}
    out = {"descriptive": desc, "pooled": loeo_auc(rs),
           "per_family": {f: loeo_auc([r for r in rs if r["family"] == f]) for f in FAMILIES
                          if any(r["family"] == f for r in rs)},
           "gain_where_wait_wins": {f: float(np.mean([r["gain_best_wait"] for r in rs if r["family"] == f and r["y"]]))
                                    for f in FAMILIES if any(r["family"] == f and r["y"] for r in rs)},
           "auc_flag": AUC_FLAG}
    out["observable"] = out["pooled"]["auc_loeo"] >= AUC_FLAG
    (ROOT / "results/s1d/analysis.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({"pooled": out["pooled"], "observable": out["observable"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
