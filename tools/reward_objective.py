"""The training objective (reward A) of every policy in a dev comparison (spec of e11_compare.py).

Reward A per cycle = alpha * service - beta * queue - eta * delay - lambda_viol * violation
(weights of ``configs/protocol_final.toml``), recomputed from the per-episode component means so
that every policy is scored with the same weights, whatever its own training reward was.  The
evaluation includes the drain period (traffic stops at the horizon), so this is a proxy for the
discounted objective PPO maximises, not the objective itself.

Learned groups are averaged over their training seeds per scene, then paired with the main group
by scene (mean and t-interval over scenes).  With ``--reference`` the share of the main group's
gain over the reference that each group keeps is reported as well.

    .venv/bin/python tools/reward_objective.py configs/experiments/e17_dev.json
    .venv/bin/python tools/reward_objective.py configs/experiments/e18_dev.json --reference 模仿起点
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy import stats

from confirm import ROOT
from fanet_next.config import load_config

PARTS = ("service", "queue", "delay", "violation")


def weights() -> dict[str, float]:
    r = load_config(str(ROOT / "configs/protocol_final.toml"))["reward"]
    return {"service": r["alpha"], "queue": -r["beta"], "delay": -r["eta"], "violation": -r["lambda_viol"]}


def _rows(path) -> dict[int, dict]:
    res = json.loads(path.read_text())["results"]
    return {r["seed"]: r for r in next(iter(res.values()))["rows"]}


def scores(spec: dict, family: str, w: dict) -> tuple[list[int], dict[str, dict[str, np.ndarray]]]:
    base = ROOT / spec["out"] / family / "lightweight"
    entries = {name: [f"{r.replace('/', '_')}.shard0.json" for r in runs] for name, runs in spec["groups"].items()}
    entries.update({b: [f"{b}.shard0.json"] for b in spec.get("baselines", [])})
    out, seeds = {}, None
    for name, files in entries.items():
        per = [_rows(base / f) for f in files]
        seeds = sorted(per[0]) if seeds is None else seeds
        def mean(key):
            return np.array([[p[s][key] for s in seeds] for p in per]).mean(0)
        comps = {k: mean(f"reward_{k}_mean") for k in PARTS}
        out[name] = {"reward": sum(w[k] * comps[k] for k in PARTS), **comps,
                     "delivery": mean("delivery_ratio"), "delay": mean("e2e_delay_mean_s")}
    return seeds, out


def paired(x: np.ndarray, y: np.ndarray) -> list[float]:
    d = x - y
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return [float(d.mean()), float(d.mean() - h), float(d.mean() + h)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--reference", help="group whose gap to the main group defines the gain")
    a = ap.parse_args()
    spec = json.loads((ROOT / a.spec).read_text())
    w = weights()
    main_name = spec["main"]
    report = {"spec": a.spec, "weights": w, "families": {}}
    for family in spec["scenarios"]:
        _, sc = scores(spec, family, w)
        m = sc[main_name]
        fam = {"main_reward": float(m["reward"].mean()), "groups": {}}
        print(f"== {family}: {main_name} reward A per cycle {m['reward'].mean():+.5f}")
        for name, s in sc.items():
            if name == main_name:
                continue
            row = {"reward_minus_main": paired(s["reward"], m["reward"]),
                   "parts_minus_main": {k: paired(w[k] * s[k], w[k] * m[k]) for k in PARTS}}
            if a.reference:
                ref = sc[a.reference]["reward"]
                gain = m["reward"].mean() - ref.mean()
                row["share_of_main_gain_kept"] = float((s["reward"].mean() - ref.mean()) / gain) if gain else None
            fam["groups"][name] = row
            d = row["reward_minus_main"]
            extra = f"  keeps {row['share_of_main_gain_kept']:+.0%} of the gain" if a.reference else ""
            print(f"   {name:16s} {d[0]:+.5f} [{d[1]:+.5f}, {d[2]:+.5f}] ({d[0] / abs(m['reward'].mean()):+.1%}){extra}")
        report["families"][family] = fam
    out = ROOT / spec["out"] / "reward_objective.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
