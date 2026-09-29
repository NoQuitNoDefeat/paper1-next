"""E10: ns-3 PHY validation of the final method (spec: configs/experiments/e10_ns3.json).

Every policy is evaluated on the same held-out seeds by two executors: the
lightweight backend (the training environment) and ns-3 (`eval --backend ns3`:
Spectrum PHY with cumulative interference, Shannon decoding, frame overhead,
propagation and continuous intra-cycle motion).  ns-3 jobs are split into seed
shards so they run in parallel; the summary merges the shards.

    .venv/bin/python tools/e10_ns3.py --spec configs/experiments/e10_ns3.json --parallel 8
    .venv/bin/python tools/e10_ns3.py --spec ... --status --watch 30  # progress board
    .venv/bin/python tools/e10_ns3.py --spec ... --summary           # tables only

Memory: an ns-3 job peaks at ~2.5 GiB while ns-3 parses an episode's INIT (~8 s)
and then runs at ~0.5 GiB, so job starts are staggered (``--gap``).  A failed
job is retried once (its log is kept as ``*.failed1.log``); rerunning the driver
only runs shards whose output is missing.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from confirm import CLI, ROOT, ci, resolve, slug

KEYS = [("delivery_ratio", "交付率", "{:.4f}"), ("termination_ratio", "终止比例", "{:.4f}"),
        ("ontime_2s", "2s 送达", "{:.4f}"), ("e2e_delay_mean_s", "平均时延", "{:.3f}"),
        ("e2e_delay_p95_s", "p95 时延", "{:.3f}"), ("exec_failed_link_frac", "整链失败", "{:.4f}"),
        ("per", "物理层误包率", "{:.4f}")]
PAIR_KEYS = ["delivery_ratio", "ontime_2s", "e2e_delay_mean_s", "e2e_delay_p95_s"]
NI_MARGIN = -0.005
FIDELITY_DELIVERY = 0.01
FIDELITY_PER = 0.01


def policies(spec: dict) -> dict[str, dict]:
    """name -> how to evaluate it: base args (without seeds/executor) and the ppo flag."""
    out: dict[str, dict] = {}
    for b in spec["baselines"]:
        out[b] = {"args": ["--config", spec["baseline_config"], "--policies", b], "runs": None}
    for name, ckpt in spec.get("fixed", {}).items():
        run_dir = str(Path(ckpt).parent.parent)
        out[name] = {"args": ["--run-dir", run_dir, "--checkpoint", ckpt, "--policies", "ppo"],
                     "runs": None}
    for name, runs in spec["groups"].items():
        out[name] = {"args": None, "runs": runs}
    for name, c in spec.get("controls", {}).items():
        sets = [a for s in c["set"] for a in ("--set", s)]
        out[name] = {"args": ["--config", spec["baseline_config"], "--policies", c["policy"], *sets],
                     "runs": None, "scenarios": c.get("scenarios")}
    return out


def result_files(spec: dict, scenario: str, executor: str, name: str) -> list[tuple[list[str], Path]]:
    """(eval args, output file) per run (group member) and seed shard."""
    e = spec["eval"]
    shards = (e["shards"] if spec["executors"][executor]
              else spec.get("shard_policies", {}).get(name, 1))  # heavy lightweight jobs
    per = e["episodes"] // shards
    p = policies(spec)[name]
    targets = ([(p["args"], slug(name))] if p["runs"] is None else
               [(["--run-dir", r, "--checkpoint", "@selected", "--policies", "ppo"], slug(r))
                for r in p["runs"]])
    out = []
    for args, stem in targets:
        for i in range(shards):
            f = ROOT / spec["out"] / scenario / executor / f"{stem}.shard{i}.json"
            out.append((args + ["--seed-offset", str(e["seed_offset"] + i * per),
                                "--episodes", str(per)], f))
    return out


def build_jobs(spec: dict) -> list[list[str]]:
    e = spec["eval"]
    jobs = []
    for executor, backend in sorted(spec["executors"].items(), key=lambda kv: kv[1] is None):
        for scenario, overrides in spec["scenarios"].items():
            for name, p in policies(spec).items():
                if (p.get("scenarios") and scenario not in p["scenarios"]) or \
                        name in spec.get("exclude", {}).get(scenario, []):
                    continue
                for args, f in result_files(spec, scenario, executor, name):
                    if f.exists():
                        continue
                    f.parent.mkdir(parents=True, exist_ok=True)
                    cmd = [CLI, "eval", *args, "--split", e["split"], "--drain", str(e["drain"]),
                           "--threads", "1", "--num-envs", "1", "--out", str(f)]
                    cmd += ["--backend", backend] if backend else []
                    cmd += [a for o in overrides for a in ("--set", o)]
                    jobs.append(cmd)
    return jobs


def launch(jobs: list[list[str]], parallel: int, gap: float, log) -> None:
    """Run jobs with at most ``parallel`` at once and starts ``gap`` seconds apart."""
    queue = [(cmd, 0) for cmd in jobs]
    running: dict[subprocess.Popen, tuple[list[str], int]] = {}
    last = 0.0

    def reap() -> None:
        for proc, (cmd, attempt) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[proc]
            out = cmd[cmd.index("--out") + 1]
            if proc.returncode == 0:
                log(f"finished {out}")
                continue
            log(f"FAILED (exit {proc.returncode}, attempt {attempt + 1}) {out}")
            Path(f"{out}.eval.log").replace(f"{out}.failed{attempt + 1}.log")
            if attempt == 0:
                queue.append((cmd, 1))

    while queue or running:
        reap()
        if queue and len(running) < parallel and time.time() - last >= gap:
            cmd, attempt = queue.pop(0)
            out = cmd[cmd.index("--out") + 1]
            fh = open(f"{out}.eval.log", "w")
            running[subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)] = (cmd, attempt)
            last = time.time()
            log(f"started {out}" + (" (retry)" if attempt else ""))
        else:
            time.sleep(2)


def status(spec: dict) -> str:
    """Read-only progress board from output files and per-episode log lines."""
    out_dir = ROOT / spec["out"]
    e = spec["eval"]
    counts = {"done": 0, "running": 0, "failed": 0, "pending": 0}
    episodes = {"ns3": [0, 0], "lightweight": [0, 0]}
    lines = []
    run_log = out_dir / "run.log"
    starts = ([l for l in run_log.read_text().splitlines() if "evaluation jobs" in l]
              if run_log.exists() else [])
    t0 = time.mktime(time.strptime(starts[-1][:19], "%Y-%m-%d %H:%M:%S")) if starts else None
    recent = 0  # ns-3 episodes finished since the latest driver launch
    for executor, backend in sorted(spec["executors"].items(), key=lambda kv: kv[1] is None):
        kind = "ns3" if backend else "lightweight"
        for scenario in spec["scenarios"]:
            for name, p in policies(spec).items():
                if (p.get("scenarios") and scenario not in p["scenarios"]) or \
                        name in spec.get("exclude", {}).get(scenario, []):
                    continue
                cells = []
                for args, f in result_files(spec, scenario, executor, name):
                    total = int(args[args.index("--episodes") + 1])
                    log = Path(f"{f}.eval.log")
                    text = log.read_text(errors="replace") if log.exists() else ""
                    done = sum(1 for l in text.splitlines() if l.startswith("episode "))
                    if f.exists():
                        state, done = "done", total
                    elif "Traceback" in text:
                        state = "failed"
                    elif log.exists():
                        state = "running"
                    else:
                        state = "pending"
                    counts[state] += 1
                    if kind == "ns3" and t0 and log.exists() and log.stat().st_mtime >= t0:
                        recent += done
                    episodes[kind][0] += done
                    episodes[kind][1] += total
                    mark = {"done": "✓", "failed": "✗", "pending": "·"}.get(state, "")
                    cells.append(f"{done:>2}/{total}{mark}")
                pad = 26 - sum(2 if ord(c) > 0x2E80 else 1 for c in name)  # CJK is double width
                lines.append(f"{scenario:<10} {executor:<12} {name}{' ' * max(pad, 1)}" + "  ".join(cells))
    pid_file = out_dir / "driver.pid"
    alive = False
    if pid_file.exists():
        try:
            os.kill(int(pid_file.read_text()), 0)
            alive = True
        except (OSError, ValueError):
            pass
    done, total = episodes["ns3"]
    head = [f"E10 进度  {time.strftime('%Y-%m-%d %H:%M:%S')}  驱动{'运行中' if alive else '未运行'}",
            f"ns-3 回合 {done}/{total}（{100 * done / max(total, 1):.1f}%），轻量回合 "
            f"{episodes['lightweight'][0]}/{episodes['lightweight'][1]}",
            f"分片：完成 {counts['done']}，运行 {counts['running']}，失败 {counts['failed']}，"
            f"等待 {counts['pending']}"]
    if alive and t0 and recent:
        rate = recent / max(time.time() - t0, 1)
        head.append(f"预计剩余：约 {(total - done) / rate / 3600:.1f} 小时"
                    f"（本次启动以来 {rate * 3600:.0f} 回合/小时）")
    head.append(f"每格：已完成回合/分片回合数（✓ 完成，✗ 失败，· 未开始）；种子偏移 {e['seed_offset']}，"
                f"排空 {e['drain']} 周期")
    return "\n".join(head + [""] + lines)


# ------------------------------------------------------------------- summary
def load_rows(spec, scenario, executor, name) -> list[list[dict]] | None:
    """Rows per run (one list for a single policy, one per training seed for a group)."""
    files = result_files(spec, scenario, executor, name)
    if not all(f.exists() for _, f in files):
        return None
    runs: dict[str, list[dict]] = {}
    for _, f in files:
        data = json.loads(f.read_text())
        if spec["executors"][executor]:
            assert data["config"]["backend"]["type"] == spec["executors"][executor], f
        rows = next(iter(data["results"].values()))["rows"]
        for r in rows:
            r["per"] = r.get("radio_rx_error", 0.0) / max(r.get("radio_tx_start", 0.0), 1.0)
        runs.setdefault(f.name.split(".shard")[0], []).extend(rows)
    return [sorted(r, key=lambda x: x["seed"]) for r in runs.values()]


def seed_avg(runs: list[list[dict]], key: str) -> np.ndarray:
    return np.mean([[r[key] for r in rows] for rows in runs], axis=0)


def fmt_ci(d: np.ndarray, digits: int = 4) -> str:
    m, lo, hi = ci(d)
    return f"{m:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def summarise(spec: dict) -> str:
    e = spec["eval"]
    names = list(policies(spec))
    group = next(iter(spec["groups"]))
    lines = [f"# {spec['name']}", "",
             f"{e['split']} 种子偏移 {e['seed_offset']}，{e['episodes']} 个场景，排空 {e['drain']} 周期。"
             "学习型策略按种子平均后再按场景配对；均值 [95% CI]。", ""]
    for scenario in spec["scenarios"]:
        data = {x: {n: load_rows(spec, scenario, x, n) for n in names} for x in spec["executors"]}
        lines += [f"## 场景 `{scenario}`", "", "| 执行环境 | 策略 | " + " | ".join(k[1] for k in KEYS) + " |",
                  "| --- | --- | " + " | ".join("---" for _ in KEYS) + " |"]
        for x in spec["executors"]:
            for n in names:
                runs = data[x][n]
                if runs is None:
                    continue
                vals = [float(np.nanmean(seed_avg(runs, k))) for k, _, _ in KEYS]
                lines.append(f"| {x} | {n} | " + " | ".join(f.format(v) for v, (_, _, f) in zip(vals, KEYS)) + " |")
        lines.append("")
        ns3 = next(x for x, b in spec["executors"].items() if b)
        light = next(x for x, b in spec["executors"].items() if not b)
        # P1/P2: the method against the baselines inside ns-3
        ref_names = [n for n in names if n != group and n not in spec.get("controls", {})]
        if data[ns3][group]:
            lines += [f"**ns-3 中 {group} 相对各策略的配对差**", "",
                      "| 对照 | " + " | ".join(dict((k, l) for k, l, _ in KEYS)[k] for k in PAIR_KEYS)
                      + " | 可靠性可接受 |", "| --- | " + " | ".join("---" for _ in PAIR_KEYS) + " | --- |"]
            for ref in ref_names:
                if data[ns3][ref] is None:
                    continue
                cells = [fmt_ci(seed_avg(data[ns3][group], k) - seed_avg(data[ns3][ref], k),
                                4 if "delay" not in k else 3) for k in PAIR_KEYS]
                lo = ci(seed_avg(data[ns3][group], "delivery_ratio")
                        - seed_avg(data[ns3][ref], "delivery_ratio"))[1]
                lines.append(f"| {ref} | " + " | ".join(cells) + f" | {'是' if lo >= NI_MARGIN else '否'} |")
            lines.append("")
        # P3: fidelity per policy, and the method's advantage over LQ in both executors
        lines += [f"**执行保真度：ns-3 − 轻量环境（同一策略、同一场景）**", "",
                  "| 策略 | 交付率差 | 平均时延差 | p95 时延差 | ns-3 物理层误包率 | 保真度可接受 |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for n in names:
            a, b = data[ns3][n], data[light][n]
            if a is None or b is None:
                continue
            dd = seed_avg(a, "delivery_ratio") - seed_avg(b, "delivery_ratio")
            per = float(np.mean(seed_avg(a, "per")))
            m, lo, hi = ci(dd)
            ok = max(abs(lo), abs(hi)) <= FIDELITY_DELIVERY and per <= FIDELITY_PER
            control = n in spec.get("controls", {})
            lines.append(f"| {n} | {fmt_ci(dd)} | "
                         f"{fmt_ci(seed_avg(a, 'e2e_delay_mean_s') - seed_avg(b, 'e2e_delay_mean_s'), 3)} | "
                         f"{fmt_ci(seed_avg(a, 'e2e_delay_p95_s') - seed_avg(b, 'e2e_delay_p95_s'), 3)} | "
                         f"{per:.4f} | {'（对照，不判定）' if control else ('是' if ok else '否')} |")
        lines.append("")
        ref = spec["baselines"][0]
        if all(data[x][k] is not None for x in (ns3, light) for k in (group, ref)):
            lines += [f"**{group} 相对 {ref} 的优势在两个执行环境中是否一致**（ns-3 优势 − 轻量优势）", "",
                      "| 指标 | 轻量环境 | ns-3 | 差 |", "| --- | --- | --- | --- |"]
            for k in PAIR_KEYS:
                dl = seed_avg(data[light][group], k) - seed_avg(data[light][ref], k)
                dn = seed_avg(data[ns3][group], k) - seed_avg(data[ns3][ref], k)
                d = 4 if "delay" not in k else 3
                lines.append(f"| {dict((a, l) for a, l, _ in KEYS)[k]} | {fmt_ci(dl, d)} | {fmt_ci(dn, d)} | "
                             f"{fmt_ci(dn - dl, d)} |")
            lines.append("")
    lines += [f"判定：可靠性可接受 = 交付率配对差 95% CI 下界 ≥ {NI_MARGIN}；保真度可接受 = 交付率差的 CI "
              f"落在 ±{FIDELITY_DELIVERY} 内且 ns-3 物理层误包率 ≤ {FIDELITY_PER}（完整 SINR 策略）。"]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--gap", type=float, default=20.0, help="seconds between job starts")
    p.add_argument("--summary", action="store_true")
    p.add_argument("--status", action="store_true", help="progress board (read-only)")
    p.add_argument("--watch", type=float, default=0.0, help="with --status: refresh every N s")
    a = p.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    out = ROOT / spec["out"]
    out.mkdir(parents=True, exist_ok=True)
    if a.status:
        while True:
            board = status(spec)
            print("\033[2J\033[H" + board if a.watch else board, flush=True)
            if not a.watch:
                return
            time.sleep(a.watch)

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with (out / "run.log").open("a") as fh:
            fh.write(line + "\n")

    if not a.summary:
        (out / "driver.pid").write_text(str(os.getpid()))
        jobs = build_jobs(spec)
        log(f"{len(jobs)} evaluation jobs")
        launch([resolve(j) for j in jobs], a.parallel, a.gap, log)
        log("all jobs finished")
    text = summarise(spec)
    (out / "summary.md").write_text(text + "\n")
    log(f"wrote {out / 'summary.md'}")
    print(text)


if __name__ == "__main__":
    main()
