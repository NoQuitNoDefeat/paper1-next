"""Read-only run dashboard for the terminal (`fanet-next status`)."""

from __future__ import annotations

import json
import subprocess
import time
import unicodedata
from pathlib import Path


def _width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _pad(text: str, width: int, right: bool = False) -> str:
    gap = " " * max(width - _width(text), 0)
    return gap + text if right else text + gap


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a line being written right now
    return out


def _running_dirs() -> set[str]:
    try:
        ps = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return set()
    dirs = set()
    for line in ps.splitlines():
        if "fanet-next" in line and "--run-dir" in line and not line.startswith(("/bin/", "zsh", "bash")):
            parts = line.split()
            dirs.add(str(Path(parts[parts.index("--run-dir") + 1]).resolve()))
    return dirs


def _fmt_time(seconds: float) -> str:
    if seconds != seconds or seconds < 0:
        return "-"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def run_rows(root: Path) -> list[dict]:
    running = _running_dirs()
    rows = []
    for meta in sorted(root.rglob("meta.json")):
        run = meta.parent
        if any(part.startswith("_") for part in run.relative_to(root).parts):
            continue  # e.g. _aborted/
        try:
            m = json.loads(meta.read_text())
        except json.JSONDecodeError:
            continue
        total = int(m.get("training", {}).get("iterations", 0))
        logs = _read_jsonl(run / "train_log.jsonl")
        # primary runs log "ppo/*"; learned-baseline runs (train-baseline) log "progress"
        its = [e for e in logs if "ppo/entropy" in e or ("progress" in e and "iteration" in e)]
        evals = _read_jsonl(run / "eval_log.jsonl")
        last = its[-1] if its else {}
        done = last.get("iteration", 0)
        recent = its[-10:]
        per_it = (sum(e.get("seconds", e.get("collect_s", 0) + e.get("update_s", 0)) for e in recent)
                  / len(recent)) if recent else float("nan")
        log_file = run.parent / f"{run.name}.log"
        if not log_file.exists():
            log_file = run.parent / f"{run.name}.train.log"
        text = log_file.read_text(errors="ignore") if log_file.exists() else ""
        errors = sum(len(e.get("errors") or []) for e in its) + text.count("Traceback")
        is_running = str(run.resolve()) in running
        if is_running:
            state = "运行中"
        elif done >= total and total:
            state = "完成"
        elif "Traceback" in text or "error:" in text:
            state = "出错"
        else:
            state = "已停止"
        sel = run / "selection.json"
        selected = json.loads(sel.read_text())["selected"]["iteration"] if sel.exists() else None
        ev = evals[-1] if evals else {}
        rows.append({
            "run": str(run.relative_to(root)), "state": state, "done": done, "total": total,
            "per_it": per_it, "eta": (total - done) * per_it if is_running else float("nan"),
            "reward": last.get("reward_raw_mean"), "ev": last.get("ppo/explained_var"),
            "entropy": last.get("ppo/entropy", last.get("entropy")), "dev_it": ev.get("iteration"),
            "dev_dr": ev.get("dev/delivery_ratio"), "dev_delay": ev.get("dev/e2e_delay_mean_s"),
            "dev_ontime": ev.get("dev/ontime_1s"), "errors": errors, "selected": selected,
            "log": str(log_file) if log_file.exists() else "",
            "updated": time.strftime("%H:%M:%S", time.localtime((run / "train_log.jsonl").stat().st_mtime))
            if (run / "train_log.jsonl").exists() else "-",
        })
    for qfile in sorted(root.rglob("queue.txt")):
        for line in qfile.read_text().splitlines():
            args = line.split("#", 1)[0].split()
            if "--run-dir" not in args:
                continue
            run = Path(args[args.index("--run-dir") + 1])
            if not run.exists():
                rows.append({"run": str(run.resolve().relative_to(root.resolve())), "state": "排队中",
                             "done": 0, "total": 0, "per_it": float("nan"), "eta": float("nan"),
                             "reward": None, "ev": None, "entropy": None, "dev_it": None,
                             "dev_dr": None, "dev_delay": None, "dev_ontime": None, "errors": 0,
                             "selected": None, "log": "", "updated": "-"})
    return rows


def pipeline_rows(root: Path) -> list[dict]:
    """Evaluation pipelines written by tools/confirm.py (``confirm.log`` + output files)."""
    rows = []
    for log in sorted(root.rglob("confirm.log")):
        lines = log.read_text(errors="ignore").splitlines()
        plan_idx = max((i for i, ln in enumerate(lines) if "evaluations" in ln and "selections" in ln),
                       default=None)
        if plan_idx is None:
            continue
        words = lines[plan_idx].split()
        n_sel, n_eval = int(words[2]), int(words[4])
        start = time.mktime(time.strptime(" ".join(words[:2]), "%Y-%m-%d %H:%M:%S"))
        after = lines[plan_idx + 1:]
        started = [ln.split()[-1] for ln in after if " started " in ln]
        outputs = [p for p in started if p.endswith(".json")]
        done = sum(1 for p in outputs if Path(p).exists() and Path(p).stat().st_mtime >= start)
        sel_done = sum(1 for p in started if not p.endswith(".json"))
        finished = any(" wrote " in ln for ln in after)
        elapsed = time.time() - start
        total = n_sel + n_eval
        progress = done + min(sel_done, n_sel)
        running = 0 if finished else max(len(started) - progress, 0)
        eta = elapsed / progress * (total - progress) if progress and not finished else float("nan")
        rel = log.parent.resolve().relative_to(root.resolve())
        name = f"{root.resolve().name}/{rel}" if str(rel) in (".", "confirm") else str(rel)
        rows.append({"name": name,
                     "state": "完成" if finished else "进行中", "progress": f"{progress}/{total}",
                     "running": str(running), "elapsed": _fmt_time(elapsed if not finished else float("nan")),
                     "eta": _fmt_time(eta), "last": (after[-1][11:19] + " " + after[-1][20:].split("/")[-1][:40])
                     if after else "-"})
    return rows


def _table(head: list[str], body: list[list[str]], right_from: int = 2) -> list[str]:
    widths = [max(_width(x) for x in col) for col in zip(head, *body)] if body else [_width(h) for h in head]
    out = ["  ".join(_pad(h, w) for h, w in zip(head, widths)), "  ".join("-" * w for w in widths)]
    for row in body:
        out.append("  ".join(_pad(x, w, right=i >= right_from) for i, (x, w) in enumerate(zip(row, widths))))
    return out


def render(root: Path) -> str:
    text = _render_runs(root)
    pipes = pipeline_rows(root)
    if pipes:
        text += "\n\n评估流水线（tools/confirm.py）\n" + "\n".join(_table(
            ["流水线", "状态", "进度", "进行中", "已用时间", "预计剩余", "最近事件"],
            [[r["name"], r["state"], r["progress"], r["running"], r["elapsed"], r["eta"], r["last"]]
             for r in pipes]))
    return text


def _render_runs(root: Path) -> str:
    rows = run_rows(root)
    head = ["运行", "状态", "迭代", "秒/迭代", "预计剩余", "训练奖励", "解释方差", "熵",
            "dev@迭代", "dev交付率", "dev时延", "1s送达", "错误", "已选", "更新"]

    def f(v, spec):
        return "-" if v is None else format(v, spec)

    body = [[r["run"], r["state"], f"{r['done']}/{r['total']}" if r["total"] else "-", f(r["per_it"], ".1f"),
             _fmt_time(r["eta"]), f(r["reward"], "+.4f"), f(r["ev"], ".2f"), f(r["entropy"], ".2f"),
             f(r["dev_it"], "d"), f(r["dev_dr"], ".3f"), f(r["dev_delay"], ".3f"),
             f(r["dev_ontime"], ".3f"), str(r["errors"]), f(r["selected"], "d"), r["updated"]]
            for r in rows]
    widths = [max(_width(x) for x in col) for col in zip(head, *body)] if body else [_width(h) for h in head]
    lines = [f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {root}  （共 {len(rows)} 个运行，"
             f"运行中 {sum(r['state'] == '运行中' for r in rows)}，"
             f"排队 {sum(r['state'] == '排队中' for r in rows)}）",
             "  ".join(_pad(h, w) for h, w in zip(head, widths))]
    lines.append("  ".join("-" * w for w in widths))
    for row in body:
        lines.append("  ".join(_pad(x, w, right=i >= 2) for i, (x, w) in enumerate(zip(row, widths))))
    lines.append("dev 列是训练中最近一次 dev 评估（8 回合，排空）；正式比较以 select / eval 的结果为准。")
    return "\n".join(lines)


def main(root: str = "results", watch: float = 0.0) -> None:
    path = Path(root)
    if not watch:
        print(render(path))
        return
    try:
        while True:
            print("\033[2J\033[H" + render(path), flush=True)
            time.sleep(watch)
    except KeyboardInterrupt:
        pass
