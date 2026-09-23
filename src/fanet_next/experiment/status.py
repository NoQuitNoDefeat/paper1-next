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
        its = [e for e in logs if "ppo/entropy" in e]
        evals = _read_jsonl(run / "eval_log.jsonl")
        last = its[-1] if its else {}
        done = last.get("iteration", 0)
        recent = its[-10:]
        per_it = (sum(e["collect_s"] + e["update_s"] for e in recent) / len(recent)) if recent else float("nan")
        log_file = run.parent / f"{run.name}.log"
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
            "entropy": last.get("ppo/entropy"), "dev_it": ev.get("iteration"),
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


def render(root: Path) -> str:
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
