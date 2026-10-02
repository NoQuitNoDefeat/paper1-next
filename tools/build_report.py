"""Build the results report page from results/ (tools/report_data.py).

    .venv/bin/python tools/build_report.py      # -> results/report/report.html
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_data import ROOT, collect  # noqa: E402

OUT = ROOT / "results" / "report" / "report.html"


def fmt_pp(x: float) -> str:
    return f"{x * 100:+.1f}"


def main() -> None:
    d = collect()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    sc = {s["id"]: s for s in d["e9"]["scenarios"]}
    lq = {k: v["diffs"]["longest_queue"] for k, v in sc.items()}
    hard = [k for k in sc if k != "default"]

    def rel(k, key):  # relative change of the mean vs longest_queue, percent
        base = sc[k]["means"]["longest_queue"][key]
        return 100 * lq[k][key]["m"] / base

    delay_rng = (min(rel(k, "e2e_delay_mean_s") for k in sc), max(rel(k, "e2e_delay_mean_s") for k in sc))
    p95_rng = (min(rel(k, "e2e_delay_p95_s") for k in sc), max(rel(k, "e2e_delay_p95_s") for k in sc))
    dr_hard = (min(lq[k]["delivery_ratio"]["m"] for k in hard), max(lq[k]["delivery_ratio"]["m"] for k in hard))
    facts = {
        "episodes": d["e9"]["episodes"], "drain": d["e9"]["drain"],
        "seeds_main": d["e9"]["seeds"]["主方法（无摘要）"],
        "delay_lo": f"{-delay_rng[1]:.0f}", "delay_hi": f"{-delay_rng[0]:.0f}",
        "p95_lo": f"{-p95_rng[1]:.0f}", "p95_hi": f"{-p95_rng[0]:.0f}",
        "dr_hard_lo": f"{dr_hard[0] * 100:.1f}", "dr_hard_hi": f"{dr_hard[1] * 100:.1f}",
        "dr_default": fmt_pp(lq["default"]["delivery_ratio"]["m"]),
        "dr_default_lo": fmt_pp(lq["default"]["delivery_ratio"]["lo"]),
        "pure_dev": " / ".join(f"{x:.3f}" for x in d["dev"]["pure_ppo_dr"]),
        "lq_dev": f"{d['dev']['lq_dr']:.3f}", "imit_dev": f"{d['dev']['imit_dr']:.3f}",
        "pure_test_default": f"{sc['default']['means']['纯PPO（从零训练）']['delivery_ratio']:.3f}",
        "lq_test_default": f"{sc['default']['means']['longest_queue']['delivery_ratio']:.3f}",
        "commit": commit, "built": time.strftime("%Y-%m-%d"),
    }
    e10 = {x["id"]: x for x in d["e10"]["scenarios"]}
    full = [f for x in e10.values() for f in x["fid"].values() if not f["control"]]
    ctrl = e10["default"]["fid"][d["e10"]["control"]]
    ref_lq = e10["default"]["fid"][d["e10"]["ref"]]
    c11 = d["e11"]["compare"]
    ext = [n for n in c11["diffs"] if n != "仅模仿"]
    learned = ("Zhao-GCN", "GRLinQ", "独立决定PPO")
    ok_all = all(c11["diffs"][n][v]["delivery_ratio"]["lo"] >= -0.005 for n in c11["diffs"] for v in c11["scenarios"])
    sig = sum(c11["diffs"][n][v]["delivery_ratio"]["lo"] > 0 for n in ext for v in c11["scenarios"])
    bp = [c11["diffs"][n][v] for n in ("backpressure", "backpressure_opt") for v in c11["scenarios"]]
    facts.update({
        "e11_n_ext": len(ext), "e11_episodes": c11["episodes"], "e11_cells": len(ext) * len(c11["scenarios"]),
        "e11_sig": sig, "e11_all_ok": "全部" if ok_all else "并非全部",
        "e11_learned_lo": f"{min(c11['diffs'][n][v]['delivery_ratio']['m'] for n in learned for v in c11['scenarios']) * 100:.1f}",
        "e11_learned_hi": f"{max(c11['diffs'][n][v]['delivery_ratio']['m'] for n in learned for v in c11['scenarios']) * 100:.1f}",
        "e11_bp_dr_lo": f"{min(x['delivery_ratio']['m'] for x in bp) * 100:.1f}",
        "e11_bp_dr_hi": f"{max(x['delivery_ratio']['m'] for x in bp) * 100:.1f}",
        "e11_bp_delay_lo": f"{min(x['e2e_delay_mean_s']['m'] for x in bp):.2f}",
        "e11_bp_delay_hi": f"{max(x['e2e_delay_mean_s']['m'] for x in bp):.2f}",
        "e11_oneshot_lo": f"{min(c11['diffs']['独立决定PPO'][v]['delivery_ratio']['m'] for v in c11['scenarios']) * 100:.1f}",
        "e11_oneshot_hi": f"{max(c11['diffs']['独立决定PPO'][v]['delivery_ratio']['m'] for v in c11['scenarios']) * 100:.1f}",
    })
    facts.update({
        "e10_episodes": d["e10"]["episodes"], "e10_drain": d["e10"]["drain"],
        "e10_dr_default": fmt_pp(e10["default"]["adv"]["ns3"]["delivery_ratio"]["m"]),
        "e10_dr_default_lo": fmt_pp(e10["default"]["adv"]["ns3"]["delivery_ratio"]["lo"]),
        "e10_dr_high": fmt_pp(e10["load_high"]["adv"]["ns3"]["delivery_ratio"]["m"]),
        "e10_fid_max": f"{max(max(abs(f['delivery_ratio']['lo']), abs(f['delivery_ratio']['hi'])) for f in full) * 100:.2f}",
        "e10_per_max": f"{max(f['per'] for f in full) * 100:.2f}",
        "e10_ctrl_per": f"{ctrl['per'] * 100:.1f}", "e10_ctrl_failed": f"{ctrl['failed'] * 100:.0f}",
        "e10_ctrl_dr": f"{ctrl['dr_ns3']:.3f}", "e10_lq_dr": f"{ref_lq['dr_ns3']:.3f}",
    })
    e12 = d["e12b"]
    lqv = e12["versus"]["longest_queue"]
    drop = [e12["effect"][n]["default"]["delivery_ratio"]["m"] for n in e12["effect"]]
    facts.update({
        "e12b_episodes": e12["episodes"],
        "e12b_dr_default": fmt_pp(lqv["c2_default"]["delivery_ratio"]["m"]),
        "e12b_dr_default_lo": fmt_pp(lqv["c2_default"]["delivery_ratio"]["lo"]),
        "e12b_delay_default": f"{-lqv['c2_default']['e2e_delay_mean_s']['m']:.2f}",
        "e12b_dr_n24": fmt_pp(lqv["c2_nodes24_same_density"]["delivery_ratio"]["m"]),
        "e12b_dr_load": fmt_pp(lqv["c2_load_5_15"]["delivery_ratio"]["m"]),
        "e12b_all_ok": "都" if all(c["delivery_ratio"]["lo"] >= -0.005 for v in e12["versus"].values()
                                  for c in v.values()) else "并非都",
        "e12b_drop_lo": f"{-max(drop) * 100:.0f}", "e12b_drop_hi": f"{-min(drop) * 100:.0f}",
    })
    e13 = d["e13"]
    zs, rt = e13["主方法"]["longest_queue"], e13["主方法·重训"]["longest_queue"]
    fid = [abs(x) for row in e13["ns3"] for k, v in row.items() if k.startswith("fid_") for x in (v["lo"], v["hi"])]
    ucsb = e13["ucsb"][0]["models"]
    rssi = e13["rssi"]["primary"][1]
    w, acf = rssi["sigma_within_db"], rssi["within_autocorr"]
    facts.update({
        "e13_zs_dr": fmt_pp(zs["flock"]["delivery_ratio"]["m"]), "e13_zs_lo": fmt_pp(zs["flock"]["delivery_ratio"]["lo"]),
        "e13_zs_delay": f"{-zs['flock']['e2e_delay_mean_s']['m']:.2f}",
        "e13_rt_dr": fmt_pp(rt["flock"]["delivery_ratio"]["m"]), "e13_rt_lo": fmt_pp(rt["flock"]["delivery_ratio"]["lo"]),
        "e13_f4": f"{-zs['fold_4mps']['delivery_ratio']['m'] * 100:.1f}",
        "e13_n30": f"{-e13['extra']['longest_queue']['n30']['delivery_ratio']['m'] * 100:.1f}",
        "e13_ns3_eps": e13["ns3_episodes"], "e13_ns3_fid": f"{max(fid) * 100:.2f}",
        "ucsb_mae_rician": f"{ucsb['rician']['test_sector_mae']:.3f}", "ucsb_mae_step": f"{ucsb['step']['test_sector_mae']:.3f}",
        "ucsb_sigma": f"{ucsb['lognormal']['params']['sigma_db']:.1f}",
        "rssi_total": f"{rssi['sigma_total_db']:.1f}",
        "rssi_slow": f"{(rssi['sigma_link_offset_db'] ** 2 + w * w * acf['2s']['corr']) ** 0.5:.1f}",
        "rssi_fast": f"{(w * w * (1 - acf['0.5s']['corr'])) ** 0.5:.1f}",
        "e13b_tag": "完成" if e13["flock_spec"] == "e13b" else "进行中",
        "e13b_note": "，包括同样在群集数据上重新训练的全部学习基线（E13b）" if e13["flock_spec"] == "e13b" else "",
        "e14_tag": "完成" if d["e14"] else "进行中", "e15_tag": "完成" if d["e15"] else "进行中",
    })
    if d["e14"]:
        x = d["e14"]
        lq_b, lq_o = x["主方法·宽"]["longest_queue"], x["主方法"]["longest_queue"]
        facts.update({
            "e14_f30_orig": fmt_pp(lq_o["flock30"]["delivery_ratio"]["m"]),
            "e14_f30_broad": fmt_pp(lq_b["flock30"]["delivery_ratio"]["m"]),
            "e14_slow_orig": fmt_pp(lq_o["slow_compact"]["delivery_ratio"]["m"]),
            "e14_slow_broad": fmt_pp(lq_b["slow_compact"]["delivery_ratio"]["m"]),
            "e14_speed_orig": fmt_pp(lq_o["speed_0_5"]["delivery_ratio"]["m"]),
            "e14_speed_broad": fmt_pp(lq_b["speed_0_5"]["delivery_ratio"]["m"]),
            "e14_default": fmt_pp(x["broad_vs_orig"]["主方法·宽"]["default"]["delivery_ratio"]["m"]),
        })
    if d["e15"]:
        y = d["e15"]
        eff = [r["delivery_ratio"]["m"] for n in y["effect"].values() for r in n.values()]
        lq = y["versus"]["longest_queue"]
        facts.update({
            "e15_eff_lo": f"{min(eff) * 100:.0f}", "e15_eff_hi": f"{max(eff) * 100:.0f}",
            "e15_default": fmt_pp(lq["c2s5_default"]["delivery_ratio"]["m"]),
            "e15_n24": fmt_pp(lq["c2s5_nodes24_same_density"]["delivery_ratio"]["m"]),
            "e15_flock": fmt_pp(lq["flock_c2s5"]["delivery_ratio"]["m"]),
        })
    html = TEMPLATE.replace("    <!--E14-->", E14_CLAIM if d["e14"] else "")
    html = html.replace("    <!--E15-->", E15_CLAIM if d["e15"] else "")
    for k, v in facts.items():
        html = html.replace("{{" + k + "}}", str(v))
    html = html.replace("/*DATA*/null", json.dumps(d, ensure_ascii=False))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


E14_CLAIM = """    <li><strong>探索一：训练分布覆盖低速和紧凑机群（E14）。</strong>用“原分布加低速紧凑分布”的混合分布重新训练后，在群集 30 架上，主方法相对 LQ 的交付率差从 {{e14_f30_orig}} 变为 {{e14_f30_broad}} 个百分点，满足可靠性标准。在合成的低速紧凑场景中，这个差从 {{e14_slow_orig}} 变为 {{e14_slow_broad}}，在合成低速场景中从 {{e14_speed_orig}} 变为 {{e14_speed_broad}}，差距约缩小一半，但仍略低于标准。原场景不退步（{{e14_default}}），代价是平均时延略有升高（图 12、图 13）。</li>"""
E15_CLAIM = """    <li><strong>探索二：5 dB 阴影作为控制变量（E15）。</strong>阴影已知、在 dB 上零均值时，主要作用是让网络更连通：所有方法的交付率都升高 {{e15_eff_lo}}–{{e15_eff_hi}} 个百分点。网络不再拥堵后，主方法相对 LQ 的交付率优势消失：默认场景 {{e15_default}}，24 节点 {{e15_n24}}，群集 {{e15_flock}} 个百分点，后两者的区间下界越过了标准。主方法的时延仍然更低，ns-3 结论一致（图 14）。</li>"""

TEMPLATE = r"""<title>双图微步调度实验</title>
<meta name="description" content="UAV/FANET 集中式 MAC 调度：双图表示 + 微步 PPO 的定稿方法、E1–E9 实验结果、ns-3 物理层验证（E10）、12 个外部基线（E11）、实测参数信道（E12b）、真实轨迹与第三方移动模型（E13），以及 E13b–E15 后续实验">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Noto+Sans+SC:wght@400;500;700&family=Noto+Serif+SC:wght@600;700&display=swap">
<style>
:root{
  --bg:#f2f4f5; --surface:#fcfcfc; --ink:#111a22; --ink-2:#4b5764; --muted:#6f7a86;
  --grid:#e3e7ea; --axis:#c3cad1; --rule:#d9dee3;
  --accent:#2a78d6; --accent-wash:rgba(42,120,214,.10); --context:#8b95a0;
  --pos:#2a78d6; --neg:#e34948;
  --serif:"Noto Serif SC","Songti SC","STSong",serif;
  --sans:"IBM Plex Sans","Noto Sans SC","PingFang SC","Hiragino Sans GB","Microsoft YaHei",system-ui,sans-serif;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme:dark;
    --bg:#0f1316; --surface:#161b1f; --ink:#eef2f5; --ink-2:#b8c2cb; --muted:#8a95a0;
    --grid:#252c32; --axis:#3a434b; --rule:#2a3238;
    --accent:#3987e5; --accent-wash:rgba(57,135,229,.14); --context:#6b7580;
    --pos:#3987e5; --neg:#e66767;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --bg:#0f1316; --surface:#161b1f; --ink:#eef2f5; --ink-2:#b8c2cb; --muted:#8a95a0;
  --grid:#252c32; --axis:#3a434b; --rule:#2a3238;
  --accent:#3987e5; --accent-wash:rgba(57,135,229,.14); --context:#6b7580;
  --pos:#3987e5; --neg:#e66767;
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:15px;line-height:1.7;
  padding-inline:20px;padding-block:32px 72px}
.page{max-width:1040px;margin:0 auto}
.prose{max-width:44em}
header{border-bottom:1px solid var(--rule);padding-bottom:24px;margin-bottom:8px}
.eyebrow{font-size:12px;letter-spacing:.08em;color:var(--muted);text-transform:uppercase}
h1{font-family:var(--serif);font-weight:700;font-size:clamp(26px,4.2vw,36px);line-height:1.3;margin:.35em 0 .4em;text-wrap:balance}
h2{font-family:var(--serif);font-weight:600;font-size:22px;line-height:1.4;margin:56px 0 12px;text-wrap:balance}
h3{font-size:15px;font-weight:600;margin:28px 0 6px}
p{margin:.6em 0}
.lede{font-size:17px;color:var(--ink-2);max-width:40em}
.meta{font-size:13px;color:var(--muted);display:flex;flex-wrap:wrap;gap:6px 18px;margin-top:14px}
.meta code{font-family:var(--sans);color:var(--ink-2)}
ul.claims{padding-left:1.1em;margin:.4em 0}
ul.claims li{margin:.35em 0}
strong{font-weight:600}
.num{font-variant-numeric:tabular-nums}
figure{margin:18px 0 8px;background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:18px 18px 14px;position:relative}
figcaption{font-size:13px;color:var(--ink-2);margin-top:10px}
figcaption b{color:var(--ink);font-weight:600}
.fig-title{font-size:14px;font-weight:600;margin:0 0 2px}
.fig-sub{font-size:12.5px;color:var(--muted);margin:0 0 12px}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:18px 22px}
.panel h4{font-size:13px;font-weight:600;margin:0;display:flex;justify-content:space-between;gap:8px}
.panel h4 span{font-weight:400;color:var(--muted)}
svg{display:block;width:100%;overflow:visible}
svg text{font-family:var(--sans);fill:var(--muted);font-size:11.5px}
svg .lbl{fill:var(--ink-2);font-size:12px}
svg .val{fill:var(--ink);font-size:12px;font-variant-numeric:tabular-nums}
svg .tick{font-variant-numeric:tabular-nums}
.grid line{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--axis);stroke-width:1}
.zero{stroke:var(--ink-2);stroke-width:1}
.controls{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 14px}
.controls button{font:inherit;font-size:13px;color:var(--ink-2);background:transparent;border:1px solid var(--rule);
  border-radius:999px;padding:4px 12px;cursor:pointer}
.controls button[aria-pressed="true"]{background:var(--ink);border-color:var(--ink);color:var(--surface)}
.controls button:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:12.5px;color:var(--ink-2);margin:2px 0 10px}
.legend i{display:inline-block;vertical-align:middle;margin-right:6px}
.key-line{width:16px;height:2px;background:var(--accent)}
.key-line.ctx{background:var(--context)}
.key-line.ref{background:var(--ink-2);height:1px}
.key-dot{width:9px;height:9px;border-radius:50%;background:var(--accent);box-shadow:0 0 0 2px var(--surface)}
.tip{position:absolute;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--rule);
  border-radius:8px;padding:8px 10px;font-size:12px;line-height:1.5;box-shadow:0 4px 14px rgba(0,0,0,.12);
  min-width:150px;z-index:5;font-variant-numeric:tabular-nums}
.tip .t{color:var(--muted);margin-bottom:2px}
.tip .r{display:flex;align-items:center;gap:8px;justify-content:space-between}
.tip .r span:first-child{color:var(--ink-2)}
.tip .r b{font-weight:600}
.table-wrap{overflow-x:auto;margin:10px 0 4px}
table{border-collapse:collapse;width:100%;font-size:13px;min-width:560px}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--rule);font-variant-numeric:tabular-nums;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{font-weight:500;color:var(--muted);font-size:12px;border-bottom-color:var(--axis)}
tr.main td{background:var(--accent-wash);font-weight:600}
td .ci{color:var(--muted);font-weight:400;font-size:11.5px;margin-left:4px}
.heat td.cell{text-align:center;min-width:92px}
.heat td.cell .ci{display:block;margin:0}
.heat td.ns{color:var(--muted)}
.swatches{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:12px;color:var(--ink-2);margin-top:8px}
.swatches i{display:inline-block;width:14px;height:10px;border-radius:2px;margin-right:6px;vertical-align:middle}
.timeline{display:grid;gap:0;border-top:1px solid var(--rule)}
.step{display:grid;grid-template-columns:64px 1fr;gap:4px 18px;padding:16px 0;border-bottom:1px solid var(--rule)}
.step .id{font-family:var(--serif);font-weight:700;font-size:18px;color:var(--ink-2)}
.step h3{margin:0 0 4px;font-size:15px}
.step p{margin:.2em 0;color:var(--ink-2);font-size:14px}
.step .out{color:var(--ink)}
.tag{display:inline-block;font-size:11.5px;padding:1px 8px;border-radius:999px;border:1px solid var(--rule);color:var(--ink-2);margin-left:6px;vertical-align:1px}
pre.mermaid{background:transparent;margin:0;text-align:center}
.flow{overflow-x:auto}
details{margin:8px 0}
summary{cursor:pointer;color:var(--ink-2);font-size:13.5px}
code{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px}
pre.cmd{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:12px 14px;overflow-x:auto;margin:10px 0}
.note{font-size:13px;color:var(--muted)}
@media (max-width:560px){ .step{grid-template-columns:1fr} body{padding-inline:16px} }
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

<div class="page">
<header>
  <div class="eyebrow">paper1-next · UAV/FANET MAC 调度 · 实验报告</div>
  <h1>双图微步调度：定稿方法与测试集结果</h1>
  <p class="lede">在从未使用过的测试集上，定稿方法在 5 个场景族中都没有比“最长队列优先”少送包，平均时延低 {{delay_lo}}–{{delay_hi}}%，p95 时延低 {{p95_lo}}–{{p95_hi}}%。在更密、更大、更拥堵或信道有误差的场景中，最终交付率还高出 {{dr_hard_lo}}–{{dr_hard_hi}} 个百分点。在 ns-3 包级物理层上重跑 {{e10_episodes}} 个新测试场景，这些结论不变。与 {{e11_n_ext}} 个外部基线（含 3 个前人的学习方法和 2 个单周期精确最优解）相比，它在 5 个场景族中的交付率都最高。在按实测文献标定的空空信道上（E12b），以及在真实无人机群集轨迹和第三方移动模型上（E13），可靠性判定仍然成立，平均时延仍然更低。真实群集飞得慢、队形紧凑，零样本时在这类场景有弱点；在数据上重新训练后，交付率在全部策略中最高。</p>
  <div class="meta">
    <span>测试：{{episodes}} 个场景 × 5 个场景族，排空 {{drain}} 周期</span>
    <span>主方法 {{seeds_main}} 个训练种子</span>
    <span>ns-3 验证：{{e10_episodes}} 个新场景 × 2 个场景族</span>
    <span>外部基线：{{e11_n_ext}} 个，{{e11_episodes}} 个新场景 × 5 个场景族</span>
    <span>实测参数信道：{{e12b_episodes}} 个新场景 × 3 个场景族</span>
    <span>真实群集轨迹：3 次飞行 × 32 个场景</span>
    <span>代码 <code>{{commit}}</code></span>
    <span>生成于 {{built}}</span>
  </div>
</header>

<section>
  <h2>结论</h2>
  <ul class="claims prose">
    <li><strong>可靠性不劣于基线，时延显著更低。</strong>预先登记的判定在 5 个场景族中全部成立：交付率配对差的 95% 置信区间下界都不低于 −0.5 个百分点，平均时延和 p95 时延的差都显著小于 0。默认场景下交付率差为 {{dr_default}} 个百分点（下界 {{dr_default_lo}}）。</li>
    <li><strong>条件越难，优势越大。</strong>在 4 个困难场景族中，交付率和时延同时改善。这些场景都没有参与训练，属于零样本泛化。</li>
    <li><strong>模仿预热是必要的，改进来自 PPO 微调。</strong>从零训练的 PPO 靠大量丢包换取低时延（默认场景交付率 {{pure_test_default}}，基线 {{lq_test_default}}）。只模仿、不做 PPO 的模型与教师持平。</li>
    <li><strong>在 ns-3 物理层上成立。</strong>用 ns-3 的真实信号叠加、Shannon 判决和帧内连续运动执行同一批计划，定稿方法的优势与训练环境中几乎相同（交付率差：默认场景 {{e10_dr_default}} 个百分点，下界 {{e10_dr_default_lo}}；高负载 {{e10_dr_high}} 个百分点）。满足完整 SINR 的策略在两种执行下的交付率相差不超过 {{e10_fid_max}} 个百分点，物理层误包率不超过 {{e10_per_max}}%（图 3）。</li>
    <li><strong>优于前人的学习方法，“逐条选择”的设计必要。</strong>相对 {{e11_n_ext}} 个外部基线，5 个场景族中的可靠性判定{{e11_all_ok}}成立，{{e11_cells}} 个比较中有 {{e11_sig}} 个交付率显著更高。相对 3 个前人的学习方法，交付率高 {{e11_learned_lo}}–{{e11_learned_hi}} 个百分点，时延也更低。与主方法只差动作形式的“独立决定”版本，交付率低 {{e11_oneshot_lo}}–{{e11_oneshot_hi}} 个百分点（图 4）。</li>
    <li><strong>与反压是一组取舍。</strong>两个反压基线的交付率比主方法低 {{e11_bp_dr_lo}}–{{e11_bp_dr_hi}} 个百分点，但平均时延低 {{e11_bp_delay_lo}}–{{e11_bp_delay_hi}} s；它们还额外用到了“每个包发往哪里”的信息。按交付率优先，主方法更合适。每周期精确求解最大权反而交付率更低：单周期最优不等于长期最好。</li>
    <li><strong>尾部时延仍有差距。</strong>HOL 加权基线的 p95 时延更低，但交付率显著更低。按“交付率优先”的原则，定稿方法更合适。</li>
    <li><strong>按实测文献标定的信道下仍然成立（E12b）。</strong>3 个场景族中，主方法相对 4 个对照的可靠性判定{{e12b_all_ok}}成立。相对 LQ，默认场景的交付率差为 {{e12b_dr_default}} 个百分点（下界 {{e12b_dr_default_lo}}），平均时延低 {{e12b_delay_default}} s；24 节点时交付率差为 {{e12b_dr_n24}} 个百分点。留出衰落裕量后，单跳可用距离从 161 m 缩到 118 m，默认场景下所有方法的交付率都下降 {{e12b_drop_lo}}–{{e12b_drop_hi}} 个百分点，排名不变（图 9）。</li>
    <li><strong>真实群集轨迹与第三方移动模型（E13）。</strong>在群集 30 的 3 次真实飞行（96 个场景）上，零样本主方法的交付率与 LQ 持平（{{e13_zs_dr}} 个百分点，下界 {{e13_zs_lo}}），平均时延低 {{e13_zs_delay}} s；在 BonnMotion 的两种移动模型上也满足可靠性标准。在群集数据上重新训练后，相对 LQ 的交付率差为 {{e13_rt_dr}} 个百分点（下界 {{e13_rt_lo}}），交付率在全部策略中最高{{e13b_note}}。ns-3 复核结论一致（图 10、图 11）。</li>
    <li><strong>低速、紧凑机群是零样本的弱点。</strong>真实群集的拓扑变化比训练场景慢 4–7 倍。在 4 m/s 的飞行上，零样本主方法的交付率比 LQ 低 {{e13_f4}} 个百分点；30 架时低 {{e13_n30}} 个百分点，没有达到可靠性标准，但时延仍然更低。</li>
    <li><strong>链路模型与实测收包相符，但实测起伏更大。</strong>在 UCSB 空地实测中，“门限加衰落”模型对各距离段收包率的预测误差为 {{ucsb_mae_rician}}，无衰落的门限模型为 {{ucsb_mae_step}}。用 RSSI 分解起伏：约 {{rssi_slow}} dB 是慢变部分，调度器可以测到；约 {{rssi_fast}} dB 是快变部分，相当于 K ≈ 5 dB 的莱斯衰落，比本模型采用的 K = 10 dB 更不稳定。</li>
    <!--E14-->
    <!--E15-->
  </ul>

  <figure id="fig-forest">
    <p class="fig-title">图 1　定稿方法相对 longest_queue 的配对差</p>
    <p class="fig-sub">点为均值，横线为 95% 置信区间。每个场景先对 5 个训练种子取平均，再按 {{episodes}} 个测试场景配对。</p>
    <div class="panels" id="forest"></div>
    <figcaption><b>读法：</b>交付率向右为好，时延向左为好。横线不跨过零线，说明差异显著。</figcaption>
  </figure>
</section>

<section>
  <h2>交付率与时延：不让低时延掩盖丢包</h2>
  <p class="prose">平均时延和 p95 时延只统计送达的包，丢得越多，剩下的包往往越“好送”。所以每个时延都要和最终交付率、2 秒内送达率一起看。2 秒内送达率的分母是全部产生的包，丢掉的包算作未按时送达。</p>
  <figure id="fig-dots">
    <p class="fig-title">图 2　各策略在测试集上的表现</p>
    <div class="controls" role="group" aria-label="选择场景族" id="scen-buttons"></div>
    <div class="legend"><span><i class="key-dot"></i>定稿方法</span><span><i class="key-dot" style="background:var(--context)"></i>对照与基线</span></div>
    <div class="panels" id="dots"></div>
    <figcaption><b>读法：</b>行按最终交付率从高到低排列。纯 PPO 和 random 的时延看起来低，是因为丢掉了大量难送的包。</figcaption>
  </figure>
  <details>
    <summary>查看该场景族的完整数据表</summary>
    <div class="table-wrap"><table id="means-table"></table></div>
  </details>
</section>

<section>
  <h2>ns-3 物理层验证（E10）</h2>
  <p class="prose">训练环境按“每周期完整累计 SINR、整窗同时发送、周期内信道冻结”来判定链路是否成功。E10 把同一批计划交给 ns-3.48 执行，检验这套判定是否可靠：DATA 帧经 Spectrum 信道发送，干扰随各链路开始和结束实时变化，由 Shannon 误码模型判决，计入帧头和传播时延；节点在周期内连续运动，每个包开始接收时按当前距离重算信道；送达由集中式控制面确认。测试集取 {{e10_episodes}} 个此前没用过的场景，排空 {{e10_drain}} 周期，每个场景用两种执行环境各跑一遍。</p>
  <figure id="fig-e10">
    <p class="fig-title">图 3　定稿方法相对 longest_queue 的优势：训练环境与 ns-3 对比</p>
    <p class="fig-sub">同一批 {{e10_episodes}} 个新测试场景；点为均值，横线为 95% 置信区间。每个场景先对 5 个训练种子取平均，再配对。</p>
    <div class="legend"><span><i class="key-dot"></i>ns-3 物理层执行</span><span><i class="key-dot" style="background:var(--context)"></i>训练用轻量环境</span></div>
    <div class="panels" id="e10-forest"></div>
    <figcaption><b>读法：</b>同一场景的两行几乎重合，说明优势不是训练环境的产物。交付率向右为好，时延向左为好。</figcaption>
  </figure>
  <h3>执行保真度：同一策略在 ns-3 与训练环境中的差别</h3>
  <p class="prose">正向对照故意只检查两两干扰、忽略累计干扰。它在 ns-3 中整链失败 {{e10_ctrl_failed}}%，物理层误包率 {{e10_ctrl_per}}%，交付率 {{e10_ctrl_dr}}（完整 SINR 的 longest_queue 为 {{e10_lq_dr}}）。这说明 ns-3 能发现违反干扰约束的计划；训练环境对这类失败的判定也与 ns-3 一致。</p>
  <div class="table-wrap"><table id="e10-table"></table></div>
  <p class="note">交付率差 = ns-3 − 轻量环境，单位为个百分点，方括号内为 95% 置信区间。误包率 = ns-3 中接收失败的 DATA 帧占全部发送帧的比例。预先登记的保真度标准：交付率差的置信区间落在 ±1 个百分点内，且误包率 ≤ 1%。</p>
</section>

<section>
  <h2>与 12 个外部基线的比较（E11）</h2>
  <p class="prose">对照包括：每周期精确求解的最大权和反压（MILP），经典启发式（longest_queue、HOL 加权、最老队首、反压贪心、LQ + 局部搜索、空间 TDMA），前人的学习方法（Zhao 等 TWC 2023 的 GCN、GRLinQ 式迭代图强化学习），以及与主方法只差动作形式的“独立决定”PPO。学习基线的训练预算和检查点选择规则都与主方法相同。所有计划都经过同一个完整 SINR 控制器，保证可行。测试集取 {{e11_episodes}} 个此前没用过的场景，方案和判定规则在开跑前登记。</p>
  <figure id="fig-e11">
    <p class="fig-title">图 4　主方法相对各基线的配对差（5 个场景族）</p>
    <p class="fig-sub">每格：主方法减基线，均值与 95% 置信区间。蓝色表示主方法显著更好，红色表示显著更差，灰色表示不显著。</p>
    <div class="controls" role="group" aria-label="选择指标" id="e11-buttons"></div>
    <div class="table-wrap"><table class="heat" id="e11-table"></table></div>
    <div class="swatches"><span><i style="background:color-mix(in oklab,var(--pos) 45%,var(--surface))"></i>主方法显著更好</span><span><i style="background:color-mix(in oklab,var(--neg) 45%,var(--surface))"></i>主方法显著更差</span><span><i style="background:var(--surface);border:1px solid var(--rule)"></i>不显著</span></div>
  </figure>
  <figure id="fig-e11s">
    <p class="fig-title">图 5　负载、规模、速度扫描</p>
    <p class="fig-sub">每个扫描点 32 个新场景。32、48 节点时 MILP 单次求解已超过 0.6 s，精确解基线不参与（–）。</p>
    <div class="controls" role="group" aria-label="选择指标" id="e11s-buttons"></div>
    <div class="table-wrap"><table class="heat" id="e11s-table"></table></div>
    <figcaption><b>读法：</b>规模越大、负载越重，主方法的交付率优势越大。节点几乎静止（速度 0–5 m/s，超出训练分布）时，主方法交付率比 LQ 低约 0.3 个百分点，没有达到预登记的可靠性标准。</figcaption>
  </figure>
  <h3>单周期近似比与决策时间</h3>
  <p class="prose">近似比 = 计划的单周期最大权 ÷ 同一状态下的 MILP 最优值（链路调度文献的常用指标）。主方法的近似比低于 longest_queue，交付率却最高：它用每周期的一部分权重换取长期的交付率与时延。精确最大权在 24 节点时平均每周期要一百多毫秒，超过 20 ms 的周期。</p>
  <div class="table-wrap"><table id="e11-opt"></table></div>
  <div id="e11-ns3-wrap"></div>
</section>

<section>
  <h2>按实测文献标定的信道（E12、E12b）</h2>
  <p class="prose">原来的合成信道没有衰落，调度时看到的信道就是执行时的信道。E12 按空空测量文献改用实测参数：路径损耗指数 2.2，发射功率相应调整，使单跳距离仍为 161 m；莱斯块衰落 K = 10 dB，每对节点每个周期独立抽样。调度器只知道平均增益，按 10% 中断概率留 3.0 dB 衰落裕量，路由也只用留出裕量后仍能用的链路。E12 运行时，路由仍按无裕量的门限选路；修正后在新的测试种子上重跑，即 E12b。以下为 E12b 的结果，方法全部冻结，没有重新训练。</p>
  <figure id="fig-e12b">
    <p class="fig-title">图 9　实测参数信道（C2）下主方法相对各基线的配对差</p>
    <p class="fig-sub">每个场景族 {{e12b_episodes}} 个新测试场景；主方法减基线，均值与 95% 置信区间。</p>
    <div class="controls" role="group" aria-label="选择指标" id="e12b-buttons"></div>
    <div class="table-wrap"><table class="heat" id="e12b-table"></table></div>
    <figcaption><b>读法：</b>蓝色表示主方法显著更好，红色表示显著更差，灰色表示不显著。低负载下主方法相对 LQ 的交付率差为 {{e12b_dr_load}} 个百分点，差异显著，但仍在可靠性标准以内。</figcaption>
  </figure>
  <h3>信道变化本身的影响：同一批场景上 C2 与 C0 的对比</h3>
  <p class="prose">单一速率下留出衰落裕量后，无干扰可用距离从 161 m 缩到 118 m，网络连通性下降，所有方法的交付率都明显降低。各方法的降幅相近，排名不变。真实系统会让弱链路降速继续使用；本模型只有单一速率，这是绝对性能下降的主要原因之一。</p>
  <div class="table-wrap"><table id="e12b-effect"></table></div>
  <p class="note">每格：C0（原信道）→ C2 的平均交付率，括号内为配对差（个百分点）及 95% 置信区间。</p>
  <h3>敏感性：路径损耗指数、莱斯 K、中断目标</h3>
  <p class="prose">用默认场景族的前 32 个场景。中断目标越严，裕量越大，所有方法的交付率越低；每个条件下主方法都满足可靠性标准。</p>
  <div class="table-wrap"><table id="e12b-sens"></table></div>
</section>

<section>
  <h2>真实轨迹与第三方移动模型（E13）</h2>
  <p class="prose">公开数据中，没有哪次实验同时记录了多机轨迹、业务和收发记录，所以数据按用途分两类：A 类提供节点运动，用于训练和测试；B 类用来检验环境的链路模型。</p>
  <ul class="claims prose">
    <li>A 类用 Vásárhelyi 等 2018 年的 30 架无人机真实群集飞行（CC0），业务、信道和执行由模型补齐。按测试飞行分 3 折：每折用另外两次飞行重新训练，在开发飞行上选检查点。</li>
    <li>第三方移动模型用 BonnMotion 生成的 RPGM（组群移动）和高斯-马尔可夫轨迹。</li>
    <li>B 类用 UCSB 的 2.4 GHz 空地实测收包记录。</li>
    <li>群集 30 的单跳距离取 66 m，使拓扑统计与合成场景相当；路径损耗指数 2.5，莱斯 K = 10 dB。</li>
  </ul>
  <figure id="fig-e13">
    <p class="fig-title">图 10　零样本主方法相对各策略（真实轨迹与 BonnMotion）</p>
    <p class="fig-sub">群集 30 每次测试飞行 32 个场景，三次合并共 96 个；BonnMotion 每个模型 64 个场景。“· 重训”表示在该数据集的训练部分上重新训练的版本。</p>
    <div class="controls" role="group" aria-label="选择指标" id="e13-buttons"></div>
    <div class="table-wrap"><table class="heat" id="e13-table"></table></div>
    <figcaption><b>读法：</b>真实群集飞得慢、队形紧凑，拓扑变化比训练场景慢 4–7 倍。在速度最低的 4 m/s 飞行上，零样本主方法的交付率显著低于 LQ。</figcaption>
  </figure>
  <figure id="fig-e13rt">
    <p class="fig-title">图 11　在数据集上重新训练的主方法相对各策略</p>
    <p class="fig-sub">同一批场景。每折的模型只用该折以外的两次飞行训练。</p>
    <div class="controls" role="group" aria-label="选择指标" id="e13rt-buttons"></div>
    <div class="table-wrap"><table class="heat" id="e13rt-table"></table></div>
  </figure>
  <h3>群集 30 附加条件（零样本，3 次测试飞行共 64 个场景）</h3>
  <div class="controls" role="group" aria-label="选择指标" id="e13x-buttons"></div>
  <div class="table-wrap"><table class="heat" id="e13x-table"></table></div>
  <p class="note">30 架时不跑两个 MILP 基线。</p>
  <h3>ns-3 复核</h3>
  <p class="prose">群集 30 主条件，每折取前 {{e13_ns3_eps}} 个场景，PHY 执行加帧内运动。衰落样本通过私有执行帧传给 ns-3，与轻量环境逐周期一致。两种执行方式下交付率差的置信区间都在 ±{{e13_ns3_fid}} 个百分点以内。</p>
  <div class="table-wrap"><table id="e13-ns3"></table></div>
  <h3>链路模型检验（UCSB 空地实测，B 类）</h3>
  <p class="prose">按序号还原每个包的发送时刻和收发距离，再把试验分成两半：一半拟合，一半检验。拟合采用的主配置是水平发射、无遮挡，对应水平接收。预先登记的判定成立：莱斯模型在两个指标上都优于无衰落门限模型。不过对数正态模型（σ = {{ucsb_sigma}} dB）拟合得同样好。收到的包都带 RSSI，据此可以分解起伏：总起伏 {{rssi_total}} dB，其中约 {{rssi_slow}} dB 持续 2 s 以上，约 {{rssi_fast}} dB 在 0.5 s 内就不再相关。这份数据是空对地链路，发射端在地面上，所以只作为参考证据。</p>
  <div class="table-wrap"><table id="e13-ucsb"></table></div>
</section>

<section>
  <h2>后续实验（E13b、E14、E15）</h2>
  <p class="prose">以下三项是 2026-10-02 决定的后续工作。每项都在开跑前写好预登记，并另设配置和结果目录，不改动上面的配置和结论。</p>
  <ul class="claims prose">
    <li><strong>E13b 真实轨迹上训练全部可训练基线</strong> <span class="tag">{{e13b_tag}}</span>：GRLinQ 式、独立决定 PPO、仅模仿，与 E13 同样按折重新训练。对照扩充到 E11 的全部 13 个。完成后，图 10 和图 11 会包含这些策略。</li>
    <li><strong>E14 探索：训练分布覆盖低速和紧凑机群</strong> <span class="tag">{{e14_tag}}</span>：每个回合以 1/2 的概率取原训练分布，以 1/2 的概率取“低速紧凑”分布。后者速度 0–5 m/s，场地 150–300 m，16–30 个节点，参数只根据与方法无关的拓扑统计确定。全部学习方法都在这个分布上重新训练，用新的测试种子评估。</li>
    <li><strong>E15 探索：5 dB 阴影作为控制变量</strong> <span class="tag">{{e15_tag}}</span>：在实测参数信道上加入 σ = 5 dB 的对数正态阴影。阴影调度器已知。有阴影和无阴影两种条件使用同一批场景和同一组快衰落样本，各测一套，方法冻结。</li>
  </ul>
  <div id="e14-wrap" hidden>
    <figure id="fig-e14">
      <p class="fig-title">图 12　宽分布训练的主方法相对各策略（E14）</p>
      <p class="fig-sub">合成场景族每族 64 个新场景；群集 30 每次飞行 32 个场景，16 架和 30 架各一套。</p>
      <div class="controls" role="group" aria-label="选择指标" id="e14-buttons"></div>
      <div class="table-wrap"><table class="heat" id="e14-table"></table></div>
    </figure>
    <figure id="fig-e14b">
      <p class="fig-title">图 13　同一方法：宽分布训练减原分布训练（E14）</p>
      <div class="controls" role="group" aria-label="选择指标" id="e14b-buttons"></div>
      <div class="table-wrap"><table class="heat" id="e14b-table"></table></div>
    </figure>
  </div>
  <div id="e15-wrap" hidden>
    <figure id="fig-e15">
      <p class="fig-title">图 14　有阴影（C2S5）时主方法相对各策略（E15）</p>
      <div class="controls" role="group" aria-label="选择指标" id="e15-buttons"></div>
      <div class="table-wrap"><table class="heat" id="e15-table"></table></div>
    </figure>
    <h3>阴影本身的影响：同一批场景、同一组衰落样本上 C2S5 与 C2 的对比</h3>
    <div class="table-wrap"><table id="e15-effect"></table></div>
    <h3>机理：阴影让网络更连通（与方法无关）</h3>
    <p class="prose">默认场景族，回合开始时。调度器可用的节点对 = 平均 SNR 达到规划门限的节点对。功率归一 = 把阴影的 dB 均值下调 σ²/(2ξ)，使平均接收功率与 C2 相同；这一行只是统计，没有预登记，也没有评估。</p>
    <div class="table-wrap"><table id="e15-conn"></table></div>
    <h3>阴影强度（默认场景族前 32 个场景）</h3>
    <div class="table-wrap"><table id="e15-sigma"></table></div>
    <div id="e15-ns3-wrap"></div>
  </div>
</section>

<section>
  <h2>方法</h2>
  <div class="prose">
    <p>集中式调度器在每个物理周期，从当前可获得的报告中选出一组能同时发送的一跳链路。定稿方法：</p>
    <ul class="claims">
      <li><strong>双图表示：</strong>通信图（无人机为节点）经 Lift 得到候选链路表示，再在链路交互图（冲突与有向干扰）上编码。每个周期只编码一次。</li>
      <li><strong>微步选择：</strong>Actor 按掩码逐条选择链路，每一步由控制器用<strong>完整累计 SINR</strong>和半双工约束更新可行掩码。没有 STOP，输出极大可行集合。定稿方法不使用学习式已选摘要（依据见下文“研究过程”）。</li>
      <li><strong>训练：</strong>先模仿 longest_queue 的逐步选择作为起点，再用微步 PPO 微调。微步转移折扣为 1，周期边界折扣为 γ；奖励为服务 − 剩余队列 − 时延压力 − λ·终止违约（λ = 20）。检查点按“交付率不劣于基线，再取时延最低”的固定规则在开发集上选择。</li>
    </ul>
  </div>
  <figure class="flow">
<pre class="mermaid">
flowchart LR
  R["决策时报告<br/>位置·信道估计·队列"] --> G["双图编码<br/>每周期一次"]
  G --> A["Actor 评分"]
  A -->|选一条链路| C["控制器<br/>完整累计 SINR·半双工"]
  C -->|更新掩码| A
  C -->|极大可行集合| P["完整计划"]
  P --> B["后端执行<br/>实际成功·包状态"]
  B --> F["执行事实"]
  F --> W["奖励 / 指标"]
  B -->|下一周期| R
</pre>
  </figure>
</section>

<section>
  <h2>研究过程</h2>
  <p class="prose">从 E5 起，每一轮实验的方案和判定规则都在看到结果之前写进文档并提交。测试集在 E9 首次使用；之后每一轮（E10–E15）都另取一批没用过的测试种子。</p>
  <div class="timeline">
    <div class="step"><div class="id">E1</div><div><h3>首次训练 <span class="tag">开发集</span></h3>
      <p>PPO 按奖励明显优于基线，但分项显示它靠多丢包压低了队列和时延分项。</p>
      <p class="out">→ 原因是丢包惩罚权重过低，不是实现错误；需要校准权重。</p></div></div>
    <div class="step"><div class="id">E2</div><div><h3>丢包惩罚权重与排空评估</h3>
      <p>在 10 秒截止时仍有数百个包在途，截止时的交付率会误导比较。加入排空评估后，从零训练的 PPO 交付率低于基线。</p>
      <p class="out">→ 可靠性一律以排空后的最终交付率为准。</p></div></div>
    <div class="step"><div class="id">E3</div><div><h3>模仿诊断</h3>
      <p>只模仿 longest_queue 的模型交付率 {{imit_dev}}，与教师 {{lq_dev}} 持平。</p>
      <p class="out">→ 双图模型能表示同等质量的决策，差距来自从零优化。</p></div></div>
    <div class="step"><div class="id">E4–E5</div><div><h3>模仿预热 + PPO 微调，多种子复现</h3>
      <p>固定检查点选择规则后，3 个种子都达到可靠性标准，时延更低。从零训练的 PPO 在 3 个种子上都不达标（开发集交付率 {{pure_dev}}）。</p>
      <p class="out">→ 确立“模仿预热 + 微调 + 可靠性优先选择”的训练协议，丢包惩罚取 λ = 20。</p></div></div>
    <div class="step"><div class="id">E6</div><div><h3>消融（默认场景与困难场景）</h3>
      <p>完整累计 SINR 约束和直接链路特征通路不可缺少。交互图主要降低时延，通信图消息传递主要提高交付率。门控求和摘要去掉后交付率不变、时延更低。</p>
      <p class="out">→ 见图 7。</p></div></div>
    <div class="step"><div class="id">E7</div><div><h3>零样本泛化</h3>
      <p>6 种未见条件中，可靠性全部不劣于基线；在更难的条件下，交付率和时延同时改善。其中“24 节点同密度”变体曾误设为 424 m，已更正为 367.4 m 并补评。</p>
      <p class="out">→ 重负载评估改用 1000 周期排空。</p></div></div>
    <div class="step"><div class="id">E8</div><div><h3>已选摘要改进</h3>
      <p>归一化摘要和控制器动态特征都能进一步降低时延，但在每个场景都以交付率为代价，未通过预先登记的规则。</p>
      <p class="out">→ 经用户确认，定稿方法不使用已选摘要；其余实现保留为可切换的对照。</p></div></div>
    <div class="step"><div class="id">E9</div><div><h3>测试集最终评估 <span class="tag">测试集</span></h3>
      <p>5 个训练种子，{{episodes}} 个测试场景 × 5 个场景族，排空 {{drain}} 周期，预先登记的全部主要判定成立（图 1、图 2）。</p></div></div>
    <div class="step"><div class="id">E10</div><div><h3>ns-3 物理层验证 <span class="tag">新测试场景</span></h3>
      <p>执行规则按本项目的方法自定：DATA 走真实物理层，集中式确认，帧内连续运动。开跑前修正了带宽取值差 1 字节的问题，以及评估时切换后端会丢失路由参数的问题。</p>
      <p class="out">→ 预先登记的 5 项判定全部满足（图 3）。</p></div></div>
    <div class="step"><div class="id">E11</div><div><h3>12 个外部基线 <span class="tag">新测试场景</span></h3>
      <p>补充每周期精确最优、反压、局部搜索、空间 TDMA 和 3 个前人的学习方法；另做近似比、决策时间、负载 / 规模 / 速度扫描和 ns-3 复核。</p>
      <p class="out">→ 5 个场景族中交付率都最高；反压时延更低；节点几乎静止时有一处未达标（图 4、图 5）。</p></div></div>
    <div class="step"><div class="id">E12</div><div><h3>实测参数信道 <span class="tag">新测试场景</span></h3>
      <p>路径损耗指数 2.2，莱斯 K = 10 dB，按 10% 中断留 3 dB 裕量。结果出来后发现，路由仍按无裕量的门限选路，部分包落在永远不会被调度的链路上。</p>
      <p class="out">→ 两个后端都修正路由，并补了测试；在新的测试种子上重跑（E12b）。</p></div></div>
    <div class="step"><div class="id">E12b</div><div><h3>修正路由后重跑 <span class="tag">新测试场景</span></h3>
      <p>预先登记的判定全部成立。所有方法的交付率因衰落裕量下降约 18–24 个百分点，排名不变；ns-3 复核一致。</p>
      <p class="out">→ 图 9。</p></div></div>
    <div class="step"><div class="id">E13</div><div><h3>真实轨迹、第三方移动模型、实测收包 <span class="tag">新测试场景</span></h3>
      <p>群集 30 的真实飞行按折评估，重新训练了主方法和 Zhao-GCN；另用 BonnMotion 的 RPGM 与高斯-马尔可夫模型，以及 UCSB 空地收包记录。ns-3 不接受逐周期变速的轨迹，C++ 改为允许速度在周期边界变化。</p>
      <p class="out">→ 预先登记的判定全部成立；低速、紧凑机群是零样本的弱点（图 10、图 11）。</p></div></div>
    <div class="step"><div class="id">E13b</div><div><h3>真实轨迹上训练全部可训练基线 <span class="tag">{{e13b_tag}}</span></h3>
      <p>GRLinQ 式、独立决定 PPO 和仅模仿按折重新训练；对照扩充到 E11 的全部 13 个。</p></div></div>
    <div class="step"><div class="id">E14</div><div><h3>探索：训练分布覆盖低速和紧凑机群 <span class="tag">{{e14_tag}}</span></h3>
      <p>以混合分布训练全部学习方法，在新的测试种子上，与原分布训练的版本比较。</p></div></div>
    <div class="step"><div class="id">E15</div><div><h3>探索：5 dB 阴影作为控制变量 <span class="tag">{{e15_tag}}</span></h3>
      <p>有阴影、无阴影各测一套；设计依据是 UCSB 的 RSSI 分解。</p></div></div>
  </div>
</section>

<section>
  <h2>训练过程</h2>
  <figure id="fig-curves">
    <p class="fig-title">图 6　训练中的开发集评估</p>
    <p class="fig-sub">每 25 次迭代在开发集的 8 个场景上评估一次（排空 250 周期）。圆点为按固定规则选中的检查点。</p>
    <div class="legend">
      <span><i class="key-line"></i>定稿方法（5 个种子）</span>
      <span><i class="key-line ctx"></i>原门控求和（3 个种子）</span>
      <span><i class="key-dot"></i>选中的检查点</span>
      <span><i class="key-line ref"></i>longest_queue 同场景参照</span>
    </div>
    <div class="panels" id="curves"></div>
    <figcaption><b>读法：</b>模仿预热后，模型从教师水平起步，时延随微调下降。训练后期交付率会滑向“以丢包换时延”，所以要按交付率优先的规则选择检查点，而不是直接取最后一次迭代。</figcaption>
  </figure>
</section>

<section>
  <h2>消融与摘要改进</h2>
  <figure id="fig-heat">
    <p class="fig-title">图 7　去掉或替换部件后的变化（相对当时的完整模型，含门控求和摘要）</p>
    <p class="fig-sub">E6 与 E6b，开发集第 16–47 个场景，每组 3 个训练种子取平均。着色表示差异显著：蓝色为更好，红色为更差；灰色为不显著。</p>
    <div class="controls" role="group" aria-label="选择指标" id="heat-buttons"></div>
    <div class="table-wrap"><table class="heat" id="heat-table"></table></div>
    <div class="swatches"><span><i style="background:color-mix(in oklab,var(--pos) 45%,var(--surface))"></i>显著更好</span><span><i style="background:color-mix(in oklab,var(--neg) 45%,var(--surface))"></i>显著更差</span><span><i style="background:var(--surface);border:1px solid var(--rule)"></i>不显著</span></div>
  </figure>
  <figure id="fig-e8">
    <p class="fig-title">图 8　已选摘要的两种改进方案（相对“无摘要”）</p>
    <p class="fig-sub">E8，判定规则：5 个场景的交付率下界都 ≥ −0.5 个百分点，且平均时延至少在 3 个场景显著更低、在任何场景都不显著更高。两种方案都未通过。</p>
    <div class="controls" role="group" aria-label="选择指标" id="e8-buttons"></div>
    <div class="table-wrap"><table class="heat" id="e8-table"></table></div>
  </figure>
</section>

<section class="prose">
  <h2>局限</h2>
  <ul class="claims">
    <li>定稿方法只在一个场景族上训练：16 节点、300 m、理想信道、单速率单信道。困难条件、实测参数信道和真实轨迹上的结果都属于零样本泛化；E13 的“重训”版本除外。</li>
    <li>ns-3 验证覆盖了包级物理层、帧内运动、莱斯衰落（通过私有执行帧）和真实轨迹回放，但没有覆盖空口 ACK、控制信道时延、报告丢失和多速率。时序仍是同步的。</li>
    <li>信道只有单一速率，衰落裕量对绝对性能影响很大；块衰落按周期抽样。UCSB 实测显示，链路起伏大于 K = 10 dB 的莱斯衰落（E15 正在探索）。真实数据只提供轨迹，业务、信道和收发都由模型补齐。</li>
    <li>环境语义与研究方案第 19 版有两处差异：路由变化后重新安置旧队列；等待区恢复时，若目标队列已满则继续等待。详见 <code>docs/decisions.md</code>。</li>
    <li>反压基线用到了主方法没有的信息（每个包的目的地），时延比主方法低。节点几乎静止或机群紧凑时，零样本主方法的交付率略低于 LQ（E11、E13）。</li>
    <li>比较没有做多重校正。测试集已分批用于 E9–E15，每轮都换了新的种子偏移。</li>
  </ul>
  <h2>复现</h2>
  <p>全部源码、配置、实验方案和文档都在 GitHub 仓库 <code>NoQuitNoDefeat/paper1-next</code> 中。运行结果单独保存在本机 <code>results/</code>。本页由 <code>tools/build_report.py</code> 从结果文件生成。</p>
<pre class="cmd"><code>.venv/bin/fanet-next train --config configs/protocol_final.toml --set seed=0 --run-dir results/runs/final-s0
.venv/bin/fanet-next select --run-dir results/runs/final-s0
.venv/bin/python tools/confirm.py --spec configs/experiments/e9_test.json
tools/ns3/setup.sh    # 构建 ns-3.48 + ns3-ai + 本项目的 fanet-scheduler 模块
.venv/bin/python tools/e10_ns3.py --spec configs/experiments/e10_ns3.json
.venv/bin/fanet-next train-baseline --config configs/baselines/zhao_gcn.toml --set seed=0 --run-dir results/e11/runs/zhao_gcn-s0
.venv/bin/python tools/e11_compare.py --spec configs/experiments/e11_compare.json
.venv/bin/python tools/e11_compare.py --spec configs/experiments/e12b_channel.json
.venv/bin/python tools/data/build_vasarhelyi.py && .venv/bin/python tools/e13_train.py
.venv/bin/python tools/e11_compare.py --spec configs/experiments/e13_flock.json
.venv/bin/python tools/train_runs.py configs/experiments/e13b_train.json configs/experiments/e14_train.json
.venv/bin/python tools/build_report.py</code></pre>
  <p class="note">实验记录：<code>docs/experiments.md</code>（E1–E15）；数据筛选：<code>docs/data-screening.md</code>；实现决定：<code>docs/decisions.md</code>；已知问题：<code>docs/known-issues.md</code>。</p>
</section>
</div>

<script>
const DATA = /*DATA*/null;
const LABEL = DATA.labels;
const SCEN = DATA.e9.scenarios;
const MAIN = "主方法（无摘要）";
const NAME = {"longest_queue":"longest_queue","hol_weighted":"hol_weighted","random":"random","仅模仿":"仅模仿",
  "主方法（无摘要）":"定稿方法","原门控求和（原设计）":"原门控求和","控制器动态特征（时延变体）":"控制器动态特征","纯PPO（从零训练）":"纯 PPO（从零）"};
const SVGNS = "http://www.w3.org/2000/svg";
function el(tag, attrs, parent){ const e=document.createElementNS(SVGNS, tag); for(const k in attrs) e.setAttribute(k, attrs[k]); if(parent) parent.appendChild(e); return e; }
function txt(parent, x, y, s, cls, anchor){ const t=el("text",{x,y,class:cls||"","text-anchor":anchor||"start","dominant-baseline":"middle"},parent); t.textContent=s; return t; }
function niceTicks(lo, hi, n){ const span=hi-lo, step0=span/n, mag=Math.pow(10,Math.floor(Math.log10(step0))); const err=step0/mag;
  const step=mag*(err>=7.5?10:err>=3.5?5:err>=1.5?2:1); const t=[]; for(let v=Math.ceil(lo/step)*step; v<=hi+1e-12; v+=step) t.push(+v.toFixed(10)); return t; }
function clear(n){ while(n.firstChild) n.removeChild(n.firstChild); }
/* create every panel of a grid before measuring any of them, so widths are final */
function makePanels(host, titles){
  clear(host);
  return titles.map(([title, sub])=>{ const panel=document.createElement("div"); panel.className="panel";
    const h=document.createElement("h4"); h.textContent=title; if(sub){ const s=document.createElement("span"); s.textContent=sub; h.appendChild(s);} panel.appendChild(h);
    host.appendChild(panel); return panel; });
}

/* ---------- shared tooltip ---------- */
function tipFor(fig){ let t=fig.querySelector(".tip"); if(!t){ t=document.createElement("div"); t.className="tip"; t.hidden=true; fig.appendChild(t);} return t; }
function showTip(fig, evt, title, rows){
  const t=tipFor(fig); clear(t);
  const h=document.createElement("div"); h.className="t"; h.textContent=title; t.appendChild(h);
  for(const [k,v,color] of rows){ const r=document.createElement("div"); r.className="r";
    const a=document.createElement("span"); if(color){ const i=document.createElement("i"); i.style.cssText=`display:inline-block;width:12px;height:2px;margin-right:6px;vertical-align:middle;background:${color}`; a.appendChild(i);} a.appendChild(document.createTextNode(k));
    const b=document.createElement("b"); b.textContent=v; r.appendChild(a); r.appendChild(b); t.appendChild(r); }
  t.hidden=false; const fr=fig.getBoundingClientRect(); const px=(evt.clientX??fr.left)-fr.left, py=(evt.clientY??fr.top)-fr.top;
  const w=t.offsetWidth; t.style.left=Math.min(Math.max(px+14,8), fr.width-w-8)+"px"; t.style.top=(py+14)+"px";
}
function hideTip(fig){ const t=fig.querySelector(".tip"); if(t) t.hidden=true; }

/* ---------- figure 1: forest plots ---------- */
const FOREST = [
  {key:"delivery_ratio", title:"最终交付率", unit:"个百分点", better:"更好 →", scale:100, dec:1},
  {key:"e2e_delay_mean_s", title:"平均时延", unit:"秒", better:"← 更好", scale:1, dec:2},
  {key:"e2e_delay_p95_s", title:"p95 时延", unit:"秒", better:"← 更好", scale:1, dec:2},
];
function drawForest(){
  const host=document.getElementById("forest"), fig=document.getElementById("fig-forest");
  const panels=makePanels(host, FOREST.map(m=>[m.title, `${m.unit}　${m.better}`]));
  FOREST.forEach((m,pi)=>{ const panel=panels[pi];
    const W=Math.max(panel.clientWidth,260), rowH=34, top=18, bottom=26, labW=96, valW=52, H=top+rowH*SCEN.length+bottom;
    const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,height:H,role:"img","aria-label":`${m.title}配对差`},panel);
    const rows=SCEN.map(s=>({label:s.label, ...s.diffs["longest_queue"][m.key]}));
    let lo=Math.min(0,...rows.map(r=>r.lo))*m.scale, hi=Math.max(0,...rows.map(r=>r.hi))*m.scale; const pad=(hi-lo)*0.08; lo-=pad; hi+=pad;
    const x0=labW, x1=W-valW, X=v=>x0+(v-lo)/(hi-lo)*(x1-x0);
    const g=el("g",{class:"grid"},svg);
    for(const t of niceTicks(lo,hi,W<420?3:4)){ el("line",{x1:X(t),x2:X(t),y1:top-6,y2:H-bottom+4},g); txt(svg,X(t),H-bottom+16,t.toFixed(Math.abs(t)<1&&t!==0?(m.scale===1?1:0):0).replace(/^-?0\.0$/,"0"),"tick","middle"); }
    el("line",{x1:X(0),x2:X(0),y1:top-8,y2:H-bottom+4,class:"zero"},svg);
    rows.forEach((r,i)=>{
      const y=top+rowH*i+rowH/2;
      txt(svg,0,y,r.label,"lbl");
      el("line",{x1:X(r.lo*m.scale),x2:X(r.hi*m.scale),y1:y,y2:y,stroke:"var(--accent)","stroke-width":2,"stroke-linecap":"round"},svg);
      el("circle",{cx:X(r.m*m.scale),cy:y,r:5,fill:"var(--accent)",stroke:"var(--surface)","stroke-width":2},svg);
      txt(svg,W,y,(r.m*m.scale>0?"+":"")+(r.m*m.scale).toFixed(m.dec),"val","end");
      const hit=el("rect",{x:0,y:y-rowH/2,width:W,height:rowH,fill:"transparent",tabindex:0},svg);
      const f=v=>(v*m.scale>0?"+":"")+(v*m.scale).toFixed(m.dec+1);
      const show=e=>showTip(fig,e,`${r.label} · ${m.title}`,[["均值",`${f(r.m)} ${m.unit}`],["95% CI",`${f(r.lo)} … ${f(r.hi)}`]]);
      hit.addEventListener("pointermove",show); hit.addEventListener("focus",e=>{const b=hit.getBoundingClientRect(); show({clientX:b.left+b.width/2,clientY:b.top});});
      hit.addEventListener("pointerleave",()=>hideTip(fig)); hit.addEventListener("blur",()=>hideTip(fig));
    });
  });
}

/* ---------- figure 2: dot plots per scenario ---------- */
let scenIdx = 0;
const DOTM = [
  {key:"delivery_ratio", title:"最终交付率", fmt:v=>v.toFixed(3)},
  {key:"e2e_delay_mean_s", title:"平均时延（秒）", fmt:v=>v.toFixed(2)},
  {key:"e2e_delay_p95_s", title:"p95 时延（秒）", fmt:v=>v.toFixed(2)},
];
function drawDots(){
  const host=document.getElementById("dots"), fig=document.getElementById("fig-dots");
  const s=SCEN[scenIdx];
  const order=Object.keys(s.means).sort((a,b)=>s.means[b].delivery_ratio-s.means[a].delivery_ratio);
  const panels=makePanels(host, DOTM.map(m=>[m.title]));
  DOTM.forEach((m,pi)=>{ const panel=panels[pi];
    const W=Math.max(panel.clientWidth,260), rowH=28, top=10, bottom=24, labW=m===DOTM[0]?118:118, valW=46, H=top+rowH*order.length+bottom;
    const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,height:H,role:"img","aria-label":`${s.label} ${m.title}`},panel);
    const vals=order.map(p=>s.means[p][m.key]); let lo=Math.min(...vals), hi=Math.max(...vals); const pad=(hi-lo)*0.1||0.01; lo-=pad; hi+=pad;
    const x0=labW, x1=W-valW, X=v=>x0+(v-lo)/(hi-lo)*(x1-x0);
    const g=el("g",{class:"grid"},svg);
    for(const t of niceTicks(lo,hi,W<420?3:4)){ el("line",{x1:X(t),x2:X(t),y1:top-4,y2:H-bottom+4},g); txt(svg,X(t),H-bottom+15,m.key==="delivery_ratio"?t.toFixed(2):t.toFixed(1),"tick","middle"); }
    order.forEach((p,i)=>{
      const y=top+rowH*i+rowH/2, v=s.means[p][m.key], isMain=p===MAIN;
      txt(svg,0,y,NAME[p]||p,"lbl").style.fontWeight=isMain?"600":"400";
      el("line",{x1:x0,x2:X(v),y1:y,y2:y,stroke:isMain?"var(--accent)":"var(--context)","stroke-width":1,opacity:.45},svg);
      el("circle",{cx:X(v),cy:y,r:isMain?5.5:4.5,fill:isMain?"var(--accent)":"var(--context)",stroke:"var(--surface)","stroke-width":2},svg);
      const vt=txt(svg,W,y,m.fmt(v),"val","end"); if(isMain) vt.style.fontWeight="600";
      const hit=el("rect",{x:0,y:y-rowH/2,width:W,height:rowH,fill:"transparent",tabindex:0},svg);
      const mm=s.means[p];
      const show=e=>showTip(fig,e,`${s.label} · ${NAME[p]||p}`,[["最终交付率",mm.delivery_ratio.toFixed(4)],["终止比例",mm.termination_ratio.toFixed(4)],["2 秒内送达",mm.ontime_2s.toFixed(3)],["平均时延",mm.e2e_delay_mean_s.toFixed(3)+" 秒"],["p95 时延",mm.e2e_delay_p95_s.toFixed(3)+" 秒"]]);
      hit.addEventListener("pointermove",show); hit.addEventListener("focus",e=>{const b=hit.getBoundingClientRect(); show({clientX:b.left+b.width/2,clientY:b.top});});
      hit.addEventListener("pointerleave",()=>hideTip(fig)); hit.addEventListener("blur",()=>hideTip(fig));
    });
  });
  drawMeansTable(s, order);
}
function drawMeansTable(s, order){
  const t=document.getElementById("means-table"); clear(t);
  const cols=[["delivery_ratio","最终交付率",4],["termination_ratio","终止比例",4],["ontime_2s","2 秒内送达",3],["e2e_delay_mean_s","平均时延 (s)",3],["e2e_delay_p95_s","p95 时延 (s)",3]];
  const th=t.createTHead().insertRow(); const c0=document.createElement("th"); c0.textContent=`策略（${s.label}，${s.n} 个场景）`; th.appendChild(c0);
  for(const [,l] of cols){ const c=document.createElement("th"); c.textContent=l; th.appendChild(c); }
  const tb=t.createTBody();
  for(const p of order){ const r=tb.insertRow(); if(p===MAIN) r.className="main"; const a=r.insertCell(); a.textContent=NAME[p]||p;
    for(const [k,,dec] of cols){ const c=r.insertCell(); c.textContent=s.means[p][k].toFixed(dec); } }
}
function scenButtons(){
  const host=document.getElementById("scen-buttons");
  SCEN.forEach((s,i)=>{ const b=document.createElement("button"); b.type="button"; b.textContent=s.label; b.setAttribute("aria-pressed",i===scenIdx);
    b.addEventListener("click",()=>{ scenIdx=i; host.querySelectorAll("button").forEach((x,j)=>x.setAttribute("aria-pressed",j===i)); drawDots(); }); host.appendChild(b); });
}

/* ---------- figure 3: training curves ---------- */
const CURVE = [{idx:1, key:"delivery_ratio", title:"开发集最终交付率", fmt:v=>v.toFixed(3)},
               {idx:2, key:"e2e_delay_mean_s", title:"开发集平均时延（秒）", fmt:v=>v.toFixed(2)}];
function drawCurves(){
  const host=document.getElementById("curves"), fig=document.getElementById("fig-curves");
  const G=DATA.curves.groups, ref=DATA.curves.lq;
  const series=[]; for(const [g,runs] of Object.entries(G)) for(const r of runs) series.push({g, main:g===MAIN, ...r});
  const panels=makePanels(host, CURVE.map(m=>[m.title]));
  CURVE.forEach((m,pi)=>{ const panel=panels[pi];
    const W=Math.max(panel.clientWidth,260), H=230, l=46, r=16, t=12, b=28;
    const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,height:H,role:"img","aria-label":m.title},panel);
    const xs=series[0].points.map(p=>p[0]); const xmin=0, xmax=Math.max(...xs);
    const all=series.flatMap(s=>s.points.map(p=>p[m.idx])).concat([ref[m.key]]);
    let lo=Math.min(...all), hi=Math.max(...all); const pad=(hi-lo)*0.1; lo-=pad; hi+=pad;
    const X=v=>l+(v-xmin)/(xmax-xmin)*(W-l-r), Y=v=>t+(hi-v)/(hi-lo)*(H-t-b);
    const g=el("g",{class:"grid"},svg);
    for(const v of niceTicks(lo,hi,4)){ el("line",{x1:l,x2:W-r,y1:Y(v),y2:Y(v)},g); txt(svg,l-6,Y(v),m.fmt(v),"tick","end"); }
    for(const v of [0,50,100,150,200]) txt(svg,X(v),H-b+15,String(v),"tick","middle");
    txt(svg,W-r,H-4,"迭代","tick","end");
    el("line",{x1:l,x2:W-r,y1:Y(ref[m.key]),y2:Y(ref[m.key]),stroke:"var(--ink-2)","stroke-width":1},svg);
    for(const s of [...series].sort((a,b)=>a.main-b.main)){
      const d=s.points.map((p,i)=>`${i?"L":"M"}${X(p[0]).toFixed(1)},${Y(p[m.idx]).toFixed(1)}`).join("");
      el("path",{d,fill:"none",stroke:s.main?"var(--accent)":"var(--context)","stroke-width":2,"stroke-linejoin":"round","stroke-linecap":"round",opacity:s.main?.85:.8},svg);
    }
    for(const s of series.filter(s=>s.main)){ const p=s.points.find(p=>p[0]===s.selected); if(p) el("circle",{cx:X(p[0]),cy:Y(p[m.idx]),r:4.5,fill:"var(--accent)",stroke:"var(--surface)","stroke-width":2},svg); }
    const cross=el("line",{y1:t,y2:H-b,stroke:"var(--axis)","stroke-width":1,visibility:"hidden"},svg);
    const hit=el("rect",{x:l,y:t,width:W-l-r,height:H-t-b,fill:"transparent"},svg);
    hit.addEventListener("pointermove",e=>{
      const bb=svg.getBoundingClientRect(); const px=(e.clientX-bb.left)*W/bb.width;
      const it=xs.reduce((a,v)=>Math.abs(X(v)-px)<Math.abs(X(a)-px)?v:a, xs[0]);
      cross.setAttribute("x1",X(it)); cross.setAttribute("x2",X(it)); cross.setAttribute("visibility","visible");
      const rows=series.map(s=>{ const p=s.points.find(q=>q[0]===it); return [`${s.main?"定稿":"门控求和"} ${s.run.replace(/^.*-s/,"s")}${s.selected===it?"（选中）":""}`, p?m.fmt(p[m.idx]):"–", s.main?"var(--accent)":"var(--context)"]; });
      rows.push(["longest_queue 参照", m.fmt(ref[m.key]), "var(--ink-2)"]);
      showTip(fig,e,`第 ${it} 次迭代`,rows);
    });
    hit.addEventListener("pointerleave",()=>{ cross.setAttribute("visibility","hidden"); hideTip(fig); });
  });
}

/* ---------- figures 4-5: significance heat tables ---------- */
const HEATM = [
  {key:"delivery_ratio", title:"最终交付率（个百分点）", scale:100, dec:2, goodSign:+1, cap:3},
  {key:"e2e_delay_mean_s", title:"平均时延（秒）", scale:1, dec:3, goodSign:-1, cap:0.3},
  {key:"e2e_delay_p95_s", title:"p95 时延（秒）", scale:1, dec:3, goodSign:-1, cap:0.6},
];
function heat(tableId, buttonsId, rowsData, rowNames, scenIds, rowLabel){
  let mi=0; const host=document.getElementById(buttonsId);
  const render=()=>{
    const m=HEATM[mi], t=document.getElementById(tableId); clear(t);
    const hr=t.createTHead().insertRow(); const c0=document.createElement("th"); c0.textContent=m.title; hr.appendChild(c0);
    for(const s of scenIds){ const c=document.createElement("th"); c.textContent=LABEL[s]; hr.appendChild(c); }
    const tb=t.createTBody();
    for(const rname of rowNames){ const r=tb.insertRow(); const a=r.insertCell(); a.textContent=(rowLabel&&rowLabel[rname])||rname;
      for(const s of scenIds){ const c=r.insertCell(); c.className="cell"; const cell=rowsData[rname][s];
        if(!cell){ c.textContent="–"; c.classList.add("ns"); continue; } const d=cell[m.key];
        const sig=d.lo>0||d.hi<0; const good=Math.sign(d.m)===m.goodSign;
        const v=d.m*m.scale; const main=document.createElement("div"); main.textContent=(v>0?"+":"")+v.toFixed(m.dec);
        const ci=document.createElement("span"); ci.className="ci"; ci.textContent=`[${(d.lo*m.scale).toFixed(m.dec)}, ${(d.hi*m.scale).toFixed(m.dec)}]`;
        c.appendChild(main); c.appendChild(ci);
        if(sig){ const k=Math.min(Math.abs(v)/m.cap,1); const pct=Math.round(14+34*k);
          c.style.background=`color-mix(in oklab, ${good?"var(--pos)":"var(--neg)"} ${pct}%, var(--surface))`;
          c.title=good?"显著更好":"显著更差"; } else { c.classList.add("ns"); c.title="不显著"; } } }
  };
  HEATM.forEach((m,i)=>{ const b=document.createElement("button"); b.type="button"; b.textContent=m.title.replace(/（.*/,""); b.setAttribute("aria-pressed",i===0);
    b.addEventListener("click",()=>{ mi=i; host.querySelectorAll("button").forEach((x,j)=>x.setAttribute("aria-pressed",j===i)); render(); }); host.appendChild(b); });
  render();
}

/* ---------- figure 3: E10 advantage in both executors ---------- */
const E10 = DATA.e10;
function drawE10(){
  const host=document.getElementById("e10-forest"), fig=document.getElementById("fig-e10");
  const rows=[]; for(const s of E10.scenarios) for(const [x,lab,ns] of [["ns3","ns-3",true],["lightweight","轻量",false]]) rows.push({s, x, ns, label:`${s.label} · ${lab}`});
  const panels=makePanels(host, FOREST.map(m=>[m.title, `${m.unit}　${m.better}`]));
  FOREST.forEach((m,pi)=>{ const panel=panels[pi];
    const W=Math.max(panel.clientWidth,260), rowH=30, top=18, bottom=26, labW=104, valW=52, H=top+rowH*rows.length+bottom+8;
    const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,height:H,role:"img","aria-label":`${m.title}：两种执行环境中的配对差`},panel);
    const ds=rows.map(r=>r.s.adv[r.x][m.key]);
    let lo=Math.min(0,...ds.map(d=>d.lo))*m.scale, hi=Math.max(0,...ds.map(d=>d.hi))*m.scale; const pad=(hi-lo)*0.08; lo-=pad; hi+=pad;
    const x0=labW, x1=W-valW, X=v=>x0+(v-lo)/(hi-lo)*(x1-x0);
    const Yr=i=>top+rowH*i+rowH/2+(i>=2?8:0);
    const g=el("g",{class:"grid"},svg);
    for(const t of niceTicks(lo,hi,W<420?3:4)){ el("line",{x1:X(t),x2:X(t),y1:top-6,y2:H-bottom+4},g); txt(svg,X(t),H-bottom+16,t.toFixed(Math.abs(t)<1&&t!==0?(m.scale===1?1:0):0).replace(/^-?0\.0$/,"0"),"tick","middle"); }
    el("line",{x1:X(0),x2:X(0),y1:top-8,y2:H-bottom+4,class:"zero"},svg);
    rows.forEach((r,i)=>{ const d=ds[i], y=Yr(i), color=r.ns?"var(--accent)":"var(--context)";
      txt(svg,0,y,r.label,"lbl");
      el("line",{x1:X(d.lo*m.scale),x2:X(d.hi*m.scale),y1:y,y2:y,stroke:color,"stroke-width":2,"stroke-linecap":"round"},svg);
      el("circle",{cx:X(d.m*m.scale),cy:y,r:r.ns?5:4.5,fill:color,stroke:"var(--surface)","stroke-width":2},svg);
      txt(svg,W,y,(d.m*m.scale>0?"+":"")+(d.m*m.scale).toFixed(m.dec),"val","end");
      const hit=el("rect",{x:0,y:y-rowH/2,width:W,height:rowH,fill:"transparent",tabindex:0},svg);
      const f=v=>(v*m.scale>0?"+":"")+(v*m.scale).toFixed(m.dec+1);
      const show=e=>showTip(fig,e,`${r.label} · ${m.title}`,[["均值",`${f(d.m)} ${m.unit}`],["95% CI",`${f(d.lo)} … ${f(d.hi)}`]]);
      hit.addEventListener("pointermove",show); hit.addEventListener("focus",()=>{const b=hit.getBoundingClientRect(); show({clientX:b.left+b.width/2,clientY:b.top});});
      hit.addEventListener("pointerleave",()=>hideTip(fig)); hit.addEventListener("blur",()=>hideTip(fig));
    });
  });
}
function drawE10Table(){
  const t=document.getElementById("e10-table"); clear(t);
  const E10NAME={"主方法":"定稿方法","longest_queue":"longest_queue","hol_weighted":"hol_weighted","仅模仿":"仅模仿"}; E10NAME[E10.control]="正向对照：两两干扰 LQ";
  const hr=t.createTHead().insertRow(); const c0=document.createElement("th"); c0.textContent="策略"; hr.appendChild(c0);
  for(const s of E10.scenarios) for(const l of ["交付率差","误包率"]){ const c=document.createElement("th"); c.textContent=`${s.label} · ${l}`; hr.appendChild(c); }
  const tb=t.createTBody(); const names=Object.keys(E10.scenarios[0].fid);
  for(const n of names){ const r=tb.insertRow(); if(n===E10.main) r.className="main"; const a=r.insertCell(); a.textContent=E10NAME[n]||n;
    for(const s of E10.scenarios){ const f=s.fid[n];
      const c1=r.insertCell(), c2=r.insertCell();
      if(!f){ c1.textContent="–"; c2.textContent="–"; continue; }
      const d=f.delivery_ratio, pp=v=>(v*100>0?"+":"")+(v*100).toFixed(2);
      c1.textContent=pp(d.m); const ci=document.createElement("span"); ci.className="ci"; ci.textContent=`[${pp(d.lo)}, ${pp(d.hi)}]`; c1.appendChild(ci);
      c2.textContent=(f.per*100).toFixed(2)+"%"; } }
}

scenButtons();
function drawAll(){ drawForest(); drawDots(); drawE10(); drawCurves(); }
drawE10Table();
drawAll();
heat("heat-table","heat-buttons",DATA.ablations.table,DATA.ablations.components,DATA.ablations.scenarios);
heat("e8-table","e8-buttons",DATA.e8.table,DATA.e8.candidates,DATA.e8.scenarios);
const E11 = DATA.e11;
const E11_ROWS = ["Zhao-GCN","GRLinQ","独立决定PPO","max_weight_opt","backpressure_opt","lq_local_search","backpressure","longest_queue","hol_weighted","oldest_hol","spatial_tdma","random","仅模仿"];
heat("e11-table","e11-buttons",E11.compare.diffs,E11_ROWS.filter(n=>E11.compare.diffs[n]),E11.compare.scenarios,E11.labels);
heat("e11s-table","e11s-buttons",E11.sweeps.diffs,E11_ROWS.filter(n=>E11.sweeps.diffs[n]),E11.sweeps.scenarios,E11.labels);
(function(){
  const t=document.getElementById("e11-opt"); const hr=t.createTHead().insertRow();
  for(const h of ["策略","近似比 · 默认","近似比 · 24 节点","决策 ms · 默认","决策 ms · 24 节点"]){ const c=document.createElement("th"); c.textContent=h; hr.appendChild(c); }
  const tb=t.createTBody(); const rows=Object.entries(E11.optgap).sort((a,b)=>b[1].default.ratio_mean-a[1].default.ratio_mean);
  for(const [n,v] of rows){ const r=tb.insertRow(); if(n==="主方法") r.className="main"; r.insertCell().textContent=n==="主方法"?"定稿方法":((E11.labels&&E11.labels[n])||n);
    for(const k of ["default","nodes24_dense"]) r.insertCell().textContent=v[k]?v[k].ratio_mean.toFixed(3):"–";
    for(const k of ["default","nodes24_dense"]) r.insertCell().textContent=v[k]?v[k].decision_ms_mean.toFixed(1):"–"; }
  if(!E11.ns3) return;
  const wrap=document.getElementById("e11-ns3-wrap");
  const h=document.createElement("h3"); h.textContent="ns-3 复核"; wrap.appendChild(h);
  const p=document.createElement("p"); p.className="prose"; p.textContent=`按预登记，主方法、longest_queue、精确反压和默认场景交付率最高的学习基线（Zhao 等 GCN）在 ns-3 物理层（含帧内运动）上重跑 ${E11.ns3.episodes} 个场景。`; wrap.appendChild(p);
  const tw=document.createElement("div"); tw.className="table-wrap"; const tt=document.createElement("table"); tw.appendChild(tt); wrap.appendChild(tw);
  const hr2=tt.createTHead().insertRow();
  for(const hname of ["策略","场景","交付率 · 轻量","交付率 · ns-3","ns-3 平均时延 (s)","ns-3 误包率","主方法交付率差 · ns-3"]){ const c=document.createElement("th"); c.textContent=hname; hr2.appendChild(c); }
  const tb2=tt.createTBody();
  for(const [n,byScen] of Object.entries(E11.ns3.rows)) for(const [v,x] of Object.entries(byScen)){
    const r=tb2.insertRow(); if(n==="主方法") r.className="main";
    r.insertCell().textContent=n==="主方法"?"定稿方法":((E11.labels&&E11.labels[n])||n); r.insertCell().textContent=LABEL[v]||v;
    r.insertCell().textContent=x.dr_light.toFixed(4); r.insertCell().textContent=x.dr_ns3.toFixed(4);
    r.insertCell().textContent=x.delay_ns3.toFixed(3); r.insertCell().textContent=(x.per*100).toFixed(2)+"%";
    const c=r.insertCell(); const d=x.diff_delivery_ratio; if(d){ const pp=v2=>(v2*100>0?"+":"")+(v2*100).toFixed(2); c.textContent=pp(d.m); const ci=document.createElement("span"); ci.className="ci"; ci.textContent=`[${pp(d.lo)}, ${pp(d.hi)}]`; c.appendChild(ci);} else c.textContent="–"; }
})();
/* ---------- E12b onwards ---------- */
const NAMES = DATA.names;
const ROW_ORDER = ["主方法","主方法·重训","主方法·宽","Zhao-GCN","Zhao-GCN·重训","Zhao-GCN·宽","GRLinQ","GRLinQ·重训","GRLinQ·宽",
  "独立决定PPO","独立决定PPO·重训","独立决定PPO·宽","仅模仿","仅模仿·重训","仅模仿·宽","max_weight_opt","backpressure_opt",
  "lq_local_search","backpressure","longest_queue","hol_weighted","oldest_hol","spatial_tdma","random"];
const rowsOf = obj => ROW_ORDER.filter(n=>obj[n]);
const fpp = v => (v*100>0?"+":"")+(v*100).toFixed(2);
const fs = v => (v>0?"+":"")+v.toFixed(3);
function ciCell(c, d, f){ c.textContent=f(d.m); const s=document.createElement("span"); s.className="ci"; s.textContent=`[${f(d.lo)}, ${f(d.hi)}]`; c.appendChild(s); }
function plainTable(id, headers, rows){
  const t=document.getElementById(id); if(!t) return; clear(t);
  const hr=t.createTHead().insertRow(); for(const h of headers){ const c=document.createElement("th"); c.textContent=h; hr.appendChild(c); }
  const tb=t.createTBody();
  for(const row of rows){ const r=tb.insertRow(); if(row.main) r.className="main";
    for(const x of row.cells){ const c=r.insertCell();
      if(x && x.d){ ciCell(c, x.d, x.f); } else if(x && x.pre!==undefined){ c.textContent=x.pre; const s=document.createElement("span"); s.className="ci"; s.textContent=x.post; c.appendChild(s); }
      else c.textContent=(x===null||x===undefined)?"–":x; } }
}
function effectRows(effect, fams){
  return rowsOf(effect).concat(Object.keys(effect).filter(n=>!ROW_ORDER.includes(n))).map(n=>({main:n==="主方法",
    cells:[NAMES[n]||n, ...fams.map(f=>{ const e=effect[n][f]; if(!e) return null;
      return {pre:`${e.ref.toFixed(3)} → ${e.new.toFixed(3)}`, post:`${fpp(e.delivery_ratio.m)} [${fpp(e.delivery_ratio.lo)}, ${fpp(e.delivery_ratio.hi)}]`}; })]}));
}
(function(){
  const E=DATA.e12b;
  heat("e12b-table","e12b-buttons",E.versus,rowsOf(E.versus),E.columns,NAMES);
  plainTable("e12b-effect", ["策略", ...E.families.map(f=>LABEL[f])], effectRows(E.effect, E.families));
  plainTable("e12b-sens", ["条件","主方法交付率","交付率差（个百分点）","平均时延差（s）","可靠性可接受"],
    E.sensitivity.map(x=>({cells:[x.label, x.dr_main.toFixed(4), {d:x.delivery_ratio,f:fpp}, {d:x.e2e_delay_mean_s,f:fs}, x.delivery_ratio.lo>=-0.005?"是":"否"]})));
  const T=DATA.e13;
  heat("e13-table","e13-buttons",T["主方法"],rowsOf(T["主方法"]),T.columns,NAMES);
  heat("e13rt-table","e13rt-buttons",T["主方法·重训"],rowsOf(T["主方法·重训"]),T.columns,NAMES);
  heat("e13x-table","e13x-buttons",T.extra,rowsOf(T.extra),T.extra_columns,NAMES);
  plainTable("e13-ns3", ["测试飞行","主方法 − LQ · 轻量","主方法 − LQ · ns-3","主方法 ns-3 − 轻量","LQ ns-3 − 轻量"],
    T.ns3.map(x=>({cells:[LABEL[x.fold]||x.fold, {d:x.lightweight.delivery_ratio,f:fpp}, {d:x.ns3.delivery_ratio,f:fpp},
      {d:x["fid_主方法"],f:fpp}, {d:x.fid_longest_queue,f:fpp}]})));
  if(T.ucsb){ const u=T.ucsb[0].models, lab={step:"无衰落门限（阶跃）",rician:"门限 + 莱斯衰落",lognormal:"门限 + 对数正态阴影"};
    const par=(k,p)=>k==="step"?`距离 ${p.range_m.toFixed(0)} m`:k==="rician"?`指数 ${p.exponent.toFixed(2)}，K ${p.k_db.toFixed(1)} dB`:`指数 ${p.exponent.toFixed(2)}，σ ${p.sigma_db.toFixed(1)} dB`;
    plainTable("e13-ucsb", ["链路模型","检验集逐包对数似然","分段收包率平均误差","拟合参数"],
      Object.entries(u).map(([k,v])=>({main:k==="rician", cells:[lab[k]||k, v.test_loglik.toFixed(3), v.test_sector_mae.toFixed(3), par(k,v.params)]}))); }
  if(DATA.e14){ const X=DATA.e14; document.getElementById("e14-wrap").hidden=false;
    heat("e14-table","e14-buttons",X["主方法·宽"],rowsOf(X["主方法·宽"]),X.columns,NAMES);
    heat("e14b-table","e14b-buttons",X.broad_vs_orig,rowsOf(X.broad_vs_orig),X.columns,NAMES); }
  if(DATA.e15){ const Y=DATA.e15; document.getElementById("e15-wrap").hidden=false;
    heat("e15-table","e15-buttons",Y.versus,rowsOf(Y.versus),Y.columns,NAMES);
    plainTable("e15-effect", ["策略", ...Y.families.map(f=>f==="flock"?"群集 30（合并）":LABEL[f])], effectRows(Y.effect, Y.families));
    if(Y.connectivity) plainTable("e15-conn", ["信道","调度器可用的节点对","有路由的业务","平均跳数"],
      Y.connectivity.map(x=>({cells:[x.label, x.usable.toFixed(3), x.routable.toFixed(3), x.hops.toFixed(2)]})));
    plainTable("e15-sigma", ["条件","主方法交付率","LQ 交付率","主方法 − LQ 交付率（个百分点）","平均时延差（s）"],
      Y.sigma.map(x=>({cells:[x.label, x.dr_main.toFixed(4), x.dr_lq.toFixed(4), {d:x.delivery_ratio,f:fpp}, {d:x.e2e_delay_mean_s,f:fs}]})));
    if(Y.ns3){ const w=document.getElementById("e15-ns3-wrap"); const h=document.createElement("h3"); h.textContent="ns-3 复核（C2S5 默认场景族）"; w.appendChild(h);
      const tw=document.createElement("div"); tw.className="table-wrap"; const tt=document.createElement("table"); tt.id="e15-ns3"; tw.appendChild(tt); w.appendChild(tw);
      plainTable("e15-ns3", ["主方法 − LQ · 轻量","主方法 − LQ · ns-3","主方法 ns-3 − 轻量","LQ ns-3 − 轻量"],
        [{cells:[{d:Y.ns3.lightweight.delivery_ratio,f:fpp},{d:Y.ns3.ns3.delivery_ratio,f:fpp},{d:Y.ns3["fid_主方法"],f:fpp},{d:Y.ns3.fid_longest_queue,f:fpp}]}]); } }
})();
let rt; new ResizeObserver(()=>{ clearTimeout(rt); rt=setTimeout(drawAll,120); }).observe(document.querySelector(".page"));
</script>
"""

if __name__ == "__main__":
    main()
