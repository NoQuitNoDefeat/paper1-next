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
    html = TEMPLATE
    for k, v in facts.items():
        html = html.replace("{{" + k + "}}", str(v))
    html = html.replace("/*DATA*/null", json.dumps(d, ensure_ascii=False))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


TEMPLATE = r"""<title>双图微步调度实验</title>
<meta name="description" content="UAV/FANET 集中式 MAC 调度：双图表示 + 微步 PPO 的定稿方法与 E1–E9 实验结果">
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
  <p class="lede">在从未使用过的测试集上，定稿方法在 5 个场景族中都没有比“最长队列优先”少送包，平均时延低 {{delay_lo}}–{{delay_hi}}%，p95 时延低 {{p95_lo}}–{{p95_hi}}%。在更密、更大、更拥堵或信道有误差的场景中，最终交付率还高出 {{dr_hard_lo}}–{{dr_hard_hi}} 个百分点。</p>
  <div class="meta">
    <span>测试：{{episodes}} 个场景 × 5 个场景族，排空 {{drain}} 周期</span>
    <span>主方法 {{seeds_main}} 个训练种子</span>
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
    <li><strong>尾部时延仍有差距。</strong>HOL 加权基线的 p95 时延更低，但交付率显著更低。按“交付率优先”的原则，定稿方法更合适。</li>
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
  <p class="prose">从 E5 起，每一轮实验的方案和判定规则都在看到结果之前写进文档并提交。测试集只在最后使用了一次（E9）。</p>
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
      <p class="out">→ 见图 4。</p></div></div>
    <div class="step"><div class="id">E7</div><div><h3>零样本泛化</h3>
      <p>6 种未见条件中，可靠性全部不劣于基线；在更难的条件下，交付率和时延同时改善。其中“24 节点同密度”变体曾误设为 424 m，已更正为 367.4 m 并补评。</p>
      <p class="out">→ 重负载评估改用 1000 周期排空。</p></div></div>
    <div class="step"><div class="id">E8</div><div><h3>已选摘要改进</h3>
      <p>归一化摘要和控制器动态特征都能进一步降低时延，但在每个场景都以交付率为代价，未通过预先登记的规则。</p>
      <p class="out">→ 经用户确认，定稿方法不使用已选摘要；其余实现保留为可切换的对照。</p></div></div>
    <div class="step"><div class="id">E9</div><div><h3>测试集最终评估 <span class="tag">测试集</span></h3>
      <p>5 个训练种子，{{episodes}} 个测试场景 × 5 个场景族，排空 {{drain}} 周期，预先登记的全部主要判定成立（图 1、图 2）。</p></div></div>
  </div>
</section>

<section>
  <h2>训练过程</h2>
  <figure id="fig-curves">
    <p class="fig-title">图 3　训练中的开发集评估</p>
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
    <p class="fig-title">图 4　去掉或替换部件后的变化（相对当时的完整模型，含门控求和摘要）</p>
    <p class="fig-sub">E6 与 E6b，开发集第 16–47 个场景，每组 3 个训练种子取平均。着色表示差异显著：蓝色为更好，红色为更差；灰色为不显著。</p>
    <div class="controls" role="group" aria-label="选择指标" id="heat-buttons"></div>
    <div class="table-wrap"><table class="heat" id="heat-table"></table></div>
    <div class="swatches"><span><i style="background:color-mix(in oklab,var(--pos) 45%,var(--surface))"></i>显著更好</span><span><i style="background:color-mix(in oklab,var(--neg) 45%,var(--surface))"></i>显著更差</span><span><i style="background:var(--surface);border:1px solid var(--rule)"></i>不显著</span></div>
  </figure>
  <figure id="fig-e8">
    <p class="fig-title">图 5　已选摘要的两种改进方案（相对“无摘要”）</p>
    <p class="fig-sub">E8，判定规则：5 个场景的交付率下界都 ≥ −0.5 个百分点，且平均时延至少在 3 个场景显著更低、在任何场景都不显著更高。两种方案都未通过。</p>
    <div class="controls" role="group" aria-label="选择指标" id="e8-buttons"></div>
    <div class="table-wrap"><table class="heat" id="e8-table"></table></div>
  </figure>
</section>

<section class="prose">
  <h2>局限</h2>
  <ul class="claims">
    <li>只在一个场景族上训练（16 节点、300 m、理想信道、单速率单信道）。困难条件的结果属于零样本泛化。</li>
    <li>只在 Python 轻量环境中评估：同步时序，没有控制时延和报告丢失。无线执行效果需要在 ns-3 上验证。</li>
    <li>环境语义与研究方案第 19 版有两处差异：路由变化后重新安置旧队列；等待区恢复时，若目标队列已满则继续等待。详见 <code>docs/decisions.md</code>。</li>
    <li>比较没有做多重校正。测试集已使用一次，之后的方法调整需要新的独立测试集。</li>
  </ul>
  <h2>复现</h2>
  <p>全部源码、配置、实验方案和文档都在 GitHub 仓库 <code>NoQuitNoDefeat/paper1-next</code> 中。运行结果单独保存在本机 <code>results/</code>。本页由 <code>tools/build_report.py</code> 从结果文件生成。</p>
<pre class="cmd"><code>.venv/bin/fanet-next train --config configs/protocol_final.toml --set seed=0 --run-dir results/runs/final-s0
.venv/bin/fanet-next select --run-dir results/runs/final-s0
.venv/bin/python tools/confirm.py --spec configs/experiments/e9_test.json
.venv/bin/python tools/build_report.py</code></pre>
  <p class="note">实验记录：<code>docs/experiments.md</code>（E1–E9）；实现决定：<code>docs/decisions.md</code>；已知问题：<code>docs/known-issues.md</code>。</p>
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
function heat(tableId, buttonsId, rowsData, rowNames, scenIds){
  let mi=0; const host=document.getElementById(buttonsId);
  const render=()=>{
    const m=HEATM[mi], t=document.getElementById(tableId); clear(t);
    const hr=t.createTHead().insertRow(); const c0=document.createElement("th"); c0.textContent=m.title; hr.appendChild(c0);
    for(const s of scenIds){ const c=document.createElement("th"); c.textContent=LABEL[s]; hr.appendChild(c); }
    const tb=t.createTBody();
    for(const rname of rowNames){ const r=tb.insertRow(); const a=r.insertCell(); a.textContent=rname;
      for(const s of scenIds){ const d=rowsData[rname][s][m.key]; const c=r.insertCell(); c.className="cell";
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

scenButtons();
function drawAll(){ drawForest(); drawDots(); drawCurves(); }
drawAll();
heat("heat-table","heat-buttons",DATA.ablations.table,DATA.ablations.components,DATA.ablations.scenarios);
heat("e8-table","e8-buttons",DATA.e8.table,DATA.e8.candidates,DATA.e8.scenarios);
let rt; new ResizeObserver(()=>{ clearTimeout(rt); rt=setTimeout(drawAll,120); }).observe(document.querySelector(".page"));
</script>
"""

if __name__ == "__main__":
    main()
