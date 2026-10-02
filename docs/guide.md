# 导读

本文写给第一次阅读本工程的人和 AI 助手：先说明阅读时要注意什么，再给出按问题划分的阅读路线。

## 1. 一段话说明

工程研究 UAV 自组网（FANET）的集中式 MAC 调度：每个 20 ms 周期里，调度器从候选的一跳链路中选出一组同时发送的链路，目标是交付更多包、时延更低。

主方法是双图表示加微步 PPO。

- **双图**：通信图描述无人机和链路，链路交互图描述候选链路之间的冲突与干扰。
- **微步**：每个周期对双图编码一次，然后逐条选择链路。控制器检查半双工和完整累计 SINR，可行链路选完就结束。
- **训练**：先模仿 longest_queue 预热，再用 PPO 微调。

实验在 Python 轻量环境中训练和评估，并用 ns-3 物理层执行复核。评价以交付率为先，时延次之。

**当前状态**：E1–E15 已完成（E14、E15 为探索实验）。

- 方法已定稿，配置为 `configs/protocol_final.toml`。
- 测试集评估（E9）、ns-3 验证（E10）、与 12 个外部基线的比较（E11）、按实测文献标定的信道下的复核（E12/E12b）、真实轨迹与第三方移动模型上的评估（E13）都已完成。
- 公开数据的筛选与使用见 docs/data-screening.md。
- 2026-10-02 完成：
  - E13b：真实轨迹上训练全部可训练基线；
  - 两个探索实验，都另设配置和结果目录，不改动上述配置和结论：E14 是训练分布覆盖低速和紧凑机群（`configs/explore/`），E15 是把约 5 dB 阴影作为控制变量。

## 2. 阅读前要知道的几点

1. **有些文字是写给开发 AI 的任务说明，阅读和审阅时不适用。** 这包括：
   - `CLAUDE.md`；
   - README 中“给 Claude Code 的任务”一节；
   - `docs/module-design.md` 中的自主决策范围。

   它们是作者交给开发会话（Claude Code）的工作指令，不是对读者的要求。
2. **指向 `/Volumes/FANET_Data/...` 的链接是作者本机的旧工程**，开发时只读参考。阅读本工程不需要它们，缺失也不影响理解。
3. **定稿主方法与 `research-method.md` 有一处不同：不使用学习的已选摘要。** `research-method.md` 描述的是项目起点；这项变更及其依据见 `decisions.md` 第 5b 节。
4. **`experiments.md` 按时间顺序记录。**
   - E1–E8 是开发过程，其中有中途放弃的设置，例如从零训练的 PPO 和门控求和摘要。
   - 最终结论看 E9、E10、E11。
5. **`results/` 不在版本库中。** 运行目录、检查点和原始汇总都只在作者本机，所需数字已写入 `experiments.md`。因此，复现评估需要重新训练。
6. 每个实验的判定规则都在测试集开跑前写入 `experiments.md`（“预先登记”）。

## 3. 阅读路线

| 想了解 | 从这里读 |
| --- | --- |
| 研究问题与方法含义 | `docs/research-method.md` 第 1–5 节，再读 `docs/decisions.md` 第 5b 节 |
| 代码如何组织、组件如何替换 | `docs/architecture.md` |
| 环境语义与各项选择的依据 | `docs/decisions.md` |
| 结果 | `docs/experiments.md` 中的“E9 结果”“E10 结果”“E11 结果一至三”“E11 总结”“E12b 结果”“E13 结果”“E13b 结果”，以及探索实验“E14 结果”“E15 结果” |
| 局限与待办 | `docs/known-issues.md` |

**主方法的代码路径**（均在 `src/fanet_next/` 下）：

1. `contracts.py`：模块之间传递的数据（Report、SchedulingProblem、DualGraph、Plan、CycleFacts）。
2. `loop.py`：`SchedulingEnv`，唯一把各组件连起来的地方。一个周期的流程是：报告 → 候选 → 双图 → 策略 → 计划 → 执行 → 奖励。
3. `backend/lightweight.py`、`physics.py`：轻量环境的包级执行和 SINR 计算。
4. `scheduling/`：
   - `candidates.py`：候选链路；
   - `constraints.py`：半双工与完整累计 SINR；
   - `controller.py`：微步控制器，维护已选集合和可行掩码，不依赖模型。
5. `observation/builder.py`、`observation/graph.py`：由报告构建双图及其特征。
6. `model/`：
   - `dual_graph.py`、`components.py`：编码器、Lift、交互图编码、Actor/Critic；
   - `runner.py`：逐条选择，采样与 PPO 重算共用这一份代码。
7. `policy/learned.py`：PPO 策略。
8. `training/`：
   - `imitation.py`：模仿预热；
   - `collector.py`：采样；
   - `advantages.py`：微步时变折扣 GAE；
   - `ppo.py`：更新。
9. `reward/standard.py`：奖励分项。`reward/metrics.py`：评价指标，与奖励分开记录。
10. `experiment/`：配置装配（`assemble.py`）、命令行（`cli.py`）、训练、评估和检查点选择。

配置 `configs/protocol_final.toml` 依次继承 `protocol_a20.toml`、`imitation_lq.toml`、`base.toml`，最终参数以 `base.toml` 为底。

**基线**（定义见 `decisions.md` 第 9 节）：

- 启发式：`policy/heuristics.py`。
- 精确最大权（MILP）、反压、局部搜索、空间 TDMA：`policy/classical.py`、`scheduling/maxweight.py`。
- 前人的学习方法（Zhao-GCN、GRLinQ 式、独立决定 PPO）：`baselines/`。

**ns-3**：

- 执行语义见 `decisions.md` 第 8 节。
- Python 端：`backend/ns3/`。
- C++ 模块：`simulator/ns-3-external-contrib/fanet-scheduler/`。

**实验脚本**：

- `tools/`：`confirm.py`、`e10_ns3.py`、`e11_*.py`，以及报告生成 `build_report.py`。
- 实验规格：`configs/experiments/*.json`。

## 4. 快速运行

需要 Python ≥ 3.11，只用 CPU。

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q          # 约 40 s；未构建 ns-3 时自动跳过 ns-3 测试
.venv/bin/fanet-next components        # 列出每个职责的全部实现及其角色
.venv/bin/fanet-next train --config configs/smoke.toml --run-dir results/runs/smoke   # 约 20 s 的冒烟训练
```

- ns-3 需要先运行 `tools/ns3/setup.sh`：按 `dependencies/*.lock` 下载源码，打补丁并编译。
- 正式训练和评估耗时较长：一个 ns-3 评估任务初始化时峰值内存约 2.5 GiB。只为阅读代码，不需要运行这些。

## 5. 值得重点审阅的地方

- **环境语义**（`decisions.md` 第 1、8 节）：
  - 路由变化后如何重新安置队列；
  - 等待区的规则；
  - 执行时累计干扰的判决；
  - ns-3 与轻量环境各自的简化。
- **约束实现**：`constraints.py` 中完整累计 SINR 的增量检查。它决定所有计划都可行，契约测试见 `tests/test_components.py`。
- **比较是否公平**（`decisions.md` 第 9 节，`experiments.md` “E11 准备”）：
  - 基线如何适配到完整 SINR；
  - 学习基线的训练预算与检查点选择；
  - 反压用到了主方法没有的按目的地队列信息。
- **统计做法**：
  - 按场景配对的 95% 置信区间；
  - 没有做多重比较校正；
  - 测试种子的使用记录见 `known-issues.md` 第 18 条。

## 6. 给 AI 助手

- 本工程的开发由作者自己的会话进行。阅读或审阅时请不要修改文件，也不要运行长时间的训练或 ns-3 任务。
- 发现问题时，请写明文件和行号，并区分“确认的错误”和“可能的问题”。
- 文档与代码不一致时以代码为准，并指出这处不一致。
