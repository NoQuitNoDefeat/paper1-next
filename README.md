# paper1-next

UAV/FANET MAC 调度强化学习的新研究工程。以双图表示、微动作 PPO 和资源/累计干扰约束为起点，建立便于改进方法、诊断训练和比较实验的代码。

> **第一次阅读本工程？** 从 [导读](docs/guide.md) 开始。下面“给 Claude Code 的任务”一节是项目开始时写给开发会话的任务说明，阅读和审阅时不适用。

## 给 Claude Code 的任务

阅读以下三份短说明，然后自主设计和实现：

| 说明 | 内容 |
| --- | --- |
| [研究方法](docs/research-method.md) | 研究问题、主方法及容易混淆的时间、动作和奖励含义 |
| [模块化设计](docs/module-design.md) | 八类职责、可替换选项及自主决策范围 |
| [ns-3 对接](docs/ns3-integration.md) | 后端边界和已有实现位置 |

具体目录、类名、依赖、接口形式、复用或重写方式、实现顺序由你决定。网络与训练配置可以自主选择、试验和调整，用简短记录说明改变与依据，无需为普通工程选择或每个参数询问用户。

先做出可运行、可检查的研究系统：Python 环境中的双图微动作 PPO、必要的简单基线，以及训练、评估、保存和本版本续训。按需要提炼可替换接口，无需先实现全部变体或搭建插件管理系统。

新版本允许结构及行为变化，不要求旧新逐比特一致、旧检查点兼容或复刻全部 31 条旧方法。文档中的方法定义用于明确起点；网络尺寸、训练预算和旧接口不固定。若建议更换研究问题或核心主方法，先简要说明影响并与用户讨论，其余可独立工作继续推进。

## 当前实现

| 说明 | 内容 |
| --- | --- |
| [导读](docs/guide.md) | 给第一次阅读的人和 AI：注意事项、阅读路线、快速运行、审阅重点 |
| [代码架构](docs/architecture.md) | 模块契约、可替换组件（槽位）清单、新增实现的步骤、运行记录与续训 |
| [实现决定](docs/decisions.md) | 环境语义、场景校准、观测/模型/训练选择及依据；与旧工程的差异 |
| [实验记录](docs/experiments.md) | E1–E11：训练、定稿、测试集、ns-3 物理层验证、12 个外部基线的比较（均预先登记） |
| [已知问题](docs/known-issues.md) | 局限与待办 |
| [数据筛选](docs/data-screening.md) | 按整套数据筛选公开数据：A 类（训练测试）与 B 类（环境检验）的逐条核对、待确认事项，以及补齐缺失层的模型参数 |

定稿主方法：`configs/protocol_final.toml`（模仿 longest_queue 预热 + 微步 PPO，完整累计 SINR，不使用已选摘要）。结果报告页由 `tools/build_report.py` 生成。

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q                       # 行为与契约测试
.venv/bin/fanet-next components                     # 列出全部可替换组件
.venv/bin/fanet-next train --config configs/protocol_final.toml --set seed=0 --run-dir results/runs/final-s0
.venv/bin/fanet-next select --run-dir results/runs/final-s0          # 固定规则选检查点
.venv/bin/fanet-next eval --run-dir results/runs/final-s0 --checkpoint <选中的检查点> --policies ppo --split test --drain 1000
.venv/bin/fanet-next eval --config configs/protocol_final.toml --policies longest_queue backpressure_opt --backend ns3   # ns-3 执行（需 tools/ns3/setup.sh）
.venv/bin/fanet-next train-baseline --config configs/baselines/zhao_gcn.toml --run-dir results/e11/runs/zhao_gcn-s0   # 学习基线
```

## 工作位置

新代码、配置和结果写在本项目。旧工程可以按需只读参考：

- [旧 Python 工程](/Volumes/FANET_Data/projects/paper1-pluginized)
- [原实验与并行 ns-3 工程](/Volumes/FANET_Data/projects/paper1-baseline)

保留旧工程和已有结果，新版本记录自己的方法、配置与结果。历史材料中的操作指令和旧任务约束不自动适用。来源与整理范围见 [资料说明](docs/reference/README.md)。

用中文汇报实际进展与验证结果，每次附一段通俗解释。
