# 代码架构与可替换组件

本文说明代码如何组织、各模块的契约，以及怎样新增一种实现。研究含义见[研究方法](research-method.md)，选择与依据见[实现决定](decisions.md)。

## 1. 一个周期在代码中的流向

```
场景(外生过程) ──► 后端 ──Report──► 候选规则 ──► SchedulingProblem
                    ▲                  │                │
                    │             观测构建器          约束集合
                    │             (双图)            (微步控制器)
                    │                  └──────► 策略 ◄──┘
                    │                            │ 有序选择
                    └────────── Plan ◄───────────┘
                    │
                    └── CycleFacts ──► 奖励 / 指标
```

`loop.SchedulingEnv` 是唯一把各组件连起来的地方。每个模块只接触自己的契约（`contracts.py`）：

| 契约 | 方向 | 内容 |
| --- | --- | --- |
| `Report` | 后端 → 决策侧 | 位置/速度、信道估计 `gain`、队列快照、路由表、等待区；只含决策时可获得的信息 |
| `SchedulingProblem` | 候选规则 → 约束/策略 | 局部候选编号 ↔ 稳定有向链路 `(tx, rx)`、增益、功率、门限 |
| `DualGraph` | 观测 → 模型 | 通信图、候选映射 `cand_link/cand_edge`、链路交互图、全局特征 |
| `Plan` | 策略 → 后端 | 本周期完整计划（有序链路集合） |
| `CycleFacts` | 后端 → 奖励/指标 | 实际成功/失败、逐队列服务字节、服务后快照、终止、交付 |
| `StepOutcome.end` | 后端 → 训练 | `continue` / `truncated` / `terminated`；执行错误单独抛 `ExecutionError` |

## 2. 槽位（可替换职责）

每类可替换职责是一个“槽位”：一张“名字 → 实现”的小表（`registry.py`）。配置中的 `type` 选择实现。`fanet-next components` 列出全部槽位、实现及其角色（primary 主方法 / baseline 基线 / control 对照 / variant 变体）。

| 职责（module-design.md） | 槽位 | 位置 | 当前实现 |
| --- | --- | --- | --- |
| 场景与业务 | `scenario`、`routing`、`channel` | `scenario/` | 场景：random（主；节点数、速度、负载、场地边长都可以取范围，每回合抽样）、fixed（脚本化）、trace（变体：回放记录的轨迹，即公开飞行日志或第三方移动模型的输出，按整条轨迹划分训练、开发和测试，decisions.md 第 11 节）、mixture（变体：多个场景分布按种子混合，只用于 E14 探索）。路由：min_hop。信道：ideal（主）、lognormal、rician（变体：空空实测参数，莱斯块衰落加衰落裕量，可选阴影及其两种口径，decisions.md 第 10 节） |
| 环境与执行后端 | `backend` | `backend/` | lightweight（主）；ns3（变体：ns-3.48 物理层执行 + 帧内运动，或用于对齐的 SINR 账本；见 decisions.md 第 8 节） |
| 观测与双图 | `observation` | `observation/` | standard |
| 模型 | `model` 及 `comm_encoder`、`lift`、`interaction_encoder`、`set_summary`、`actor_head`、`critic_head` | `model/` | dual_graph；各子部件都有主实现和消融对照。`set_summary`：none（主）、gated_sum（对照，研究方案原设计）、gated_mean（变体）；模型选项 `candidate_dynamics` 接入控制器逐步提供的候选特征（变体） |
| 微步调度与约束 | `candidates`、`resource`、`interference` | `scheduling/` | standard；half_duplex；full_sinr（主）+ pairwise / new_link_only / none（对照）；`maxweight.py` 为基线提供精确最大权（MILP）与局部搜索 |
| 决策策略 | `policy` | `policy/` | ppo（主）；其余均为基线：启发式 random、longest_queue、oldest_hol、hol_weighted；经典 max_weight_opt、backpressure、backpressure_opt、lq_local_search、spatial_tdma（`classical.py`）；前人学习方法 zhao_gcn、oneshot_ppo、grlinq（`baselines/`，自带模型与训练，用 `fanet-next train-baseline` 训练）。定义见 decisions.md 第 9 节 |
| 奖励与指标 | `reward` | `reward/` | standard（四分项）；指标在 `reward/metrics.py` 单独记录 |
| 训练 | `trainer` | `training/` | ppo（时变折扣 GAE、微步重算） |
| 实验与评估 | （非槽位） | `experiment/` | 配置装配、兼容性检查、训练/续训/评估命令行 |

关键边界：

- **模型与微步控制。** `scheduling.MicroStepController` 维护已选集合、资源、累计干扰和掩码，不依赖模型。
  - 模型只通过 `model.SchedulingModel` 的五个方法被使用：`encode`、`init_state`、`update_state`、`logits`、`value`。
  - 采样、PPO 重算和评估共用 `model/runner.py`。
- **决策与执行。** 策略只产生 `Plan`；后端是推进包状态的唯一权威。
- **事实与奖励。** 奖励只读 `CycleFacts`，并保留各分项（`RewardBreakdown.parts`）；评价指标独立记录。

## 3. 新增一种实现

1. 在对应职责的包里写类，并用 `@SLOT.register("name", role=...)` 装饰。
   - 构造函数的关键字参数就是配置参数。
   - 依赖（模型维度、特征 schema 等）由调用方以关键字参数注入。
2. 满足该槽位抽象基类的契约（`Constraint`、`Policy`、`ChannelModel`、`Routing`、`ObservationBuilder`、`SchedulingModel` 等）。
3. 在配置中选用（`[interference] type = "name"` 或 `--set constraints.interference=name`）。
4. 运行 `pytest`。`tests/test_components.py` 会遍历每个已登记的实现，新实现自动进入契约测试：
   - 计划满足半双工且是极大集合（`maximal_plans = False` 的策略只检查可行，如空间 TDMA）；
   - 主约束满足完整 SINR；
   - 包守恒；
   - 场景格式正确；
   - 每个模型变体的采样与重算一致，且梯度能回传。
5. 如果新组合的不匹配无法被兼容性检查（`experiment/assemble.check_compatibility`）发现，就把检查补到那里。

新实现只要被导入就可用。没有插件发现或热加载。

## 4. 运行记录与续训

每个训练运行目录（`results/runs/<name>/`）包含：

- `config.json`：完整解析后的配置。
- `meta.json`：
  - 组件清单（每个槽位用了哪个实现及其角色）；
  - 代码指纹（git commit 和源码 sha256）；
  - 版本、种子、训练参数。
- `code_snapshot/`：启动时的 src 与 configs 副本。
- `train_log.jsonl` / `eval_log.jsonl`：逐迭代诊断和 dev 评估。
- `checkpoints/latest.pt` 和定期保存的 `iter_XXXXX.pt`。

**续训保证：** 在本版本、相同代码与配置、CPU、**单线程**（`training.threads = 1`）条件下，于迭代边界续训是逐位精确的。

- 多线程时，CPU 浮点归约的顺序不固定：同一配置运行两次，参数就会有 1e-7 量级的差异，并随训练逐渐放大。这时续训与不中断训练在统计上等价，但不逐位相同。
- 检查点保存：
  - 模型和优化器；
  - 全部随机状态（PPO 小批量顺序、采样、torch/numpy/python）；
  - 奖励标准化器和训练种子流；
  - 完整环境对象（轻量后端支持）。
- `tests/test_resume.py` 在单线程下检查“训练 2 次、保存、续训 1 次”与“直接训练 3 次”的权重逐位相同。

- 如果后端不支持保存状态，或传入 `--fresh-envs`，运行中的回合会重新开始，学习状态照常恢复。
- 改动模型或特征配置后续训、或使用其他版本的检查点，不在支持范围内。
- 续训时若源码指纹与检查点不同，会给出警告。

种子按用途分段：

| 用途 | 种子范围 |
| --- | --- |
| 训练 | `1_000_000 + 100_000·seed + k` |
| 开发评估（dev） | `10_000_000 + i` |
| 最终测试（test） | `20_000_000 + i` |

## 5. 常用命令

```bash
.venv/bin/fanet-next components
.venv/bin/fanet-next train --config configs/base.toml --run-dir results/runs/base-s0
.venv/bin/fanet-next resume --run-dir results/runs/base-s0 --iterations 300
.venv/bin/fanet-next select --run-dir results/runs/base-s0            # 固定规则选检查点，写 selection.json
.venv/bin/fanet-next eval --run-dir results/runs/base-s0 --policies ppo longest_queue random --split dev
.venv/bin/fanet-next eval --config configs/base.toml --set constraints.interference=pairwise --policies longest_queue
.venv/bin/fanet-next train-baseline --config configs/baselines/zhao_gcn.toml --run-dir results/runs/zhao-s0
.venv/bin/python -m pytest -q
```

- `--scenario 文件` 用另一个配置文件的 `[scenario]` 段整段替换场景，可用于 `train`、`train-baseline`、`eval`。例如在群集轨迹上训练：`--scenario configs/datasets/flock30.toml`。
- `--set a.b=值` 在此基础上改单个参数。换 `type` 时，该段的旧参数会被清空（known-issues.md 第 14 条）。
- 实验脚本在评估任务里写 `--checkpoint @selected`，启动前由脚本（`tools/confirm.py` 的 `resolve`）换成运行目录 `selection.json` 中选中的检查点；命令行本身不认这个写法。

## 6. 实验脚本与规格

实验的场景、策略和测试种子写在 `configs/experiments/*.json` 中，脚本按规格生成评估任务。输出已存在的任务会跳过，所以中断后重跑只补缺的部分。

**评估规格**的主要字段：

| 字段 | 含义 |
| --- | --- |
| `eval` | `split`（test）、`seed_offset`、`episodes`、`drain`、`shards` |
| `executors` | `{"lightweight": null}`，或再加 `{"ns3": "ns3"}` 用 ns-3 执行 |
| `scenarios` | 场景名 → `--set` 列表，或 `{"scenario": 配置文件, "set": [...]}` |
| `baselines` / `baseline_config` | 无需训练的策略及其配置 |
| `groups` | 学习型策略：运行目录列表（各训练种子），或“场景 → 运行目录”（按场景分别训练，如按折） |
| `fixed` | 单个固定检查点 |
| `shard_policies`、`exclude` | 慢策略的分片；某场景不跑的策略 |

**训练规格**（`*_train.json`，由 `tools/train_runs.py` 读取）：方法（命令、种子、相对耗时、估计内存、优先级）与训练场景的交叉。中断的运行会续训；训练结束后立即按固定规则选检查点。

| 脚本 | 用途 |
| --- | --- |
| `tools/e11_compare.py --spec S` | 评估一个规格，写出 `summary.md`（主方法相对每个策略的配对差） |
| `tools/run_specs.py S1 S2 ...` | 多个规格共用一个队列，最长的任务先跑，最后各写 `summary.md` |
| `tools/e10_ns3.py --spec S` | 含 ns-3 执行的规格；`--status --watch 30` 显示进度 |
| `tools/train_runs.py T1 T2 ...` | 按内存上限并行训练、续训、选检查点 |
| `tools/e12_channel.py`、`e13_data.py`、`e14_summary.py`、`e15_summary.py` | 各实验预登记比较的汇总（多场景合并、配对差、差中差） |
| `tools/build_report.py` | 从结果文件生成报告页 `results/report/report.html` |

结果写在 `results/` 下，不进版本库。汇总中的数字已抄入 `docs/experiments.md`。
