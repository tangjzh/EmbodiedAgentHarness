# Embodied Agent Harness

用真实 Claude Code 在独立机器人任务中观察、写 Python、执行、读取反馈并继续调试。每个任务从空工作区开始，**没有跨任务 codebase 迁移、历史对话继承或预装任务解法**。同一任务内部可以维护自己的代码和记忆。

收录档案站的非 RoboCasa 任务：**34 个实际 Agent 任务 + 1 个仅环境自检的衣物案例**，覆盖 dm_control、Hopper、ManiSkill、DexMimicGen、DLO-Lab、Factory、RoboDojo、DexJoCo、LIBERO/LIBERO-PRO、robosuite。

## 快速开始

Linux + Python 3.11。按环境组安装，避免 MuJoCo / NumPy / Isaac 的依赖冲突。

~~~bash
git clone https://github.com/tangjzh/EmbodiedAgentHarness.git
cd EmbodiedAgentHarness
python3.11 -m venv .venvs/state
.venvs/state/bin/pip install -e . -r requirements/state.txt
.venvs/state/bin/python -m eca list
.venvs/state/bin/python -m eca doctor finger_spin
~~~

Claude Code 是外部闭源依赖，不随仓库分发。原实验为 2.1.304，doctor 核对 SHA256。导出自己的端点和凭据：

~~~bash
export ECA_CLAUDE_BIN=/absolute/path/to/claude
export ANTHROPIC_BASE_URL=https://api.anthropic.com
export ANTHROPIC_API_KEY=YOUR_KEY
~~~

~~~bash
# 只看计划，无模型/仿真调用
python -m eca develop finger_spin --out outputs/finger-fresh --dry-run
# 真正让 Coding Agent 从零开发
python -m eca develop finger_spin --out outputs/finger-fresh
python -m eca summarize outputs/finger-fresh
# 显式重放该任务自己的历史冻结程序，无模型调用
python -m eca replay finger_spin --out outputs/finger-reference
~~~

输出目录不可复用。新 Agent 只收到当前任务的公共模型/传感器说明与新工作区，reference_runs 不挂载进去。

## 视觉任务

~~~bash
docker build -f docker/Dockerfile.agent -t eca-agent:2026-11 .
python -m eca qualify maniskill_peg --out outputs/peg-check
python -m eca develop maniskill_peg --out outputs/peg-fresh
python -m eca replay maniskill_peg --out outputs/peg-reference
~~~

需要相应模拟器、上游 clone 和资产。固定版本、渲染、空间服务及安装步骤见 [环境说明](docs/ENVIRONMENTS.md)。各任务预算、观测权限用 python -m eca show TASK 查看。

## 保存的历史效果

| 任务 | 历史结果 | 口径 |
|---|---:|---|
| Finger spin | 94.77% | 3 留出平均奖励时间占比 |
| Ball in cup | 98.4% | 3 留出平均奖励时间占比 |
| Acrobot | 697.018 | 3 留出平均原生回报 |
| Hopper | 1135.301 | 3 留出平均原生回报 |
| ManiSkill 侧向插销 | 3/3 | 冻结程序原生成功 |
| DexMimicGen 双手穿环 | 3/3 | 冻结程序原生成功 |
| DLO 解绳 | 0.848057 | 连续原生分数，无二元阈值 |
| Factory 螺母 | 3/3 | 原始独立单任务程序 |
| Factory 插销 / 齿轮 | 10/10、9/10 | 空前缀 / fresh 条件最终程序 |
| RoboDojo InsertKey | 1/3 | 原生正式成功，阶段分另列 |
| RoboDojo SweepBlocks | 3/3 | 实际策略为逐块抓放 |
| DexJoCo 锤钉 | 开发中成功 | 历史留出首动作前崩溃，无有效独立成功率 |
| Garment fling | 未调用 Agent | 原生环境自检失败，禁止付费启动 |

**以上是保存的历史证据，不是本次发版后重新测出的成绩。** LLM 重新开发有随机性；固定程序重放用于核对原生结果。

早期 LIBERO/CaP-X 使用过多种模型循环和演化接口。仓库保留历史代码和来源；新的 develop 使用固定上游 API 和空工作区。旧接口尚未统一成等价重放，因此该组不提供冒充旧分数复现的 replay。新入口需要在对应仿真与感知服务上重新测量。

## 目录

- eca：命令、空工作区启动、环境检查、输出预算、无模型重放。
- src：实际实验提取的 MCP、robot RPC、观测与原生执行器。
- tasks/catalog.json：35 个任务契约；排除 RoboCasa。
- references：Agent 可读的公共模型/传感器资料。
- reference_runs：操作者显式重放的任务本地历史程序，不用于 Agent 初始化。
- provenance：上游 commit、harness 与参考程序哈希。

~~~bash
python -m unittest discover -s tests -v
~~~

本仓库不启动迁移课程，也不改变原实验或正在运行的任务。视频与逐工具源码见 [实验档案站](https://embodied-agent-task-atlas.t1357489297.chatgpt.site)。
\n## 本次发布的实际验证\n\n9 项契约检查通过。新仓库中实际重放了四个状态任务、每个3个原生留出回合：Finger94.7667%奖励时间、Ball98.4%奖励时间、Acrobot697.017775平均回报、Hopper1135.301385平均回报；与历史记录一致。均无新增模型调用。详细 seeds、逐回合数值和执行有效性见 docs/VALIDATION.json。GPU视觉任务与新的LLM从零开发未在本次发版整批重跑。\n