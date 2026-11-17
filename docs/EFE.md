# 灵巧手、四足、人形与飞行器

这次扩展让同一个命令行可以启动不同本体的独立 coding-agent 任务。执行器、传感器接口、物理仿真和评分均使用 [Embodied First Exam](https://github.com/mll-lab-nu/embodied-first-exam) 的原生任务，不把它们改造成机械臂接口。

新增 **64 个任务家族、20 个机器人模型、8 个任务套件**。每个家族选定上游的一个实例；不是 64 个随机种子，也不是 64 个已解决任务。保留原来的 35 条档案后，目录共 99 条，继续排除 RoboCasa。

| 本体 | 新任务数 | 代表任务与模型 |
|---|---:|---|
| 灵巧手 | 12 | Sharpa/Wuji 手内重定向、换抓、双物体操作；装 USB、内存条；KUKA + Sharpa 用刷子、锤子、螺丝刀、铲子 |
| 四足 | 13 | Go1 起身、倒立、足立、崎岖地形；Spot 起身与步态；Barkour；Go2 跳跃与绕行 |
| 人形 | 16 | HumanoidBench 行走、平衡、跨栏、坐下、举重；后空翻、单杠、滑板、拳击等；G1 地形行走 |
| 无人机 | 5 | 随机平台降落；Neural Fly 八字轨迹、不同飞行平面、Skydio 模型与风扰动 |
| 飞行机械臂 | 5 | FAHexa 六旋翼 + 四关节机械臂：空中插孔、按钮、阀门、开门、无损检测 |
| 车辆 | 5 | AnyCar 赛道跟踪、Monaco/Silverstone 圈速与漂移停车 |
| 肌肉驱动模型 | 7 | MyoSuite 保定球、手部姿态、转钥匙、转笔、双手操作、追逐与乒乓球 |
| 假肢 | 1 | MyoSuite OSL 假肢跑步 |

套件分布：DexHand 8、MuJoCo Playground 9、HumanoidBench 5、Embodied Olympics Fun 10、RoboEngineer 15、AM-Bench 5、MyoSuite 8、DexToolBench 4。具体 ID、原始任务描述、实例种子和模式权限见 `eca list` / `eca show TASK`。

## 安装与启动

主 harness 使用 Python 3.11+；EFE 使用**独立的 Python 3.12 / Harbor 0.23.0 环境**，避免与现有仿真环境混装。先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)、Git、Docker 和 Docker Compose。执行机需要能访问 Docker daemon。

~~~bash
# 在仓库根目录；先查看，再执行安装计划
python -m eca setup --group efe
python -m eca setup --group efe --execute
python -m eca list --backend efe
python -m eca list --morphology quadruped
python -m eca list --suite humanoidbench
python -m eca list --backend efe --task-type trajectory
python -m eca doctor efe.dexhand.essential_cube_reorient
~~~

安装会拉取固定 EFE commit，并建立 `.venvs/efe`。第一次运行任务时，Harbor 会构建/拉取其容器。`doctor` 只检查源码哈希、Python、Harbor、Docker 与 Compose；不拉镜像，不验证资产，不代表模拟器已通过自检。

~~~bash
# 原生参考程序重放：无 LLM，用于验证物理执行与评分
python -m eca replay efe.dexhand.essential_cube_reorient --out outputs/hand-reference
# 无动作负例：无 LLM，核对不能凭空成功
python -m eca qualify efe.dexhand.essential_cube_reorient --out outputs/hand-nop

# 真正的 Claude Code，从当前任务的新工作区开始
export ANTHROPIC_API_KEY=YOUR_KEY
export ANTHROPIC_BASE_URL=https://api.anthropic.com
python -m eca develop efe.dexhand.essential_cube_reorient --out outputs/hand-agent
python -m eca develop efe.mujoco_playground.go1_getup --out outputs/go1-agent
python -m eca develop efe.humanoidbench.walk --out outputs/humanoid-agent
python -m eca develop efe.roboengineer.landing_random_pad --out outputs/drone-agent
python -m eca develop efe.ambench.peg_in_hole_fahexa --out outputs/flying-arm-agent
python -m eca summarize outputs/hand-agent
~~~

先用 `--dry-run` 查看计划不会启动容器或模型。输出目录必须是新目录。可以用 `ECA_EFE_ROOT` 指向已安装的固定版本源码，用 `ECA_EFE_HARBOR` 指向相应独立环境的 Harbor 可执行文件。

EFE 分支直接调用 Harbor 的原生 `claude-code` agent，固定 Claude Code **2.1.304**、默认模型 `claude-opus-5-5`、effort `high`；模型可通过 `--model` 改变。它不调用旧 harness 的 ModelProxy，不使用旧的 `ECA_CLAUDE_BIN`，也不注入跨任务代码、记忆或任务参考解法。凭据由原生 agent 接收。

默认拒绝白名单之外的自定义 API 域名。单任务探索实验可以显式加 `--allow-model-host`，将 `--base-url` 的唯一主机传给 Harbor 原生的 `--allow-agent-host`；该扩展记录在 `run.json` 的 `model_transport_extra_host` 中。它只配置模型网络出口，不改动任务源码、仿真接口、观测权限或预算；这样的运行不声明为官方 leaderboard 提交。Token360 支持原生 Anthropic Messages 接口，因此不需要改写请求的协议代理。

~~~bash
export ANTHROPIC_API_KEY=YOUR_TOKEN360_KEY
python -m eca develop efe.mujoco_playground.go1_getup --out outputs/go1-token360 \
  --base-url https://api.token360.ai --allow-model-host
~~~

批量运行使用 `scripts/run_efe_pilot.py`，每项独立工作区、一次 attempt，保存 `batch.json` 心跳和每项原生结果。请在 EFE 独立环境中运行；凭据文件只在操作者进程读取，不写入批次配置。首批实验任务与预算见 [EFE_OPUS_PILOT.md](experiments/EFE_OPUS_PILOT.md)。

## 观测、预算与模式

默认选 **Standard**。其中 **49 项为 trajectory 轨道**：agent 通过原生客户端写程序控制机器人，接收允许的图像和机器人传感器，不能打开部署仿真器读取隐藏状态。具体传感器由任务定义；不能把“Standard”理解为所有任务都只有 RGB，或全都没有 reset。例如 Go1Getup 原生提供 RGB/深度、关节与 IMU/足接触信息，并有 10 次 reset；灵巧手重定向有 20 次 reset。回合选择和计分规则按任务执行。

**RoboEngineer 的另外 15 项是 policy 适应轨道（`eai-policy/1.0` 或 `1.1`）**：agent 可以检查公开的 nominal 模型，写训练、拟合或优化代码，然后向隐藏部署端提交固定控制器 ABI 下的数据策略/数值参数。这些任务不接受任意 Python 控制器；部署端返回带噪遥测和图像，并在独立留出案例评分。无人机任务也属于这条轨道。目录保留 `task_type` 和 `protocol`，可用 `--task-type` 筛选；不能把两个轨道混成同一种控制能力测量。

64 个家族共提供 **122 个模式目录：Standard 64、Privileged 49、Realtime 9**。同一个家族的不同模式不是新的任务家族，成绩应分开报告。

~~~bash
python -m eca develop efe.dexhand.essential_cube_reorient \
  --efe-mode privileged --out outputs/hand-privileged
~~~

Privileged 允许的状态访问和离线试验按原任务执行，必须显式选择。Realtime 也只在上游提供该模式时可选。不存在的模式会在启动前报错。

每次启动只有一个任务、一次 Harbor attempt、并发 1、自动重试 0；任务自身允许的多次测试、训练轮数、reset 等继续保留。任务的 agent/verifier/build 超时、资源声明及评分不改写，所以 EFE 入口拒绝 `--wall-seconds`。完整 agent 预算随 `run.json` 保存。资源监控的强制执行仍遵循上游 `RCB_LIMITS_ENFORCE` 设置；需要官方强制执行方式时按上游文档设置，不能把非强制监控的结果冒充官方成绩。

这些命令是**单任务原生集成**，不构成官方 leaderboard 提交。正式 benchmark 批次、资源强制执行、种子和官方 usage gateway 请使用 [EFE 官方运行规范](https://github.com/mll-lab-nu/embodied-first-exam/blob/main/docs/protocol/running.md)。

## 依赖与尚未实测的部分

- DexHand、MuJoCo Playground、HumanoidBench 等使用上游固定 digest 的公开镜像；首次下载和构建时间另计。
- **RoboEngineer 的 evaluator 需要在本地构建**。请按固定 checkout 中 `tasks/roboengineer/README.md` 的 “Getting the images” 执行。AnyCar/Go2/G1 使用 `images/control-transfer/fetch_assets.sh` 和 `build.sh`；降落使用 `images/quadrotor-landing/prepare_assets.sh` 后构建对应镜像。这里的 `efe-private` 是上游镜像名，不是本项目给出的私有凭据。
- **Neural Fly 的 force-fit 资产不随仓库分发**，需要按上游条款从原始飞行记录自行生成。目录中收录这四项不等于已经取得该资产，也不意味着可以直接运行。
- AM-Bench 的 Isaac Sim / Isaac Lab 渲染需要带 RT 核心的 RTX 级 GPU、容器运行时和对应资产；H200 不满足这项渲染要求。G1 及部分其他任务也有 GPU 要求。GPU 设备分配遵循原生任务 Compose 配置，旧入口的 `--gpu` 不覆盖该配置；MuJoCo CPU 任务不要求 RTX GPU。
- 部分任务上游标为 `provisional`，已保留该字段。Standard 任务中 oracle 完整 38 项、部分 5 项、无 oracle 21 项；无 oracle 的任务支持 `develop` / `qualify`，拒绝 `replay`。部分 oracle 的结果也不保证成功。

本仓库会检查 EFE commit、运行脚本及任务完整目录哈希；文件内容或可执行位变化会拒绝运行。运行时复制任务，并使用上游 content-derived mtime 脚本防止 Docker BuildKit 混用不同任务的构建缓存。不会改动固定源码中的观测、预算或评分规则。

## 结果与复核

`run.json` 保存任务、模式、角色、固定版本、原生命令和预算；`runner.log` 保存外层输出；`jobs/trial/` 保留 Harbor 结果、agent 日志、生成的程序、评分与原生视频。确切文件集合由任务的 artifacts 声明决定。

`eca summarize` 分别报告 `coding_agent`、`reference_replay`、`negative_control`，保留全部原生 reward 字段，不将不同任务的 reward 混成总平均。不把 oracle 当作 agent 解题成绩，也不把 `infra_error` 或未产生 verdict 的运行记为任务失败。Harbor 有时在 trial 异常后仍以进程码 0 结束，本入口会继续检查 trial 是否产生有效评分。

新增任务的本次实际验证见 [EFE_VALIDATION.json](EFE_VALIDATION.json)。**目录接入与配置检查不能证明 coding agent 已能解题。**

本次实跑均为 Standard 原生链路自检，无模型调用。下面三个视频是**上游参考程序**的执行结果；它们不是本次 Claude Code 从零开发的结果。

| 核对 | 原生 success | 视频 |
|---|---:|---|
| Sharpa 手内重定向，oracle | 1 | [播放 MP4](media/efe/hand-reorient-oracle.mp4) |
| 同一灵巧手任务，nop 无动作 | 0 | 单帧负例，完整结果见验证 JSON |
| Go1 从倒地状态起身，oracle | 1 | [播放 MP4](media/efe/go1-getup-oracle.mp4) |
| HumanoidBench 坐下，oracle | 1 | [播放 MP4](media/efe/humanoid-sit-oracle.mp4) |

~~~bash
python -m unittest discover -s tests -v
# 已安装 EFE 独立环境后：校验全部 122 个任务模式及三种原生 agent 的 CLI 配置
.venvs/efe/bin/python -B scripts/validate_efe.py
~~~

任务清单的可重复导入入口是 `scripts/import_efe.py`；来源与版权见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。
