# 多本体 Coding Agent 首批探索

本批实际启动 Claude Code 自主观察、写程序、调用机器人并按反馈修改代码。目标是收集跨本体的成功/失败案例及过程证据；每项只做一次独立尝试，不用于估计稳定成功率。

统一设置：Claude Code 2.1.304，Opus 5.5（`claude-opus-5-5`），effort high；固定 EFE revision `13bdc8481166ae873eb5e1eebcc51b1a81ce4e2d`；Standard 模式；空任务工作区；不继承跨任务代码、对话、参考解法；资源监控强制执行 `RCB_LIMITS_ENFORCE=1`。任务原生的 reset/部署次数、动作接口、agent 超时、评分均保留。

模型通过 Token360 的原生 Anthropic 接口直连。仅为该模型域名显式配置 Harbor 的运行时网络出口扩展；记录为 standalone 探索实验，不作官方 leaderboard 声明。

| 任务 ID | 内容 | Agent 原生预算 | 轨道 |
|---|---|---:|---|
| `efe.dexhand.essential_pen_regrasp` | 灵巧手调整笔的握持位置和朝向 | 60 分钟 | trajectory |
| `efe.dexhand.usb_plug_insert` | 灵巧手拿起、转正并插入 USB | 60 分钟 | trajectory |
| `efe.mujoco_playground.go1_getup` | Go1 倒地后自主起身 | 30 分钟 | trajectory |
| `efe.mujoco_playground.go1_handstand` | Go1 用前腿倒立并平衡 | 30 分钟 | trajectory |
| `efe.humanoidbench.walk` | H1 行走 | 60 分钟 | trajectory |
| `efe.embodied_olympics_fun.backflip` | G1 原地后空翻并站稳 | 120 分钟 | trajectory |
| `efe.myosuite.baoding` | 肌肉驱动手盘保定球 | 60 分钟 | trajectory |
| `efe.myosuite.osl_run` | 带下肢假肢的肌肉驱动身体跑步 | 60 分钟 | trajectory |
| `efe.roboengineer.landing_random_pad` | 无人机适应未知动力学并安全降落 | 60 分钟 | policy |
| `efe.ambench.peg_in_hole_fahexa` | 飞行机械臂悬停插孔 | 60 分钟 | trajectory |

前八项并行启动。后两项先构建原生仿真环境并跑 nop 自检，产生有效原生评分后才自动启动 coding agent。Agent 预算不包含镜像构建时间；构建、自检、模型调用、正式评分分别记录。无人机任务提交固定 ABI 下的数据策略，不能与任意控制程序的 trajectory 任务混合解释。

每项保留：模型原生会话与工具日志、agent 写的代码、机器人交互记录、原生 verifier rewards、可用视频、异常和时间。模型/服务/环境故障与机器人未完成任务分开报告；参考程序的已有成绩不计入本批结果。

这份文件描述启动计划，不是实时结果页。本批全部结束前，不报告整体成功率。

批量启动示例（EFE 独立环境已经安装，Docker 与原生任务资产就绪）：

~~~bash
.venvs/efe/bin/python -B scripts/run_efe_pilot.py \
  --out outputs/efe-pilot-new \
  --credential-file /absolute/path/to/private/token360.env \
  --base-url https://api.token360.ai --allow-model-host --workers 4 \
  --tasks efe.dexhand.essential_pen_regrasp efe.mujoco_playground.go1_getup \
          efe.humanoidbench.walk efe.myosuite.baoding
~~~
