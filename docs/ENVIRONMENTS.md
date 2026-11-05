# 运行环境

先用 python -m eca setup --group GROUP 打印命令，只有加 --execute 才安装。每组独立环境，固定上游见 provenance/upstreams.json。克隆之后仍需安装各上游的 native runtime、资产与软件包。

## 通用

Claude Code 2.1.304，ECA_CLAUDE_BIN 指向原生 Linux 二进制。外部 API 用原生 Anthropic Messages 兼容协议；密钥只在宿主转发器，Agent 只见本地占位值。每次输出目录独立，拒绝复用。

数值/视觉 Agent 使用本仓库 Dockerfile.agent 构建的离线 CPU 镜像。RAX 的 Messages/container bridge 来自固定 RAX clone；默认 external/rax-robodojo，可设 ECA_RAX_ROOT。旧项目目录、其他任务和 reference_runs 不会挂载。

## state

Python 3.11 + requirements/state.txt。系统需 bubblewrap、ffmpeg、libosmesa6，默认 OSMesa。任务看到原生状态向量和公共静态 model.xml，不能称为视觉完成。原留出 seeds 为 951001..951003。replay 走原先 policy(observation, step, memory) 的逐 tick 冻结执行器。

## maniskill / dexmimicgen

独立 Python 3.11 + requirements/precision.txt。DexMimicGen 另装固定 external/dexmimicgen 的 editable package；按上游下载资产。固定 RAX clone 提供桥接文件，Docker 镜像按 README 构建。

ManiSkill3.0.1，PegInsertionSide-v1，RGB-D、pd_ee_delta_pose、100步。DexMimicGen TwoArmThreading，两个 Panda、固定与双腕 RGB-D、1000步。无物体GT，留出973001..973003。--gpu / ECA_RENDER_GPU 选择 EGL 设备。

## factory

原生视觉渲染需要 RTX 节点，原8×H200节点不能替代。独立安装 requirements/factory.txt，并 editable 安装固定 IsaacLab commit afca7b09... 的对应 packages；按上游接受许可证和下载 Factory USD。源码预期 external/rax-robodojo/RoboDojo/third_party/IsaacLab。

模拟器固定逻辑 cuda:0，CUDA_VISIBLE_DEVICES 选择物理卡。保留原生预抓、竖直手、闭合夹爪、工作区 clamp。插销 / 齿轮参考来自空前缀 / fresh 条件，测试 seeds 为5101001..5101010、5102001..5102010。螺母为原始独立20开发/3留出版，973001..973003。不是迁移课程的螺母程序。

## dlo

独立 requirements/dlo.txt + 固定 external/DLO-Lab。ECA_DLO_ROOT 可改该路径。按其 experiments 环境配置安装 assets。原生预设两个抓点；同种绳结平移，不是未知拓扑泛化。最终参考控制器仅用末端本体位置，接口仍提供 RGB-D。连续分数不换算成功率。

## robodojo / dexjoco

分别用固定 external/rax-robodojo、external/rax-dexjoco，不混用。按各版 RAX scripts/setup_host_env.sh 和 simulator / Docker / native CLI 指南安装。ECA_RAX_PYTHON 可指定该组现有主机解释器。

RoboDojo 固定 ee67a146...、匹配 IsaacLab 和 patches/robodojo-rtx.patch；补丁为原运行的相机、碰撞与 RTX 兼容修改。下载原生物体/场景资产。原 Agent 看到 completion_sequence 视觉参考，不得到专家动作和物体GT。20开发/3正式，关闭 adaptation_source、handoff，单正式worker。DexJoCo 历史留出首动作前模拟器崩溃，仓库不把它写成已独立复现成功。

## capx_libero / capx_robosuite

固定 CaP-X、LIBERO-PRO / robosuite commit，分别装环境，避免不兼容的 robosuite。ECA_CAPX_ROOT 指向 clone。按该版 CaP-X 安装 extras、benchmark assets，以及 SAM3、Contact-GraspNet、PyRoki 服务：

~~~
ECA_SAM3_URL=http://127.0.0.1:8114
ECA_GRASPNET_URL=http://127.0.0.1:8115
ECA_IK_URL=http://127.0.0.1:8116
~~~

SAM3 权重需要自己的 Hugging Face 授权。新 Agent 通过 robot.command('execute_code', code=...) 使用固定 CaP-X APIs。沿用非特权观察/代码 AST 边界；这不是通用安全沙箱。controller.py 定义 main()，内部调用 robot.command；具体API在当前观测 capx_api_prompt。

历史演化接口与新入口不等价，不移用旧成绩。不具备原接口的历史代码块不提供伪等价重放。

## 验证口径

doctor 只是依赖检查；qualify 不付费启动 Agent。RAX / CaP-X 的 qualify 说明了实际检查范围，不能把仅初始化检查写成完整动作资格。结果读取原生字段；基础设施故障和执行器退出码必须同时核对。每次真实重测写到新的 outputs，不改历史 record.json。
