# 来源

统一启动器、任务清单、适配器和 Agent 生成的参考程序来自本研究。Claude Code 二进制、模型权重、场景资产、密钥与完整 rollout 不在仓库中。

RAX、CaP-X、LIBERO-PRO、robosuite、DexJoCo、DexMimicGen、ManiSkill、DLO-Lab、IsaacLab/Isaac Sim、dm_control 与 Gymnasium 是独立上游，安装使用遵守其各自条款；本仓库不重新许可这些依赖。URL/commit 见 provenance/upstreams.json。references 的模型 XML 来源于相应公开 benchmark。

provenance/harness_sources.json 记录提取与发布哈希；reference_runs 是操作者检查结果的程序，不用于新 Agent 初始化。

## Embodied First Exam

The existing runtime integrations retain their original upstream attributions in
`provenance/` and the source files. This notice covers the new EFE task metadata.

`tasks/efe_catalog.json` contains selected descriptions and configuration metadata
from [Embodied First Exam](https://github.com/mll-lab-nu/embodied-first-exam), commit
`13bdc8481166ae873eb5e1eebcc51b1a81ce4e2d`. Task implementations and solutions are
downloaded separately from that repository; benchmark assets and simulator images
may have their own licenses and usage terms. The EFE source license is:

MIT License

Copyright (c) 2026 Kangrui Wang and Embodied First Exam contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
