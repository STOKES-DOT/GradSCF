# 200 体系联合训练启动记录

日期：2026-10-11。用户明确要求 200 个训练体系、1000 epochs，不设置测试集或 validation；确认从 24 体系 epoch-1000 最优参数开始，保留收缩系数和指数、重置 Adam，并归档停止旧任务。

## 新任务

- Slurm **95573**，`jy_gpu_2` / **n001**，**NVIDIA A100 80GB PCIe ×1**。
- CPU 32 核，主机内存 256 GiB，时限 120 小时；JAX 0.10.0、float64，GPU 预分配关闭。
- 数据：原 stage2 `selection/train.jsonl` 的全部 **200** 个中性 CHNOF 结构；不读取 `validation.jsonl`，heldout_count=0。
- 数据 SHA256：`9bf275038c21f3775111c749cf43287d1bc494503928a6079cd5d12fa0098f8f`。
- 初始化：任务 95277 的 `best.npz`，epoch 1000；7171 个 MACE 向量条目及 14 个指数缩放均保留，Adam 状态重置。
- 初始化 SHA256：`61ab159e03b354b422d6251914b6aeff1929f016a919c228a960203ce7805700`。
- Adam：MACE 学习率 1e−5、指数学习率 1e−3、指数 log_bound=0.25；目标 1000 epochs。
- native contracted 积分与基组 VJP，稳定 DF-RHF 前向，implicit SCF 容差 1e−8。未通过稳定性或梯度检查的候选不得推进 accepted checkpoint。
- 已确认源码核对、native 构建及 GPU 初始化完成；当前正在准备 200 体系的初始计算，尚无已接受 epoch。以远程 progress.json 为实时依据。

## 源码与结果

完整源码共 1630 文件，来自当前 tracked 工作区和本次新增回归；未包含无关 untracked assets/examples。

- 基础 Git commit：`1242c9346ab0ddc9fc6bc741efa2b0d3fc32ccf1`，加本次尚未提交的修复。
- 源码摘要：`6bc6401cfc36ca0832f01a64af10ff6661e39af95dbfd00c832a5f37b27102b8`。
- 正式服务器源码：`/home/user_yuanjiao/private/homefile/GradSCF/releases/joint200-6bc6401cfc36ca08`。
- 本任务实际运行源码：`artifacts/nnao-joint200-20261011/job-95573/source`（服务器 GradSCF 根目录下）。
- 训练结果：`artifacts/nnao-joint200-20261011/training/`，包括 metadata/runtime、batch-progress、progress、best/last 检查点。
- 日志：`artifacts/nnao-joint200-20261011/train-95573.out`。
- 本地 `source-manifest.json`、`remote/deployment.json` 和 `remote/training/` 保存来源及启动快照。

## 旧任务与检查

旧任务 94863 和续训 94864 已按用户授权停止。旧 200 体系训练最后接受到 **epoch 184**，mean E=`−383.5844382953929 Ha`；best/last 相同，SHA256=`f0df3ff133c75571ff55842684e5c1f0635a7adc5e92e79f3c159ff38707843f`。检查点和历史保存在服务器本任务 `previous-stage2/`，原目录也保留。旧 validation 文件仅作为历史归档，不进入新训练。

修复回归：远程 22 项通过；N₂ native/MACE/implicit 联合梯度与有限差分误差 2.48e−9；原始 24 体系初始批次 24/24 稳定且梯度有限。200 体系的新初始批次尚在准备，不能将 24 体系检查等同于 200 体系已经通过。
