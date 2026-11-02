# NNAO-SZP-L：500 体系训练 loss 数据与曲线

这是用户所指的 `NAOO-SVP-L` 对应的原始 **NNAO-SZP-L / L-SZP** 训练记录，
源任务为 `99005`；不是目前 H200 上的 L-TZP 训练。
2026-10-27 已重新与 `jy_gpu_2` 原文件核对 SHA256，确认完整终态历史与已有 Case 11 一致。
旧 Case 11 与其校验文件保持不变，此处提供可直接找到并离线复现的独立训练数据入口，
并补齐绘图所需的原始 stdout。

## 直接使用

- [完整 loss 曲线及后期放大图 PNG](training-loss.png) · [PDF](training-loss.pdf)。
- [能量变化的对数趋势 PNG](energy-log-trends.png) · [PDF](energy-log-trends.pdf)。
- [全部 597 条 epoch 记录 CSV](loss-curve.csv)。
- [原始 history.json](snapshot/train/history.json)、[原始训练日志](snapshot/restart-single-pass-99005.log)。
- [汇总统计](summary.json)、[来源与校验](provenance.json)、[验证记录](verification.json)。

500 个训练结构沿用[既有结构数据](../nnao500-20261021/README.md)，不重复复制。
推理模型沿用[共享 update-3351 权重记录](../../computational_examples/2026-10-21/models/update-3351/parameter-export.json)。

## 训练条件

全电子 DF-RHF / `def2-universal-jkfit`，float64，固定 primitive 指数、固定内层收缩，
由共享 MACE 输出环境相关的可变收缩系数；使用 implicit SCF 反传。
500 个体系，Adam batch size 50，逐分子 micro-batch 1，每 epoch 10 次更新。
初始学习率 1e−3，每 100 epoch 减半；最后保存点为 3.125e−5。
原作业：n001 / 一张 A100 80G / 32 CPU / 512 GiB 申请内存，72 小时时限。
详细模型超参数见 `snapshot/inputs/model-config.json`；原始代码来源见 `snapshot/source.json`。

原任务于 **2026-10-19 17:10:22（北京时间）因时限结束**，不属于收敛早停。
保存参数到 update 5969，对应 epoch 597 的 9/10 个 batch；最后完整 loss 是 epoch 596。
原始 `progress.json` 保留当时的 `running` 字段；终态以 stdout 的 `DUE TO TIME LIMIT` 为依据，
没有修改原始状态文件。

## loss 口径与缺失值

| 记录 | 含义与处理 |
|---|---|
| Epoch 0–3 | 历史整集评估值，单独显示 |
| Epoch 4 | 流程切换时无有效 loss，原样保留空值，不插值、不连线 |
| Epoch 5–596 | 592 条完整 `online_pre_update_mean`，每条覆盖全部 500 个体系 |
| Epoch 597 | 只完成 9 个 batch，不加入完整 epoch 曲线 |

在线均值汇总 10 个更新前参数状态的 batch 能量，**不是保存后固定模型的整集重评分**。
不将旧整集评估与在线均值拼成同口径的总改善量。

| 同口径在线指标 | 数值 |
|---|---:|
| Epoch 5 | −886.8742023766324 Eh/体系 |
| Epoch 596 | −887.4054334748351 Eh/体系 |
| Epoch 5→596 下降 | 531.231098 mEh/体系 |
| 最近 100 epoch 净下降 | 0.04484759 mEh/体系 |

曲线保留逐轮能量回升，不强制单调。对数图分别展示到最低已记录在线均值的差距和相邻轮变化的绝对值；
不对负的总能量取对数，不给零差距添加人为 epsilon，也不将经验差距称为 CBS 误差或真实优化极限。

## 与论文计算模型的关系

乙烷旋转/解离、乙烯密度等已归档比较使用 **update 3351**。
本训练曲线延伸到完整 epoch 596 / 保存 update 5969，不能把这些测试结果自动归属于最终模型。
旧 `best.npz` 和 `last_full_epoch.npz` 仍是早期整集评估检查点，也不能仅按文件名视作最后参数。
此目录不复制大权重、积分缓存或编译产物，不改变任何训练任务。

## 文件与离线复现

`loss-curve.csv` 保留 epoch、update、loss_kind、样本覆盖、平均能量、逐轮变化、累计在线下降及滑动统计。
能量列单位为 Eh/体系，变化列单位为 mEh/体系；缺失值为空。
`snapshot/` 保存未经修改的原始 history、progress、batch/phase、模型配置、检查点位置、代码来源、流程切换记录和 stdout。
`FILE_MANIFEST.json` 校验此入口文件；`provenance.json` 同时记录当前远程文件的 SHA256。

在本目录运行：

```bash
MPLCONFIGDIR=/tmp/nnao-loss-matplotlib python plot_loss.py
```

只需要 NumPy、Matplotlib；脚本逐条核对 592 个在线均值与原始 stdout 后重建 CSV 和四个 PNG/PDF。
不启动 SCF、网络推理或训练。由于汇总中的生成时间会刷新，重新出图后需同时刷新文件校验清单。
本次只整理训练数据、图和复现说明，不修改论文正文。
