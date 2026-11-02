# MACE 默认容量：24 体系同条件对照

本实验是 2026-10-14 side session 中新授权的任务，不接续或改动主线程的 DZ 测试。
用户确认先各跑 200 epoch。复用原 24 体系结构（最大 12 原子），从相同 seed 0 的初始网络重新训练。

| 参数 | small | default |
|---|---:|---:|
| channels | 8 | 32 |
| max_ell | 1 | 2 |
| num_interactions | 2 | 2 |
| correlation | 2 | 2 |
| cutoff / Å | 5 | 5 |
| epoch | 200 | 200 |

两组使用 H/C/N/O/F 元素表、`szp663_direct`/core6、相同的固定 primitive 指数和原始收缩初始化；不采用模型类默认的 `szp442_direct`，以免同时改变基组。
指数来自当前 200 体系训练 epoch 629 保存的候选池；仅复用指数，不继承权重或 Adam 状态。
目标为 24 个结构的等权平均全电子 DF-RHF 总能量，Adam 学习率 1e-4，全局梯度范数裁剪 1，float64，现有 implicit SCF。

## 隔离与执行

只新增本实验目录。复用远程 `GradSCF_resume443_stability_20261013` 已验证源码，入口校验 scientific source SHA-256 `d7a4e0ace93ad1faccae713c7ff5cb4b67cbbbd39035ba8e9ee2a42baf286090`。
网络配置只在每次新 Python 进程内覆盖，不修改任何公共源码或主线程文件，不影响运行中的 200 体系训练。

一张 V100、32 CPU、128 GiB 主存、4 小时时限；先 default、后 small，分进程串行执行，共享只依赖几何与固定指数的积分缓存，使用 mmap，不保存完整四指标 ERI。
默认网络参数必须与实际 `MACEBasisModel` 构造函数签名一致。首先验证两组对全部 24 个结构输出相同的参考初始收缩，再分别执行完整冷/热梯度重放和逐结构参数方向有限差分。通过后才进入训练。
任一模型失败仍尝试另一组，保留失败记录；不静默跳过失败分子，也不把未完成曲线当作 200 epoch 结果。

## 对比产物

- `preflight.json`：实际参数量、配置、初始收缩一致性。
- `default/`、`small/`：训练历史、初始与最佳/最后检查点、梯度检查及各分子 SCF 诊断。
- `comparison.json`：匹配 epoch 的平均能量差、成功率、耗时与实际完成状态。
- `loss-curves.csv`、`per-molecule.csv`：逐 epoch 曲线与匹配 epoch 的逐分子对照。

这是单 seed、训练集上的默认架构组合对照，同时改变宽度和角向阶数，不能单独归因于宽度，不能作为泛化成绩。
历史 24 体系训练使用不同指数/流程，不直接作为此次受控对照的 baseline。
