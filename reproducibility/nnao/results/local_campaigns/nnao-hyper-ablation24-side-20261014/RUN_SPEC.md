# 24 体系 MACE：三项单变量对照

用户确认在 side conversation 中进行这三项实验，各训练 200 epoch，从 seed 0 重新初始化：

| 实验 | 相对 64 通道、2 层、max_ell=2、correlation=2 基线的唯一改动 |
|---|---|
| lr3e4 | Adam 学习率 1e-4 → 3e-4 |
| correlation3 | correlation 2 → 3，学习率仍为 1e-4 |
| radial64 | radial_MLP (32,32) → (64,64)，学习率仍为 1e-4 |

保持同一批 24 个结构、固定的 epoch 629 指数池、szp663_direct/core6、全电子 DF-RHF、float64、implicit SCF、相同初始参考收缩、全批次梯度平均和裁剪阈值 1。无参数迁移，不更新指数，不添加能量标签损失。

## 源码隔离

独立 checkout：`/home/user_yuanjiao/private/homefile/GradSCF_mace_ablation24_side_20261014`。
原始 checkout、旧实验以及主线程任务保持不变。副本仅对 `model/nnao/encoder.py` 作两处调整：构造函数新增 radial_MLP 参数，默认仍为 (32,32)；将这个参数传给上游 MACE。
源码 SHA 与精确文件差异见 `source-change.json`；新指纹 `a8d7d28a01831d22c15ed1922247f0ef1e812456546c99f5ab08cc12c79de634`。

每组在 Slurm 内验证：

1. 保持 (32,32) 默认径向网络时，seed 0 的 222,755 个参数逐字节复现原 64 通道基线初始化。
2. 实际 backbone.radial_MLP 与所选配置一致，结构改动后参数量增加。
3. 24 个结构的初始输出均为同一参考收缩。
4. 原训练入口执行冷/热能量及梯度重放、24 个参数方向有限差分检查，通过后再开始训练。
5. 最终汇总再次核对逐结构初始能量与原基线一致，按匹配 epoch 比较。

## 调度与结果

三组在同一 Slurm 作业中分进程串行执行；申请一张 V100、32 CPU、128 GiB 主存、4 小时时限。固定积分缓存只准备一次，通过 mmap 复用；不保存四指标 ERI。
单组预检查或训练失败会保存日志，并继续其他独立实验；不跳过失败结构，不将部分训练当作 200 epoch 完成。

各组结果在 `<variant>/training/`；`comparison.json`、`loss-curves.csv`、`per-molecule.csv` 汇总能量、覆盖率和耗时。
这仍是单 seed、训练集、有限训练预算的对照；不能由此直接判断泛化能力或完全优化后的容量上限。
