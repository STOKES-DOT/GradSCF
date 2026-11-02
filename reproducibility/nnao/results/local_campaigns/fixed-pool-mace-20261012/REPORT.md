# NNAO 固定指数 MACE 短训练验证

2026-10-12；Slurm 作业 96061，n005，NVIDIA A800 80GB PCIe，8 CPU / 32 GiB 申请资源。
作业 COMPLETED，ExitCode 0:0，Slurm 记录总耗时 7 分 3 秒（含构建、测试及两次启动）。

## 方法与输入

- H2、CH4、NH3、H2O、HF、CO，几何来自 `data/pool_pilot.json`；中性、单重态、等权重。
- 全电子 RHF、spherical szp663_direct/core6、def2-universal-jkfit DF；SCF float64。
- 固定 `candidate-exponents.json` 中通过前期 A/B 评估的候选指数，保留内层基组参数。
- 共享 MACE 输出完整原始收缩系数；参考收缩初始化输出头，Adam 学习率 1e-4，seed=0。
- 一个 epoch 遍历全部 6 个结构并更新一次平均能量梯度；不使用能量标签或 validation 集。
- 反向为标准 implicit SCF → 含归一化的 JAX 积分投影 → MACE。指数不是训练变量。
- 缓存混合池 S/h/packed DF；内层已收缩；没有完整四指标 ERI。

## 结果

先训练 5 epoch，再从 last.npz 恢复至 epoch 6。best.npz 与 epoch 6 的参数相同。

| 项目 | 数值 |
| --- | ---: |
| 初始化平均能量 / Eh | -64.166324284501 |
| epoch 6 平均能量 / Eh | -64.169527973812 |
| 平均能量下降 / mEh 每分子 | 3.203689 |

| 分子 | 6 epoch 能量下降 / mEh |
| --- | ---: |
| H2 | 0.582934 |
| CH4 | 1.227483 |
| NH3 | 2.614222 |
| H2O | 3.841230 |
| HF | 6.505533 |
| CO | 4.450733 |

## 验证与计时

- 远程相关回归 36 passed，78.22 s，包括真实 MACE 梯度与恢复测试。
- 真实 H2 测试的连续训练与中断恢复结果：参数、梯度、Adam 序列化状态逐项一致。
- GPU 6 分子参数方向导数均通过：步长 1e-4，绝对容差 2e-6，相对容差 2e-5。
  保存的 `train/gate.json` 为 epoch 5 续训前检查，最大 AD/FD 差约 1.715e-6 Eh。
- 热调用能量/梯度一致；检查点能量与梯度重新求值通过；所有最终 SCF 收敛且稳定。
- 本地新增指数变形测试通过：投影与 native contracted S/h/DF 和系数 VJP 一致。
  `projection-regression.log` 记录 5 passed；该新增参数化案例未在本次远程 36 项中计数。
- 禁止 native 计划调用后，缓存重载、投影及系数梯度测试仍通过；指数变化或校验和错误被拒绝。
- 806 个同步源码及数据文件的 SHA256 与本地清单一致；远程 native 库在计算节点重新构建。
- 原训练 epoch 2–5 完整 batch 更新耗时 0.17–0.19 s（含检查点写入）；首次更新 0.35 s。
- `train/timings.json` 为**续训启动这次调用**：首次全 batch（含重新编译）104.43 s，
  热 batch 能量与梯度 0.307 s，含准备/检查的这次调用总耗时 156.58 s。
  它覆盖了首次训练同名计时文件，不能据此报告原始缓存构建耗时。
- RSS 为 psutil 每 20 ms 采样的进程内存；GPU 显存峰值未测量。

## 可复现文件与限制

`run.slurm` 保存实际命令，`source-sha256.txt` 保存同步源文件清单；`train/` 内有
完整 identity、逐轮/逐分子能量、梯度检查、计时、best/last 检查点。
服务器目录：`/home/user_yuanjiao/private/homefile/GradSCF/artifacts/fixed-pool-mace-20261012/`。

这是端到端短训练验证，MACE 尚未充分优化；没有测量偶极矩、极化率、BSSE 或泛化。
当前数据为理想化小分子，不能外推 200 分子的速度或显存容量。阶段 D 尚未接入。
本次没有提交 GitHub，也没有启动 1000 epoch 的长期任务。

初始失败记录保留：96044 缺少节点 GCC 启用脚本；96047 发现批记录重复字段；
96060 在修复传输完成前启动，仍读到旧文件。96061 使用已核对的修复文件并完成。
