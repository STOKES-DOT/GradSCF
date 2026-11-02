# 九基组自由收缩能力诊断

2026-10-14；用户确认同一 S/M/L 档的 SZ/DZ/TZ 共用一套固定指数，缺失档先标定。

## 结构与控制条件

| 档 | C/N/O/F 活性 primitive | H 活性 primitive |
|---|---|---|
| S | 3s3p2d | 3s2p |
| M | 4s4p2d | 4s2p |
| L | 6s6p3d | 6s2p |

各档价层分别使用 1/2/3 个独立径向收缩通道，极化层一套，内层一套固定收缩，全电子 RHF。同一角动量的 m 分量共享径向系数。
本族 TZ 仅增加价层径向通道，不等同于 cc-pVTZ 的完整高角动量极化结构。

沿用 G2RC/2、BH76/37、S22/3 的完整 7 个结构，原坐标、电荷、自旋和化学计量不变；共 63 个基组—结构组合、252 次起点优化。
这是前序误差筛选后的诊断面板，不是独立、全面的泛化评测。

## 指数池与内层来源

统一保留 epoch 629 对照实际使用的 `core_primitives=6` 内层数值。新版 BSE 6-31G 与旧快照存在末位数值差异，因此不能混用后宣称完全一致。

- S：复用已完成的旧六参考结构 S-DZ 标定及自由收缩选择结果，候选名 `calibrated_candidate`。源目录记录于 provenance.json。
- L：使用 `nnao-relative200-20261014/snapshot.json` 保存的 epoch 629 固定指数候选，数值不变。
- M：使用与 S 完全相同的 H2、CH4、NH3、H2O、HF、CO 六个参考问题、相同内层；20 次指数更新预算、每个候选 3 起点×300 步自由 DZ 收缩；按既有检查和选择逻辑选池，再对全部 SZ/DZ/TZ 冻结。

S/M 的候选压缩评估使用 DZ，L 沿用历史 SZ 标定来源。跨档比较包含指数池与选择历史的影响；不是严格嵌套 primitive 层级，也不能把档间差异完全归因于 primitive 数量。
候选选择完成只表示该有限预算流程结束，不表示指数已经达到全局最优。

初步检查曾考虑复用新版 26 参考结构 S 标定；发现内层数值不同后，未启动计算即改为上述严格保持旧内层的六参考方案。

## 优化预算与可靠性

每结构四起点：可用最佳解或其嵌入、参考系数、两者各一个 seed 固定的扰动。
SZ 的 S/M 档没有已训练 MACE，使用确定性参考扰动作为 warm；L-SZ 以旧自由 SZ 最佳解起步。
同档 DZ/TZ warm 嵌入本轮低 zeta 最佳解，保留已找到的低维空间；L-DZ 额外保留旧 DZ 候选并核验能量重放。
各档每起点本轮最多 500 步，Adam 学习率 0.01，梯度阈值 1e-5，最多 10 次回退。历史 warm 候选的预算不同，不作严格同耗时优化器比较。

保留全部起点状态、梯度残差、能量跨度和失败；`max_steps` / `rejected` 不等于收敛。原始有效起点不被更差解覆盖。
S-TZ 的三列价层已张成全部三 primitive 空间；固定这部分可逆列表示，继续优化极化及 SCF 轨道。必须验证完整价层的梯度近零和随机可逆换基的能量不变性。

## 数值与独立核验

源 checkout 从已验证的 GPU 标定版本独立复制：科学源码 `ca9ad5856741b7ddfdc8e868fb7c33d300e9fd02e48279c60320821db593836a`；native SHA 见 provenance.json。既有训练不修改。

- 每档每个结构只准备一次 S/h/packed DF 缓存，三个 zeta 复用；不存四指标 ERI，不计算固定指数 VJP。
- native contracted 与固定池投影的 S/h/DF 因子逐项核对：atol 1e-8、rtol 1e-10。
- 系数 implicit SCF 梯度方向有限差分：步长 1e-4、3e-5，误差阈值 2e-6 + 2e-3×|方向导数|。
- 扩展初始空间不得高于低 zeta 已接受能量（1e-7 Eh 容差）；反序必须检查，不静默接受。
- 最佳固定基组由 GPU4PySCF 独立复核，能量差 ≤1e-7 Eh。
- 只用独立复核通过的完整组分组装相对能量。HF/cc-pVTZ 有限基组参考与 GMTKN55 高级别参考分开报告；非共价能不作 CP 校正。

## 调度与产物

一项 Slurm 作业：V100×1、32 CPU、128 GiB 主存、12 小时时限；顺序 S-SZ→S-DZ→S-TZ→L-SZ→L-DZ→L-TZ→M 标定→M-SZ→M-DZ→M-TZ。
每组独立进程，组间释放 JIT；不同分子不采用共享 JIT。原有两个运行任务不停止，新增最多一张 GPU。
运行入口 `sbatch artifacts/free-family9-20261014/run.slurm`。

campaign.json/phase.json 提供阶段状态；results/ 保存九组优化与独立核验；m-calibration/ 保存缺失池标定；最终汇总为 README.md、relative-energies.csv、per-structure.csv、optimization-status.csv。未完成或失败的反应不填补、不作为成功结果。
