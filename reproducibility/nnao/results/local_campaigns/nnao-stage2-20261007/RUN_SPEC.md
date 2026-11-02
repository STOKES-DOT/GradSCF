# NNAO 第二阶段：200 训练结构与 50 个独立验证结构

日期：2026-10-07（北京时间）。用户确认模型从头初始化，沿用 HF 变分能量目标与 implicit backward，不加载首轮权重。

## 数据与训练

GMTKN55 v2 固定快照，CHNOF、中性、spin=0、偶电子，2–40 个原子。200 train / 50 validation，按化学式分组，不跨训练与验证；验证排除首轮已见化学式。每式最多两个几何，坐标哈希去重。该分组不等同于完整反应连通分组，不能据此宣称完整 GMTKN55 反应能基准表现。

球谐全电子 `szp663_direct/core6`，固定几何、primitive 指数与内层；MACE 预测价层及多 primitive 极化壳的完整收缩系数。一个共享模型，CHNOF 词表、channels=8、2 interactions、max_ell=1、correlation=2、seed=0、fresh reference-bias initialization。

目标：200 个收敛 RHF／DF 总能量的等权平均。1000 epoch，每 epoch 完整遍历200个训练结构并更新一次 Adam，global gradient clip=1；cosine 学习率从1e-4降至1e-5。验证50个结构不参与梯度，每20 epoch及结束时记录其收敛能量，单独保存 `best_validation.npz`。

前向保持能量容差1e-12 Ha、密度1e-10、轨道梯度1e-9、max_cycle=150；共享implicit模块的线性求解容差1e-8、max_iter=100、restart=40，无附加正则。

## 内存与准入

DF辅助基为 `def2-universal-jkfit`。裸缓存估计总量304.810 GiB，单结构最大5.961 GiB，包含辅助指标全秩上限，未包含构建、SCF或backward临时内存；实际rank可能由metric裁剪降低。

固定积分写入既有未压缩NPZ。numeric NPY成员以只读memmap映射原文件，文件页由OS回收，每结构作为JAX输入传给GPU；不将305 GiB数组整体载入RAM，也不另存一份。单结构GPU峰值仍需实测。没有保存完整四指标ERI。

全部250个结构分别完成两项方向FD、finite coefficient/parameter implicit梯度及5步Adam准入；FD扰动只求重新收敛的能量，保留同样的SCF质量检查。初始AD与Adam步仍执行真正backward。任何失败都记录并阻止正式训练，不删除结构重新平均。spin=0及偶电子不证明轨道稳定；本轮未进行稳定性Hessian认证。

## 作业与依赖

| 作业 | 任务 | 资源 / 时限 | 依赖 |
|---|---|---|---|
| 93225 | 最大两个结构的V100检查 | 1 V100、32 CPU、128 GiB / 2 h | 独立检查，当前等待GPU |
| 93243 | 250个固定DF缓存准备 | 32 CPU、128 GiB / 24 h | 已在n001启动 |
| 93255 | 200 train及50 heldout完整数值准入 | 1 A100_80G、32 CPU、128 GiB / 48 h | afterok:93243 |
| 93256 | 200结构、1000 epoch正式共享训练 | 1 A100_80G、32 CPU、128 GiB / 48 h | afterok:93255 |

CPU准备、验证、训练按依赖串行；独立probe若同时运行，总CPU申请不超过64。GPU阶段尚未开始，不能把排队作业或缓存准备完成当作SCF/训练数值成功。

代码/日志：`~/private/homefile/GradSCF/artifacts/nnao-stage2-20261007/`。正式缓存：`~/private/datafile/GradSCF_GMTKN55/nnao-stage2-20261007/cache/`。缓存总量约305 GiB，保留远程；TF结果归档不下载大型缓存。

当前本地联合验证：56 passed in23.76s，脚本编译与 `git diff --check` 通过。源选择清单、CLI、Slurm脚本和实际结果中均记录数据/代码/缓存哈希。

早期检查作业93221在调用不存在的 `/usr/bin/time` 时启动失败（exit127），没有积分或GPU结果；93225已改用shell内置time，旧日志仅作诊断归档。
