# NNAO stage-two 第 41 轮恢复

2026-10-09，jy_gpu_2。用户已授权继续原 1000 轮训练。

## 作业链

| 作业 | 工作 | 资源与上限 | 启动条件 |
| --- | --- | --- | --- |
| 94855 | 修复后完整 train200 / heldout50 准入复验，包含两方向FD、有限implicit梯度及各5步Adam | V100 ×1、32 CPU、128 GiB、8 h | 已在 n002 运行 |
| 94863 | 第41轮兼容性重评、检查点迁移、继续至总计1000轮 | A100 80GB ×1、32 CPU、128 GiB、48 h | afterok:94855 |
| 94864 | 超时后的检查点续训 | 同上，48 h | afterany:94863，且确认本父作业TIME LIMIT、迁移完整、没有数值失败、尚未完成1000轮 |

依赖串行，整条链同时最多32个分配CPU核心。所有脚本使用现有 jax_scf Python 的绝对路径、早期EXIT记录和正式仓库路径，预分配关闭、显存比例.90，float64。原约305 GiB primitive packed DF缓存直接复用，不构造完整ERI。

## 保持的训练定义

RHF/native DF，szp663_direct/core6，def2-universal-jkfit；200体系同一参数点的等权平均收敛总能量。Adam global-norm clip1，cosine学习率1e-4→1e-5，总步数1000，seed0，每20轮评估50个独立验证体系。SCF max150，能量/密度/梯度阈值1e-12/1e-10/1e-9，rescue .2 Ha。implicit最末真实残差认证1e-8，内部GMRES目标1e-9，max100 restart cycles、restart40。

## 显式恢复与来源

新输出目录为 `run-epoch41/`，原 `../nnao-stage2-20261008-repair/train-93538/` 只读。固定 parent/new 清单和正式修复回归94841证明仅三个修复文件变化，其余模型、原生积分、数据与缓存保持原训练版本。

`recover_train.py` 先由正式 trainer 验证新的250体系 gates、所有cache与完整配置，再核对旧 θ41、Adam序列化字节和两个计数均为41，环境版本与旧记录一致。fresh模型只用于重建结构/反序列化模板，实际评价使用旧训练参数。

在 θ41 上完整重算200能量/梯度及50 heldout能量。逐体系能量差≤1e-8 Ha、均值差≤1e-9 Ha、梯度allclose atol2e-6/rtol2e-5后，刷新g41；不执行optimizer update、不增加epoch、不重置moments/count。旧history0–41保留并标记原源码来源，新实现从42起训练。当前val41优于旧best-val40时保存参数41的验证最佳，否则保留来源明确的旧40。

三个检查点在staging完成校验后发布，最后写 `migration.json.status=complete`。部分发布或未完成迁移不能当普通resume。普通续训仍核对冻结764文件来源与运行环境。旧数值失败summary不会复制进新run；新的数值失败保存现场并停止，不自动跳过体系或使用无效梯度。

## 已验证与尚未完成

迁移、来源和训练控制本地48项通过；脚本语法通过；真实旧checkpoint解码后重编码字节相同、两个count=[41,41]。源代码修复已通过正式GPU100次反向与两方向四阶FD回归94841。

本次完整准入复验和实际检查点迁移仍由上述远程作业执行。提交作业不代表它们已经完成；查看gate summary、migration.json、run-epoch41/progress.json及每个job的独立status记录来判断阶段。
