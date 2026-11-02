# 固定指数 NNAO 性能修复与两阶段边界

## 诊断

阶段 C 原来已经不计算 native 指数 VJP。实际发现的额外开销是：

1. 固定积分投影前向、SCF/implicit 反向、投影 VJP 分开调用；前后向分别接收 mmap 数组，
   系数梯度还经历 GPU→CPU→GPU 的中转。
2. 逐次 RHF 稳定性检查通过 Cayley 旋转能量的二阶 AD 构造 HVP。这项检查在旧的
   278.39 s/epoch 固定指数版本中尚未出现，不能简单移除来追平历史速度。
3. 方向有限差分另用仅能量入口，导致已完成能量/梯度预热后再次编译每个体系的 SCF。

14 分 37 秒是首次热批评估，不是稳定训练 epoch。原代码后来实际 epoch 3 为
362.50 s；同数据的更早版本最快为 278.39 s。不同参数、指数、求解策略和 GPU 的历史
数值不能直接作为受控加速比。

## 实现

- `training/pool.py`：固定积分投影、既有 implicit SCF、系数反向合并为一次 JIT；
  `evaluate_device()` 保持系数梯度在设备上，主存 API `evaluate()` 保留。
- `scf/stability.py`：精确 real RHF occupied/virtual Hessian-vector product。
  DF 因子只转换一次到 MO 分块；Davidson 的各次 HVP 复用这些三指标因子。
  不使用轨道能差对角近似替代真实 Hessian，不构造四指标 ERI。
- `workflows/fixed_pool.py`：有限差分复用已预热的能量/梯度执行体；
  写出 `phase.json`、逐分子 `gate-progress.json` 和各主要阶段的计时。

SCF 收敛阈值、稳定性判据、负曲率逃逸、implicit GMRES、Adam 和检查点接受逻辑不变。
阶段 C 仍只更新 MACE。指数微调需要的 native 路径及其指数相关归一化导数保留；
新动态指数必须使用对应的积分，不能复用旧指数的固定数值缓存。
本次没有重新引入废弃的两阶段控制器，也没有新增阶段 D 的完整编排。

## 同硬件实测

最终审计作业 96229：n001，A100 80GB，8 CPU / 64 GiB，JAX 0.10.0。
seed=0，原候选指数、同一 geometry/cache/model，完整 MACE forward + energy + backward。
每组一次冷调用、三次热调用，表中为热调用中位数。三个变体均逐次比较能量、7171 维
网络梯度和最低曲率；完整记录见 `audit-results-mo.json`、`audit-96229.log`。

| 体系 | 原实现 / s | 仅融合 / s | 融合＋MO 分块 HVP / s | 相对原实现耗时下降 |
| --- | ---: | ---: | ---: | ---: |
| S22/08a | 0.03509 | 0.03101 | 0.02907 | 17.2% |
| S66/18 | 0.40357 | 0.36105 | 0.23059 | 42.9% |
| MCONF/44 | 6.29646 | 5.81137 | 1.69058 | 73.2% |

全部比较的网络梯度最大绝对差为 8.75e-14，最低曲率最大绝对差为 4.89e-15；
这三个体系保存的总能量一致。接受容差分别为能量 1e-8 Eh、梯度 atol1e-6/rtol1e-5、
曲率 atol1e-7/rtol1e-6。小体系几十毫秒的时间易受调度影响，不能把单次百分比当作普遍保证。

第一轮 AO 解析 HVP 对照单独保存在 `audit-results-ao.json`；最终采用上述 MO 分块版本。

## 验证边界与全批检查

- 本地相关回归：66 passed、1 skipped；跳过真实 MACE 依赖测试。
- 独立远程真实栈：20 passed，包含真实 MACE 训练/恢复、native 指数梯度和稳定性测试。
- 解析 HVP 与原 Cayley 能量的完整 AD Hessian 比较，覆盖不同占据数、非正交 AO 和
  非驻点轨道；另有方向有限差分、不稳定驻点逃逸和 implicit 高阶响应回归。
- 最终六个源文件/测试哈希已核对；见 `final-source-sha256.txt`。
- 作业 **96230** 已在 n001 运行：完整 200 分子、256 GiB、32 CPU、A100 80GB，
  原候选池、fresh seed=0、同 mmap 缓存。依次做冷/热检查、方向导数、一次 Adam 更新。
  输出目录为远程本目录的 `full200/`，最多 6 小时。**整批 epoch 收益尚未测完。**
- 原任务 **96071** 的源码和检查点未覆盖。尚未将新实现切入该长任务，也未提交 GitHub。

复现入口：`audit.slurm` 与 `full200.slurm`，均使用本实验的隔离源码目录 `source/`。
