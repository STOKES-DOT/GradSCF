# NNAO Stage 2：显存与 SCF 修复记录

记录日期：2026-10-08。本记录区分本地编译分析、远程四体系复验、已完成的 200/50 完整验证和仍等待资源的正式训练。

## JAX 显存设置

原 stage-2 作业均已设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false`，没有启用启动时预占用显存。但按需分配仍受默认 BFC 分配器上限约束。

作业 93522 在 n003、Tesla V100-SXM2-32GB、JAX/JAXlib 0.10.0 上运行三个独立进程，仅分配 2×2 float64 数组：

| 配置（均关闭预分配） | 分配器 bytes_limit | GiB |
| --- | ---: | ---: |
| 默认 memory fraction | 25,554,419,712 | 23.80 |
| `XLA_CLIENT_MEM_FRACTION=.90` | 30,665,302,842 | 28.56 |
| `XLA_PYTHON_CLIENT_ALLOCATOR=platform` | 未提供 memory_stats | — |

默认及 .90 进程的 `peak_pool_bytes` 均为 2 MiB，证明启动探针未预占用上述上限。这些探针不是科学计算的峰值显存测量。后续科学作业保留默认分配器、关闭预分配，显式设置 90% 使用上限。JAX 0.10 使用 `XLA_CLIENT_MEM_FRACTION`；不同时设置旧别名 `XLA_PYTHON_CLIENT_MEM_FRACTION`。

## DF 反向求导内存修复

`project_factors` 原先在反向 AD 中保存所有辅助基分块的 primitive 展开矩阵。MCONF/21 的单份 `(24,64,957,957)` 缓存就占 10.48 GiB。现在在 `fori_loop` 的 step 使用 `jax.checkpoint(step, prevent_cse=False)`，反向时逐块重算。

数学收缩、dtype、辅助基及分块大小保持不变。没有增加完整四指标 ERI，保留 JVP/VJP/HVP。CPU JAX 0.8.1 对真实形状的 abstract 编译分析：implicit 方法 arguments 均为 5.38 GiB，temporary 从 39.61 GiB 降至 11.62 GiB（减少 70.7%）。这是编译估计，不是 GPU 峰值。

## SCF 救援与 implicit 边界

S66/33 的初始 hcore 猜测出现 pyridine/ethene 片段间电荷振荡。新增可选 `--scf-rescue-level-shift .2`：原始 SCF 不收敛才用 0.2 Ha level shift 重试，然后撤去 shift，传入一致的密度、MO 系数及 MO 能量进行原始 SCF polish。

原始 `h+J−K/2` Aufbau 密度固定点及 implicit backward 保持不变，GMRES tolerance 仍为 1e-8。最终必须收敛，原始轨道残差和 `||Phi_raw(D)-D||` 均 ≤1e-7；未引入驻点拉格朗日量或零梯度回退。该检查不构成全局 HF 最小值或轨道 Hessian 稳定性证明。

## 四失败体系的 GPU 复验

93522：V100 32GB，16 CPU，128 GiB 主存；RHF/DF，`szp663_direct`，core6，`def2-universal-jkfit`，float64，seed0，mmap 原缓存。SCF max150，能量/密度/梯度阈值分别 1e-12/1e-10/1e-9。每个体系均检查 implicit 梯度有限、四阶系数 FD、MACE 参数 FD及5步 Adam（lr1e-4，clip1）。FD step1e-4，atol2e-6，rtol2e-5。

| 体系 | 系数 FD 绝对误差 | MACE FD 绝对误差 | 5 步能量下降 / mHa | 耗时 / s | 结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| MCONF/21 | 7.78e-9 | 4.53e-9 | 12.340075 | 150.32 | 通过 |
| Amino20x4/ARG_xci | 7.68e-10 | 4.70e-9 | 14.302988 | 142.88 | 通过 |
| MCONF/44 | 1.83e-9 | 1.04e-9 | 12.367247 | 135.02 | 通过 |
| S66/33 | 1.28e-9 | 3.66e-9 | 4.630052 | 78.52 | 通过 |

三个 OOM 案例的 rescue 参数为0；S66/33 为.2，最终 unshifted polish2次，raw fixed-point residual4.37e-10，轨道残差2.78e-10。完整结果保存在 `probe-93522/{large-results,scf-results}/`。四案例总数值复验耗时506.75s，不包括作业启动和分配器探针。

本地相关控制/数值回归60项通过；DF修复亦覆盖零收缩系数、双输入JVP及HVP。CPU S66新API的方向FD误差7.17e-9。原250个固定primitive DF缓存无需重建；原失败记录保留。

## 恢复的作业链

- 93537：V1001、32CPU、128GiB、48h。先用最终训练 rescue=.2 配置复验 MCONF/21，再重跑训练200和heldout50全部验证。任意组失败都不允许训练启动。
- 93538：A100_80G1、32CPU、128GiB、48h，`afterok:93537`。从头训练1000epoch，Adam1e-4→1e-5，full-batch均值，每20epoch只做heldout能量评估。
- 原93256显示 `CANCELLED / DependencyNeverSatisfied`，运行时间0，未产生正式训练结果。

本次四体系复验之后，已完成250体系完整验证；正式1000epoch训练尚未开始。新作业结果另存，训练工具会验证源码、结构、缓存和 rescue 配置的一致性。

93537 的最终配置最大体系前置复验已通过：MCONF/21 的系数FD、MACE参数FD及5步Adam全部成功，rescue=.2已启用但正常收敛路径未触发救援。证据另存 `gate-93537/largest/`。

## 完整验证完成（2026-10-08 15:58:48，上海时间）

93537成功结束，`stage=complete`、`exit_code=0`，训练集与独立验证集的命令退出码均为0。2026-10-08 16:06:16重新读取并逐条核验两份summary：

| 数据组 | 通过 | 失败 | 系数方向FD最大绝对误差 | MACE参数方向FD最大绝对误差 |
| --- | ---: | ---: | ---: | ---: |
| 训练结构 | 200/200 | 0 | 1.054e-8 | 1.071e-8 |
| 独立验证结构 | 50/50 | 0 | 7.551e-9 | 4.689e-9 |

全部250条记录的implicit梯度有限，系数及MACE参数FD均通过，且每条都接受了5步Adam。两份summary的scientific source hash一致，implicit tolerance=1e-8、rescue=.2。原先4个失败体系在完整训练集验证中也全部通过。

证据已保存到 `gate-93537/train/summary.json`、`gate-93537/heldout/summary.json`、`gate-93537/status.txt` 和 `gate-93537/validation-status.txt`。

正式训练93538于15:58:48解除验证依赖；16:06:16的调度快照为`PENDING / Resources`，0epoch。调度器当时预计2026-10-09 20:23:30开始，这一估计可能变化。完整准入验证通过不代表共享模型已训练，也不代表heldout泛化表现已测得。

Scientific source SHA-256：`7f066083658982a552fda47e0144e762796f69788c6a1f1c7e70c77fbfec403a`。本地/远程四个实现文件逐一核对hash一致。

## 正式训练进度（2026-10-09 16:21:35，上海时间）

93538于2026-10-09 11:50:19启动，当前在n001的A100 80GB上运行，分配32 CPU、128 GiB主存、48小时。已经接受25/1000次full-batch Adam更新；每次使用全部200个训练结构，implicit backward tolerance仍为1e-8。

| 指标 | 初始 | 最新 |
| --- | ---: | ---: |
| 200训练结构平均HF总能量 / Ha | -383.3973973210102 | -383.43075104275243（epoch25） |
| 50独立验证结构平均HF总能量 / Ha | -412.6238380905584 | -412.6537675595766（epoch20） |

训练集平均每体系能量降低33.35372 mHa；独立验证集平均降低29.92947 mHa。两组均相对各自epoch0计算，不能直接比较不同组成体系的平均总能量。验证仅评估能量，每20epoch进行一次，不用于Adam梯度更新。

最近一轮耗时304.77秒，稳定阶段约5分钟/epoch；按当前速度1000epoch约需83小时，另有初始化和验证开销，超过单次48小时分配。已提交94688（`afterany:93538`，同样A100 80GB、32 CPU、128 GiB、48小时），当前为`PENDING / Dependency`。只有父作业日志明确记录因时间上限终止才续训；数值失败或主动取消不会自动重试。续训复用原输出目录的参数、Adam状态及全局学习率计数，目标仍是总计1000epoch。

超时触发保护及现有训练恢复回归共30项通过，另通过作业脚本语法检查。启动前会归档父作业元数据和日志；相关文件为`resume_guard.py`、`test_resume_guard.py`及`run_resume.sbatch`，未修改生产模块。
