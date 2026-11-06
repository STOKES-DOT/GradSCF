# AO 网格优化：已完成的本轮验收

源码基线：e1537173；候选只修改 `integrals/grids/ao.py` 的低阶整数坐标幂。
既有 direct J/K 筛选补丁保持独立。完整原始结果现已同步到 `raw/`。

## 改动和依据

原编译图每个主网格块含 9 个大尺寸通用坐标幂运算。0–4 的整数幂现在使用精确乘法；保留原 custom JVP、非整数和高阶幂路径。基组、网格点/权重、归一化、dtype 与筛选阈值均未改变。没有停止任何物理参数的梯度。

没有发现每次循环复制完整 31 GB 缓存的编译证据。本补丁处理的是 AO 生产成本，而不是把未经证实的全量复制作为根因。最终 AO 存储量仍约 31.37 GB。

## 固定输入内核对照

96 原子 beta-carotene，Cartesian def2-SVP，880 AO，grid level 3，1,113,968 点，PBE，float64。前后在同一作业 116546 中串行运行，每项一次首次执行及三次 warm。下表是 warm 中位数，单位秒；额外工作区为编译器估算，不是 RSS。

| 设备/阶段 | 原版 | 候选 | 编译工作区变化 |
|---|---:|---:|---|
| CPU AO 缓存准备 | 103.415 | 33.983 | 536.86 → 27.04 MB |
| CPU 缓存后 XC | 10.529 | 11.419 | 288.49 → 288.49 MB |
| GPU 流式 XC | 3.151 | 2.280 | 547.96 → 490.32 MB |

CPU 缓存准备减少约 67.1%；GPU 流式 XC 减少约 27.6%。CPU 缓存消费核本轮反而更慢，已如实保留，不归因于“纯噪声”也不声称这一阶段已优化。其消费者代码与编译缓冲结构未改变，仍需后续单独核对。

物化微基准在替换返回值时短暂持有新旧输出，进程峰值约 64 GB；不要将它当作生产 SCF 的常驻需求。完整 SCF 的实际 RSS 另列如下。

## 完整 SCF（原作业已完成，无需重跑）

作业 116579 于 2026-11-06 15:18:25 +08:00 正常结束，ExitCode=0:0。
CPU/GPU 各一冷三热，8 次全部通过。使用独立生产构建，diagnostics=OFF；实际导入及库路径、哈希已核对。

| 项目 | CPU | GPU |
|---|---:|---:|
| cold 完整调用 / s | 508.553 | 329.361 |
| warm 完整调用中位数 / s | 459.674 | 232.254 |
| warm SCF 中位数 / s | 455.109 | 227.598 |
| 迭代数 | 24 | 22 |
| 进程组 sampled RSS 峰值 / GB | 36.1138 | 4.05459 |

CPU warm 完整调用原始样本：[460.46692016907036, 459.67391793243587, 459.583183279261] s。

GPU warm 完整调用原始样本：[244.59273512475193, 232.25386779382825, 229.92481658793986] s。

调用计时包含分子/输入准备及 SCF 同步完成，不包含进程 import、设备初始化和末尾核验；包含全部四次调用及这些额外工作的进程墙钟为 CPU 1892.792 s、GPU 1216.298 s。

同一几何的新对象 warm 允许复用正常程序/网格缓存，不复用收敛轨道。能量与原算子的轨道残差逐次通过；容差没有放宽。

与前一轮 58b01bb3 的 645.25 s / 266.68 s warm 完整调用比较，只能作为整体进展参考：本轮包含 direct J/K 筛选和 AO 两项改动，CPU 迭代数由 26 变成 24，实际 affinity 也不同；旧 CPU 对照只有一个 warm 样本。因此不能把总耗时差额全部归因于 AO 补丁，也不能称为严格单因素端到端加速比。上述同作业内核对照才是该 AO 补丁的直接性能证据。

## 科学与 AD 验收

RED：2 个编译运算门槛失败、7 个数值检查通过。GREEN：CPU 47、GPU 12、六方法 6、独立缓存 1，合计 66 项通过，无跳过。

覆盖整数幂值、零点处一至三阶导数、PySCF AO/三阶空间导数、跨块/尾块 s/p/d/f 值与 Laplacian、中心更新、坐标/指数/系数/模型参数的窄 JVP/VJP/HVP、传统及 global neural XC 和六种 HF/KS 的小体系 direct/dense 一致性。

大网格前后：CPU XC 能量差 2.84e-14 Ha、Vxc 最大差 6.66e-16 Ha、257 个真实网格点样本的 AO/导数最大差 1.33e-15；GPU 能量差 0、Vxc 最大差 4.44e-16 Ha。所有声明门槛通过。

这不是所有命名泛函、真实 NNAO 训练、完整分子 Hessian 或全部 conventional/RI 组合的最终验收；这些条目仍保留在固定 plan.md 中。

## 证据与复现

- `verification.json`：逐项核对状态、源码/动态库哈希、完整 SCF 原始样本。
- `before_after.csv`：同作业内核对照。
- `raw/`：原始 JSON、XML、HLO、日志；不纳入提交。
- `observed_summary.json` 和 `maintenance_checkpoint.md`：维护期间的历史检查点，不代表当前验收状态。
- 远端目录：`jy_gpu_2:/home/user_yuanjiao/private/homefile/gradscf-hfks-grid-20261106/`。
- 可重跑薄脚本：`benchmarks/probes/scf_100/grid/`；只在批准的 Slurm 分配内运行，选择新的输出目录。

核心回归：`PYTHONPATH=src:tests JAX_ENABLE_X64=1 python -m pytest -q tests/test_ao_integer_power.py tests/test_grid_ao_jax.py tests/test_scf_grid_execution_budget.py tests/test_streamed_xc_grid.py tests/test_streamed_external_xc.py tests/xc/test_custom_xc.py tests/test_ao_cache_reference.py`。

完整 SCF 使用先前已核验的 `gpu_warm.py`，一冷三热，单进程超时 5400 s、200 GiB RSS 保护。核坐标、基组和模型参数依赖没有冻结。原始完整调用均来自新的 production/src 及新的无插桩 .so。
