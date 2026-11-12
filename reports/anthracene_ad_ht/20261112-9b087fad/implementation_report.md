> **Current run — 2026-11-12 13:19:23+08:00:** job **122200** is running on
> n006 with one NV5090/64 logical CPU/256 GiB. The dynamic-pullback mechanism
> test passed and the unchanged native B3LYP geometry check is reused with
> exact source hashes. Actual anthracene AD started **13:19:20+08:00**;
> 0/32 Cartesian components at launch, no completed AD spectrum.

# Anthracene AD-HT — implementation and bounded acceptance report

日期：2026-11-12（Asia/Shanghai）。基线 `9b087fad`，分支
`examples/anthracene-ht-20261110`，工作树
`/Users/jiaoyuan/.codex/worktrees/bd0a/GradSCF`。本报告记录已归档的实现验收，
实际蒽 S1–S8 AD-HT 尚未完成。主控确认正式蒽 AD 阶段于
`2026-11-12 10:58:55+08:00` 在 job `122119` 启动；当前列出的 receipts
尚未包含全部 32 个真实蒽 pullback、66 模导数或最终物理谱的完成证明。

## Methods — 实现与物理输入

`src/gradscf/tddft/casida.py` 为 restricted full-TDDFT 的显式
`tda_gradient_mode="implicit_eigenvector"` 接入共享 `solve_rpa`。
TDHF 的 `JH` action 转成耦合 `H=[[A,B],[B,A]]`，一次物理遍历保留 X/Y 耦合及原
microbatch；`structured_rpa.py` 新增可选 `h_operator`，使 primal、稳定性检查及
向量响应复用该 action。向量签名 closure conversion 提升闭包中的动态 Fock、轨道、
积分与 bound-XC 参数；默认 `eigenvalue_only` 仍保留原行为。

对实数、正 metric、孤立根，内部归一化为 `zᵀJz=1`，满足 `Hz=ωJz`；
一阶响应使用 `dω=zᵀ(dH)z`，并在 `zᵀJ dz=0` 规范下求解增广线性方程。
公共振幅使用 `X,Y=z/√2`，故 `2(XᵀX−YᵀY)=I`。跃迁偶极保留 X+Y 与移动 AO
dipole integrals 的共同响应。该路径支持这里验收的 restricted／real／isolated-root
一阶导数；未建立 unrestricted、复数体系或高阶 TDDFT 向量 AD 验收。
外部 guard root 检查与完整物理残差保留；稳定性最低 Ritz 检查不等于严格全空间证明。
共享 solver 默认 `gap_tol=1e-8`、`stability_tol=1e-10`，与 TD facade 的
`excitation_threshold=1e-3 Ha` 属于不同检查，不能相互替代。

`examples/tdscf/anthracene_ht/05_ad_response.py` 在移动几何上保留 native integrals、
overlap、AO／grid weights、隐式 SCF 轨道响应、完整占据与虚轨道 Fock 块、X/Y 与
跃迁偶极的依赖。已收敛的旧 MO 系数仅提供数值原点，stationarity 机制提供 `dC/dR`。
一次 `jax.vjp` 线性化承载 8 个 gap 与 8×3 个有符号 dipole 的 32 个 scalar
pullback；每个 Cartesian 梯度单独保存并检查有限性。公开 `TDDFTResult.converged`
仅表征 primal，不能替代这些逐分量响应检查。

| 固定输入／约定 | 值或边界 |
|---|---|
| 分子与电子空间 | 蒽 C14H10，24 原子，中性闭壳层；既有基线为 260 AO、47 occupied／213 virtual，保留全部轨道 |
| 电子结构 | B3LYP、Cartesian def2-SVP、grid level 3；SCF energy/density/gradient 容差分别 `1e-11 Ha`／`1e-9`／`1e-8` |
| full-TDDFT | 8 个目标 singlets；driver tolerance `1e-8 Ha`，vector adjoint tolerance `1e-9`、最多 200 次；共享 solver 另取第 9 个 guard root |
| 物理 action | direct J/K；AD response `direct_tol=0`、`integral_cache="none"`；未新增 native/C++ 实现 |
| 几何／正常模来源 | 复用已接受的 ground geometry、全部 72 列 Hessian 和 66 个正频率模；不重算 ground Hessian |
| Cartesian 与 Q 单位 | 坐标 bohr；gap Cartesian gradient 为 Ha/bohr，dipole Cartesian derivative 为 e；投影使用 `Q=bohr·sqrt(electron mass)` |
| 谱模型 | 0 K，independent equal-curvature VG + 每态 linear HT，电子态强度非相干相加；Gaussian FWHM `100 cm⁻¹` |

8 个请求态与第 9 个 guard root 是输出与根隔离检查范围，伴随方程仍使用完整
占据×虚轨道响应空间，不将物理响应截断成九态。旧 FD vertical anchor 保留 12 个
candidate roots，用有符号 AO transition-density matching 确定目标根排列与常量相位。

## Results — 已测回归与能力边界

所有下列测试在同一远端 Slurm job `122119` 中执行，无本地科学测试。
节点 `n006` 的 CPU 为 AMD EPYC 9654 96-Core Processor；allocation 为 64 logical
CPU／256 GiB／1×NVIDIA GeForce RTX 5090（32607 MiB，driver `570.172.08`），
24 h 上限。硬件快照见 [allocation.txt](raw/allocation.txt)、
[lscpu.txt](raw/lscpu.txt)、[gpu.csv](raw/gpu.csv)。`allocation.txt` 是启动时快照，
不作为后续运行状态或释放证明。

测试默认 `JAX_PLATFORMS=cpu`、float64、禁用 JAX compilation cache；
`OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1`，`GRADSCF_NATIVE_THREADS=8`，
每个回归进程使用 `srun -n1 -c8 --cpu-bind=cores`。
GPU 两项单独设置 `JAX_PLATFORMS=cuda,cpu`。精确环境及顺序见
[run.sbatch](raw/run.sbatch) 与 [queue scripts](raw/queue)。

| 检查与证据 | RED | GREEN | 已证明的范围 |
|---|---|---|---|
| [RSS](raw/rss-red.xml)／[组合 GREEN](raw/array-green.xml) | 1 failed／4 passed，0.084 s | 5 项包含于 45 passed | glob 惰性枚举 ESRCH 被确定性复现；逐 PID ENOENT/ESRCH/EACCES、group 筛选和 RSS 字节单位保留 |
| [full-TDDFT vector](raw/vector-red.xml)／[组合 GREEN](raw/array-green.xml) | 5 failed／2 passed，18.159 s | 当时 7 项包含于 45 passed | 独立 X/Y JVP/VJP、动态 Fock/XC、nested JIT、dense oracle、metric 与无效响应 |
| [fused H](raw/fused-red.xml)／[组合 GREEN](raw/array-green.xml) | 5 failed，1.696 s | 5 项包含于 45 passed | 耦合 action、shape 验证及既有 initial guess／matrix epsilon 参数传递 |
| [CPU 几何 RED](raw/geometry-red.xml)／[GREEN](raw/geometry-green.xml) | 3 failed，362.482 s | 3 passed，431.983 s | 原生 moving-integral／grid／relaxed-SCF／full-TDDFT gap 与有符号 dipole 的实际几何导数 |
| [谱 provenance RED](raw/provenance-red.xml)／[GREEN](raw/provenance-green.xml) | 1 failed，4.068 s | 5 passed，7.569 s | fixture 上的共轴绝对面积、status/signature gate 与 AD 标签传递 |
| [custom-XC MLP](raw/neural-green.xml) | 额外窄验收，无独立 RED receipt | 1 passed，20.794 s | 非局部 AO density-matrix Functional 的 MLP 参数保留 X/Y 响应；不代表训练完成 |
| [独立 GPU 数组 AD](raw/gpu-array.xml) | 单独 backend 窄验收 | 2 passed，141.260 s | fused-H／闭包 A/B 数组上的 nested-JIT JVP/VJP；未覆盖 native 几何 |

[array-green.xml](raw/array-green.xml) 的 **45 passed／0 failed／0 skipped，282.139 s**
包括 RSS 5、fused 5、当时 vector 7 及已有 matrix-free/lazy/config 回归。
该组合与 CPU 几何、MLP、谱 fixture 共 54 个不同 CPU 测试身份；GPU 两项重跑其中
两个身份，不把它们计为两个新的独立 CPU 测试。以上耗时是 pytest suite wall time，
包含编译与检查，不能当成实际蒽耗时或实现加速比。

CPU 几何测试在 `sto-3g`、grid level 0、float64 上比较正负位移各自重新收敛的 SCF，
并以 AO transition-density overlap 独立对齐偶极符号。HF 使用异核 HeH+，额外检验
占据／虚轨道 SCF 混合；PBE/B3LYP 使用 H2。方向步长为 `2e-4 bohr`；
gap/dipole 检查 `atol=4e-6, rtol=4e-4`，scalar gradient 对 FD 使用
`atol=5e-6, rtol=4e-4`，JVP/VJP 对偶一致性使用 `atol=3e-8, rtol=2e-6`。

| CPU native 几何用例 | gap 方向误差（Ha/bohr） | dipole 方向最大误差（e） |
|---|---:|---:|
| HF／HeH+ | `5.02363573e-9` | `9.27461441e-10` |
| PBE／H2 | `9.84447607e-9` | `1.30046326e-9` |
| B3LYP／H2 | `8.52307147e-9` | `1.39540146e-9` |

数值来自 [geometry-green.xml properties](raw/geometry-green.xml)。原 energy-only
路径在这三个 RED 几何用例上保留了 gap 响应，却不能给出正确 dipole 响应，故该
测试能够识别只接回 eigenvalue 的实现。数组向量测试另用 `1e-4` 参数步长、
X/Y JVP `atol=2e-8, rtol=2e-6` 及伴随一致性 `atol=2e-10`；primal／adjoint
容差为 `1e-10`／`1e-11`。这些测试均只验收一阶响应。
几何测试的值与导数均由 GradSCF/native 路径产生；本报告没有 PySCF 外部对照，
也不把内部独立重收敛的 FD 核验称为外部程序验证。

**CPU native 几何 AD 与 GPU 数组 AD 分别有证据。** 正式蒽示例显式将 JAX default
device 设为 CPU，因为现有 native integral／J-K geometry primitives 在 CPU lower。
queue 10 的 GPU-capable 环境和单卡 allocation 不改变该几何链执行边界。GPU 两项
只验证数组 operator 的响应，不构成 GPU 原生积分、GPU SCF 几何 AD 或实际蒽 GPU
端到端验证。

RSS 修改仅将递归 `Path("/proc").glob("[0-9]*/stat")` 换成 `/proc` 根目录枚举与
数字 PID 筛选，每个 `PID/stat` 仍在原异常保护内读取；不吞掉顶层 `/proc` 枚举故障，
保持 process-group 过滤及 `rss_pages×SC_PAGE_SIZE`。该改动修复监控 wrapper
竞争，不改变科学求解器。

## 实际蒽运行与待验收项

旧 job `119234` 于 `2026-11-12 04:03:45+08:00` 因 RSS glob 的
`ProcessLookupError(errno=3)` 中止，保留 39/132 个 primary FD 点，即 19 个完整
正负模式对及第 20 模正向点；每点有 8 个目标态／12 个 candidates。
这些数据是未来 AD 的独立验证基线，不是 AD-HT 成品，也没有完整 FD 谱。
旧 ground／72-column Hessian／66 正频率模式的既有验收可以复用；旧进度快照中的
`ad_ht_implemented=false` 属于旧停止时点，不代表本报告的新实现状态。

正式 AD 运行目录为
`/home/user_yuanjiao/private/homefile/gradscf-ad-ht-20261112`，
输出隔离在 `reports/anthracene_ht/20261110-b3lyp-def2svp/ad_eight_states/raw`。
[queue 10](raw/queue/10-anthracene-ad.sh) 等待 CPU 组合、geometry、谱 provenance、
MLP 成功退出后启动 AD，使用 64 CPU／native threads 64；guard 为 `72000 s` 与
`236223201280 bytes`（220 GiB）RSS。代码按 source/native/input hashes 拒绝陈旧
checkpoint；同一 VJP 的 32 个分量可在身份一致时续算。此处未归档正式阶段的
成功 process receipt，不给出完成时间、峰值内存或速度估计。

验收仍需取得全部 32 个 `(24,3)` Cartesian pullback，分别检查有限性及平移不变性
（gap `1e-5 Ha/bohr`、dipole `1e-5 e` 数值阈值）；投影后应有 `(8,66)` gap
gradients 与 `(8,3,66)` dipole derivatives。旧 FD 完整模式对中选择四个方向，
含低序模式及较强 gap／dipole 方向；步长 `0.01 bohr·sqrt(amu)` 经质量单位转换后，
对当前 Q 导数分别使用 gap `atol=3e-8, rtol=3e-3`、dipole
`atol=3e-7, rtol=3e-3`。FD 数组只进入误差断言，不产生或覆盖谱模型导数。
只有这些门槛全部通过，`mode_response_status.json` 才写 `complete=true`。

[queue 11](raw/queue/11-ad-spectrum.sh) 仅在正式 AD 阶段成功后生成谱。
最终还需核对 8 态／66 模、统一绝对能量轴、每态及总光学面积 `1e-5` 门槛、
CSV/NPZ 与 PNG/PDF 的实际内容和渲染，取得 spectrum/status/process receipts；
[queue 12](raw/queue/12-release.sh) 的成功资源释放也需另行确认。
当前 fixture 的绘图通过不代表已生成或查看真实蒽谱。

## Provenance — 可审查证据

实现入口：`src/gradscf/tddft/casida.py`、
`src/gradscf/solvers/eigen/structured_rpa.py`、
`examples/tdscf/anthracene_ht/05_ad_response.py`、`input_ad_eight_states.json`、
`04_spectrum.py` 与 `benchmarks/probes/tddft_perf/run.py`。
无本轮 native/C++ 修改；未运行本地科学计算、未在本报告中宣称已提交或合并。

完整命令保存在 [raw/queue](raw/queue)，退出码 `.exit` 与各 XML 相互对应。
组合 GREEN 的准确测试文件为 `test_perf_rss_sampler.py`、
`solvers/test_rpa_fused_operator.py`、`test_full_tddft_vector_ad.py`、
`solvers/test_matrix_free_rpa.py`、`solvers/test_rpa_lazy_response.py`、
`solvers/test_rpa_response_config.py`，均相对于 `tests/`。
其余 geometry、provenance、MLP、GPU 窄测试的准确选择器分别在 queue 05、07、08、09。

该报告验收的是已列实现和窄测试范围；实际蒽 32 pullbacks、66 模结果、最终谱、
耗时／内存及 allocation 结束状态，保留为 [固定计划](PLAN.md) 的未完成项。

## Restart after the first reverse compilation memory failure

The first actual shared AD forward completed in 2422.243 s. The original
`jit(lambda weight: pullback(weight)[0])` then captured112.57 GB in lowering;
its process-group sampled RSS reached236708753408 bytes, exceeding220 GiB.
The process stopped11:43:58; job122119 released11:58:59. This was not a
logged eigensolver convergence failure. Original logs and failed process
receipt remain in `raw/anthracene-ad.*`; no failed result is accepted.

The minimal repair is `cartesian_pullback(pullback, weight)`: saved residuals
enter JIT as dynamic PyTree inputs, remain available for all32 adjoints, and
are not donated or frozen. No integral/grid/SCF/XC/root/normal-mode precision
or physical dependency changes. `pullback_residual_bytes` is a sum of logical
leaf sizes, potentially including aliases; it is not a measured RSS value.

The actual B3LYP/H2 geometry JVP/VJP/FD check passed in job122195 with the new
boundary. The separate diagnostic initially assumed constants appeared in
the outer ClosedJaxpr, which was false for nested JIT; the corrected test uses
actual MLIR lowering warnings at4096 bytes. It reproduces24.60KB captured in
the old closure and no large-constant warning for the dynamic path, and finds
four vector residuals among lowered dynamic inputs. Two numerical origins
and two cotangents agree to2e-12. The same pullback reuses compilation for
different cotangents. Distinct pullbacks have different PyTree definitions;
the test records their additional compilation instead of claiming cross-
linearization executable reuse. These two diagnostic assertion errors were
preserved, not hidden as scientific failures or used to loosen tolerances.
The corrected mechanism test passed in2.001s (job122200).

The restart script verifies the exact production and B3LYP-test hashes before
reusing its passed XML testcase; it does not repeat the four-minute geometry
check. Native SHA and all physical settings are unchanged. The run uses a
fresh output, reuses immutable ground/Hessian/FD archives through explicit
paths, and rebuilds the in-memory VJP tape lost with the old process. Its
220 GiB RSS guard remains active. Final32 pullbacks and spectrum remain
unverified; preventing literal capture alone does not prove total workspace
will fit. Runtime boundary, raw receipts and status are in `restart_122200.json`.
