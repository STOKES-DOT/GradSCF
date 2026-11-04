# 共享 SCF/DIIS 求解器：实现与验证报告

更新：2026-11-04。工作分支：`perf/bse-tddft-grid3-20261101`；起点：`709e4ea`。实现位于隔离 worktree；未合并、未推送，原 release worktree 及用户改动保留。

**共享求解器、所列 CPU/AD 回归和严格菲案例已完成；60 体系正式矩阵正在运行。** 中性菲有同节点单次前后计时；阳离子旧路径未收敛，不能给出其有效速度比。尚无整体内存下降或 GPU 数值验收结论。原子初猜性能消融未执行，不是运行中。

## 1. 方法与实现

### 1.1 版本、环境与证据边界

生产候选冻结归档为 `/tmp/gradscf-scf-candidate-precision-20261102.tar.gz`，SHA256：

```text
91fdbb6eefaac9048fda3132f504a4d85da9ab3e9d555b4f9f00c0a14b986516
```

原远程目录为 `jy_gpu_2:~/private/homefile/gradscf-scf-solver-20261102/candidate-precision/`。2026-11-04 正式矩阵发现既有 GHF/GKS 校验问题后，创建 `candidate-roundoff-20261104/`；逐文件核验其 `src/**/*.py` 仅 `scf/gks.py` 与原冻结目录不同。两份源码保留。新目录只更改 eager Hermiticity 校验，详见2.5。独立 harness 位于 `harness/frozen-20261104/`，科学 worker 仍为原冻结 `runner-140f5dd1e3d1.py`；哈希在 `source-roundoff-20261104.json`。

Linux 原生库 SHA256 为 `ce51d885d4b65e753b5f56863470f1f1bb9c41d4b0762e47df5128e0d0b8feb8`。本次未修改 native C 源码。已记录的环境为 Python 3.11、JAX 0.10.0、NumPy 2.4.4、PySCF 2.13.0。所有量化测试在 Slurm 中执行，CPU affinity 明确限制；严格前后对照使用同一节点 n006 的 64 个逻辑 CPU。节点 CPU 型号未在本地交接记录中给出，不作推断。

资源上限为同时至多一个 allocation、1 GPU、64 个逻辑 CPU、256 GiB。所列通过证据来自 CPU；分配到 GPU 资源不代表完成 GPU 计算。

### 1.2 数值层与物理层分工

| 文件/模块 | 最终职责 |
| --- | --- |
| `src/gradscf/solvers/scf/types.py` | `SCFState`、`SCFConfig`、`SCFResult` 及指标/历史契约 |
| `src/gradscf/solvers/scf/iterate.py` | 单一有界 `lax.scan`，统一 DIIS 状态、damping、收敛冻结、最佳迭代与可选历史 |
| `src/gradscf/solvers/scf/metrics.py` | 占据加权 MO 残差及共同收敛策略 |
| `src/gradscf/solvers/scf/orthogonalization.py` | 有截止保护的实空间轨道正交化因子 |
| `src/gradscf/solvers/nonlinear/diis.py` | 既有规范 DIIS 环形历史与归一化 Gram 求解，继续复用 |
| `src/gradscf/scf/iteration.py` | restricted/unrestricted 密度、占据、level shift、历史选择与 J/K 缓存的薄适配 |
| `src/gradscf/scf/{rks,uks,gks}.py` | 保留各自 Fock/XC/积分物理构造，调用共同迭代器；RHF/UHF/GHF 经对应家族复用 |
| `src/gradscf/scf/differentiable.py` | 既有固定密度、显式扫描和隐式微分工作流调用共同适配器 |
| `src/gradscf/scf/init_guess.py` | 可选、受限的原生原子 UHF 密度叠加初猜 |
| `tools/scf_coefficient_gradients.py` | 已迁移至共同 restricted 适配器，保留逐步轨迹契约 |

数值层不导入电子结构模块。Fock/能量回调继续拥有积分、XC、交换及其辅助数据；ROHF/ROKS 保留原有共同轨道物理算法，仅迁移邻接工具的导入。

### 1.3 残差、收敛和精度处理

给定 MO 系数 \(C\)、原始 Fock \(F\) 与占据 \(n_i\)，共同残差为

\[
 F^{\rm MO}=C^\dagger FC,\qquad
 g=\frac{1}{\sqrt2}\left\|F^{\rm MO}_{ij}(n_j-n_i)\right\|_F.
\]

leading spin 维度共同进入范数。整数 restricted 占据下为 \(2\|F_{vo}\|\)；unrestricted 为两个自旋占据-虚轨道块的联合范数；广义单占据为 spinor 规范。非相等分数占据之间的旋转响应被保留，复数使用共轭转置。DIIS 仍由 AO 度量交换子构造误差，复杂误差以实部、虚部拼接形成实度量；收敛认证始终评估候选密度处的原始 Fock。

默认同时要求绝对能量差、AO 密度 RMS 和轨道梯度低于指定阈值，并要求 D、C、MO 能量、总能量、Fock 和指标有限。至少有一个先前周期后才允许收敛。`conv_tol=0` 保留固定周期执行；严格残差模式中的零密度阈值也保留训练的固定周期契约。能量单指标模式保留显式选择，并不跳过有限性检查。

直接 J/K 的增量累积若初步通过收敛门槛，会用同一密度重新构建 fresh J/K/Fock 后再认证；fresh 检查失败则继续原扫描。原有 R/U/G 家族的 DIIS 启动周期、damping 与 level shift 顺序保留。非有限历史的选择规则已统一：返回态与周期/RMS 元数据对应，全无效历史暴露失败态。

实空间内建 R/U 前向使用新的轨道因子：当 \(S-\epsilon I\) 正定时，\(S=LL^T\)、\(X=L^{-T}\)，从而 \(X^TSX=I\)。不满足截止条件时沿用原有截断对称逆平方根；默认 \(\epsilon=10^{-10}\)。全局 `inverse_sqrt` 与既有 DifferentiableSCF 对称轨道框架保留。此变换在非截断正定区域与广义本征问题等价；截止边界附近仍有分段算法选择，不能由已通过的小型导数测试推断任意跨边界平滑性。

能量差保留精确能量回调的相减语义；共同 API 提供可选 `energy_change` 回调，但生产中没有把不透明或神经 XC 能量静默替换为二次近似。

### 1.4 AD、内存及可选初猜

物理数组、模型参数和辅助 PyTree 保持 JAX 动态依赖。共同扫描没有新增 `stop_gradient`；固定密度、显式反向扫描、既有隐式反向路径均保留。验收涵盖所列 JVP/VJP/HVP、混合几何/force-loss 导数及 NNAO 参数/度量响应，不能扩展解释为所有方法、积分后端、几何和模型的完整微分支持。

常规前向仅保留标量历史；需要训练迭代选择或轨迹分析时才显式保留 D/C/ε 历史。DIIS 历史、当前/最佳态及 J/K 辅助数据仍占内存；显式反向模式也可能保留扫描中间量。因此本报告没有“AD 内存与迭代次数无关”的结论。严格大体系的整体 RSS 主要由准备阶段决定，尚无整体内存降低证据。

`init_guess='atom'` 使用原生孤立原子 UHF 密度块叠加，并按分子电荷/自旋缩放；它是非球平均 UHF 叠加，不能称为 PySCF 球平均 SAD。LRU 上限 128，缓存项是不可变 NumPy 常量，key 包含原始基组壳层和 native SHA256；冷缓存在外层几何 JIT 内具体求值，避免缓存 tracer。预算检查计入分子 resident storage。支持范围是 H–Ar 和 STO-3G、3-21G、6-31G*、cc-pVDZ、def2-SVP、def2-TZVP 六类已验证的命名全电子基组；ECP、ghost/修改核电荷、原始或可训练基组等明确拒绝。默认 hcore 初猜及已有模型基组路径保留；性能收益属于独立待完成消融。

### 1.5 正式比较协议与执行控制

冻结的 [plan-v2](plan-v2/run_plan.json) 包含 60 个物理体系，每种主基组 12 个：STO-3G、6-31G*、def2-SVP、cc-pVDZ、def2-TZVP；不是 60×5 的完整笛卡尔积。原子数 5–150，覆盖 H–Ar 中 15 种非惰性元素。SVWN、PBE、PBE0、CAM-B3LYP 四类各分配 15 个主体系，均使用 Cartesian AO、CPU float64、DFT grid level 3。

五个理想配位阳离子-水输入提供 Li/Be/Na/Mg/Al 覆盖；最大体系为 RAP 144 原子加两个远离水分子的 150 原子容量输入。它们是明确构造的 benchmark，不能称为实验或优化几何。其他输入的 PubChem/RCSB CCD 来源和文件哈希见 [fixture manifest](../../../benchmarks/probes/scf_solver/fixture_manifest.json) 与 [catalog](../../../benchmarks/probes/scf_solver/FIXTURE_CATALOG.md)。

注册规模为 858 个唯一 worker（每引擎 429），其中 720 个具有 primary scope；684 个配置要求单次运行，174 个要求 cold+3 次新对象重复，`repeats` 求和为 1380 次计划调用额度。去重后代表性跨基组/XC 控制与主案例可共享配置。NA、unsupported 或资源前置检查终止的 worker 不会实际执行全部调用额度。按配置可预先辨认 20 个 RHF/RKS 开壳层 NA worker 和 17 个 GradSCF GKS-RSH unsupported worker；这些是预期分类入口，不是已执行结果。

普通矩阵的三个门槛为 \(|\Delta E|<10^{-10}\) Ha、密度 RMS \(<10^{-8}\)、\(g<10^{-7}\)；PySCF 使用同一三项回调，包括其最终额外周期。能量配对门槛为 \(10^{-7}\) Ha。小 R/U（≤128 AO）采用 conventional packed；较大 R/U 采用明确 weigend 辅助 RI；RSH R/U 的注册策略为 direct；GHF/GKS 保持支持的 packed 路径及资源上限。RI 比较核对物理近似、AO 维数与 Naux，不能将 RI 和无辅助近似混为一组。

PySCF 可能同时保留普通 s8 ERI 并直接构建长程交换；runner 记录 `packed_plus_range_direct` 等实际策略及 retained ERI 字节数。两侧物理任务一致、维数/Naux 符合要求且能量通过，但实际策略不同时，汇总标为 `matched_task_different_strategy`。这种比值只比较端到端 facade 工作流，不是 SCF 核或同算法性能隔离结果。

`scf_facade_seconds` 包含新建分子/方法对象、积分与网格准备、SCF/JIT 和同步；imports、预先确定基组维数、最终诊断及状态文件 I/O 在区间之外。guard 的 whole-worker wall/RSS 覆盖这些额外工作。warm 样本是在相同进程、相同几何中新建对象，允许程序复用，不能称为独立冷启动。只有双方均要求一次运行时才给出 whole-process 比值；四次重复 worker 的总进程时间不当作一次 SCF 成本。

每 worker 的 `max_memory=180000` decimal MB，RSS guard 为 200 GiB；小案例 CPU8、大案例 CPU64，单 facade 上限分别 1800 s/6000 s，whole-worker 上限为 `repeats × 单次上限 + 120 s`。超时作用域明确记录。controller 串行使用已有 allocation，不额外提交资源；不足剩余时间的案例进入 deferred。恢复时从不可变 guard/worker 记录重建 `.run.json`；无 process 记录的旧日志保留为 uncertain，人工核实后才能重试。接纳必须同时满足 guard 成功、终态 ok、完整重复数、每次科学通过，后续失败不能被之前的成功重复掩盖。

## 2. 已获得结果

### 2.1 冻结生产候选的必要 CPU 回归

| 最终测试组 | 报告通过数 | 耗时 / s | 覆盖范围 |
| --- | ---: | ---: | --- |
| core / complex / AD / representation / adjacent RO | 99 | 84.36 | 共同数值层、六方法、精度和相邻算法边界 |
| optional atomic guess | 20 | 25.74 | 公开入口、缓存/预算、命名基组边界、冷几何 JIT/FD |
| neural / force | 11 | 143.76 | 神经/自定义 XC、受支持 force 与混合导数 |
| NNAO | 8 | 64.26 | 参数、RI 度量、HVP 等受支持 fixtures |
| small MACE | 1 | 20.25 | 双几何更新和 checkpoint fixture |

另有12项 neural guards/streamed mixed-AD 回归通过47.78s；原 harness 元数据7项通过0.10s。以上不与早期83项 baseline、120项初版 candidate 或其他探索性组累加，也不宣称完整套件通过。2026-11-04 小补丁的 GHF/GKS23项通过16.53s（含既有测试，不能作全新23项累加）；更正记录/汇总元数据8项通过0.11s。

较早的组合 neural 进程发生编译阶段 ABRT；对应必要案例后来在新进程分组通过。这证明所列分组可通过，不证明全局编译问题已修复。额外 legacy DifferentiableSCF 探索先有 42 项通过，随后一个把合法 `explicit` 当作非法值的旧测试失败；该测试在 baseline 同样失败，改为检查 `unknown` 后通过，未回退生产 API。

GPU-only array 组在 CUDA/NUMA 初始化时停止，记录含 `hwloc_set_cpubind Invalid argument`，尚未进入可接受的 GPU 数值验证；没有 GPU PASS 或 GPU 性能结论。

### 2.2 同节点严格阳离子前后对照

体系：固定归档几何的 phenanthrene radical cation，24 原子，Cartesian def2-TZVP，564 AO，UKS/PBE/grid3，weigend RI，CPU64，同一 n006 节点，hcore 初猜，最多 200 周期。保持原始更严格门槛：能量差 \(10^{-12}\) Ha、密度 RMS \(10^{-11}\)、轨道梯度 \(10^{-10}\)。

| 实现 | 周期数 | facade 时间 / s | 最终物理 MO 证书 | 原实现收敛标志 | 比较接纳 |
| --- | ---: | ---: | ---: | --- | --- |
| clean legacy baseline | 200 | 1299.5693 | 约 \(3.45\times10^{-11}\) | false：旧 AO 残差门槛拒绝 | 否 |
| frozen precision candidate | 69 | 468.2799 | \(2.130111161582478\times10^{-11}\) | true | 通过本案例收敛门槛 |

候选总能量为 −538.755078177922 Ha。baseline 在其自身门槛下未通过，即使事后物理 MO 证书已小，也不能把这两个时间组成有效速度比；本报告不计算或报告该比值。legacy API 兼容仅存在于显式 `solver_api='legacy'` 的 benchmark 证书路径，没有生产积分后端 fallback。

两侧整体 RSS 均约 9.7 GB，候选记录约 9.696 GB，主要由准备阶段决定。此案例支持消除旧残差认证/精度瓶颈并减少到达正式收敛的周期；不能据此宣称整体内存下降，也不能推广到其他体系、泛函、基组或后端。

中性 phenanthrene 的同节点、同原门槛对照已回收：

| 实现 | 周期数 | facade 时间 / s | 最终 MO 梯度 | 科学通过 |
| --- | ---: | ---: | ---: | --- |
| clean legacy baseline | 82 | 493.779866 | 3.627251784058428e-11 | 是 |
| frozen precision candidate | 51 | 256.654833 | 7.313091486604222e-11 | 是 |

能量差3.61e-11Ha。这是两个独立新进程的单次观察，没有三次 warm 样本，不作正式统计速度比。原 atom 消融脚本随旧分配停止，两个消融尚未执行。

### 2.3 六个 harness smoke

Methane/STO-3G/PBE/grid3、CPU8、hcore 的 RKS/UKS/GKS × GradSCF/PySCF 六个 worker 均科学通过，成对能量差约 \(10^{-13}\) Ha。该 smoke 验证三种 KS 布局、双引擎调度和记录路径；它不替代多基组、多泛函、大体系或四次重复控制验收。

### 2.4 正式矩阵状态

**60 系统 / 858 worker 的正式 plan-v2 正在 allocation115757 上运行；1380次注册调用额度不是完成数。** 2026-11-04T14:52:37+08快照为206个唯一配置已尝试，188个科学通过、8个 unsupported、10个 PySCF 配置未通过最终共同门槛；覆盖18个体系的至少一个配置，并非18个体系全部完成。有效能量配对85组，最大能量差3.4618e-11Ha。当前主矩阵尝试范围为5–8原子，不能代替计划中的大体系验收；24原子菲的严格对照见2.2。实时可重建汇总在远端 `results/summary-20261104/`，静态快照以保存时间为准。本报告尚无全矩阵收敛率、分层速度或容量曲线。`ngrid` 不可得时为 null，不是没有网格。

原分配114687日志显示于2026-11-02T14:04:11取消；114720也已不在队列，旧正式矩阵未启动。恢复分配115757从2026-11-04开始，NV5090×1、64逻辑CPU、256GiB、48h，科学计算只在Slurm step内进行。配置串行测试；已有完成输出不覆盖，未完成配置继续。

### 2.5 正式矩阵暴露的问题及局部修复

氯甲烷/def2-TZVP（96AO）的原生 hcore 全部有限，最大元素146.905123Ha，最大反 Hermitian 差1.100786e-13Ha。既有 GHF/GKS 输入 gate 使用固定 `100*eps=2.220446e-14`，在进入共同 solver 前拒绝该积分。该 gate 与709e4ea相同，属前既存问题。

修复为 `100*eps*max(1,max(abs(A)))`，不修改输入数组、不对称化、不改变物理积分或 AD 边界。两个接受舍入的测试先 RED，其余5项拒绝无效输入测试通过；修复后连同既有复杂 spinor/GHF/GKS 回归共23项通过。五个受影响的初始配置均单独复测并收敛：氯甲烷GHF、氟甲烷GHF/PBE0-GKS、甲酸GHF/SVWN-GKS。原日志保留在 `final-matrix/`，更正尝试在 `roundoff-retries-20261104/`；`--overlay` 汇总显式标明被替代记录和源码。

10个 PySCF 未接纳配置运行了约14–17步，其最终额外周期的梯度略高于登记的1e-7门槛（约1.04e-7至2.26e-7）。这是本benchmark的严格最终状态验收失败，不能表述为PySCF在200步内无法收敛或一般性的程序缺陷；不据此计算速度比。

另有 NFS 跨节点可见性延迟：guard 已写完 `.process.json` 并退出，login端即时派生记录有时暂时读不到文件，显示 `interrupted_or_running`。原记录可读后重新派生即可恢复，不重跑科学计算、不把 exit0 当科学通过。`refresh_results.py` 与每300s的元数据 watcher 实施恢复。

最初43个配置部分与小诊断重叠，保守全部列入 `timing-exclusions-20261104.json`，仍保留科学结果，但不计算速度比。后续 clean 运行无并行诊断。物理匹配、过程完整、科学通过与计时是否可用是独立字段。

## 3. 支持范围与待完成门槛

当前保留 RKS/UKS 的既有神经/自定义 XC 训练与 fixed_density/explicit/implicit 功能。共享数值层不新增 spinor GKS 神经/自定义 XC 训练，也不新增 GKS 的 DF/direct/RSH/MGGA 或 RI geometry tracing。收敛认证不是稳定性分析或唯一基态证明。

尚需完成正式矩阵，按方法、基组、XC、近似与实际策略汇总，完成资源清理。原子初猜性能消融未运行，不以单元测试替代性能数据。NA、unsupported、resource_limited、timeout、error、nonconverged、deferred、uncertain 均保留，不进入科学通过或速度比较。本报告不是全工程验收完成声明。

## 4. 证据来源与复核入口

| 内容 | 证据入口 |
| --- | --- |
| 实现授权、边界、分组及原始严格门槛 | [SCF_SOLVER_PLAN.md](../../response_grid3/20261101-2043d7d/SCF_SOLVER_PLAN.md) |
| 归档/native hash、环境、远程目录和运行流程 | [handoff.md](handoff.md)；其中 baseline/smoke 的 queued/running 描述早于本草稿回收状态 |
| 冻结生产分组和当前状态 | [state.json](state.json)、[validation_summary.json](validation_summary.json) |
| 原子数、来源、元素、基组与几何限制 | [fixture_manifest.json](../../../benchmarks/probes/scf_solver/fixture_manifest.json)、[systems.json](../../../benchmarks/probes/scf_solver/systems.json) |
| 注册配置与额度 | [plan-v2/run_plan.json](plan-v2/run_plan.json) 及同目录 `configs/`；plan-v1 已 superseded，未执行 |
| 严格前后对照输入 | [before-after-configs/](before-after-configs/)；原始结果位于远程 campaign 的 `results/before-after/` |
| 六个 smoke 输入与结果 | [smoke-configs/](smoke-configs/)；原始结果位于远程 campaign 的 `results/smoke/` |
| runner/controller/summary 及元数据回归 | [scf_solver harness](../../../benchmarks/probes/scf_solver/)、[test_scf_campaign_records.py](../../../tests/test_scf_campaign_records.py) |
| 同节点四个严格 SCF 观察 | [before_after.json](before_after.json) |
| 本地/远端源码一致性及环境 | [source-roundoff-20261104.json](source-roundoff-20261104.json)、[environment.json](environment.json) |
| 进行中的矩阵快照 | [counts.json](closing-summary-20261104/counts.json)、[summary.csv](closing-summary-20261104/summary.csv)；不是最终验收 |

原始日志、JUnit、guard 和逐次测量已同步至本地 `raw/`（Git忽略），远端仍是权威原始来源。结构化验收摘要保留JUnit哈希。跨 allocation 恢复时，以 worker `runtime.json` 的实际job/host和原始guard确定谱系，不用重建派生记录时的新lease名代替原执行allocation。上述静态快照随采集时间而定；实时结果由远端watcher更新。
