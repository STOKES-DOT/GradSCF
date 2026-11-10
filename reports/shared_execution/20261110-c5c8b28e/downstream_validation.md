# 下游数值接受与导数凭据审计

2026-11-10；依据本目录已经下载的 queue08 JSONL、manifest、comparison 和 JUnit XML。

三组 parent/candidate 对照全部通过数值比较，共 98 对阶段记录：H2 六种均场/TD 48 对、H2 四种 GW/BSE 40 对、水 G0W0/BSE 10 对。所有**实际记录的有限数值**逐项差为 0.0 Ha，computed/converged/branch、稳定性与 screening/response flags 一致；缺失诊断的 `null` 不属于零残差。这里接受的是相同输入、相同算法的数值保持，不是基组/频率网格或外部参考精度收敛。

## 输入与源码证据

- Slurm job `119123`，node `n006`，AMD EPYC 9654；这些工作流均为 CPU float64、affinity `[8]`。parent steps 42/43/44 与 candidate steps 45/46/47 顺序执行，全部 process returncode=0。`OMP/OPENBLAS/MKL_NUM_THREADS=1`、`GRADSCF_NATIVE_THREADS=8`、JAX compilation cache=false，配对环境相同。
- JAX/JAXLIB 0.10.0、NumPy 2.4.4、SciPy 1.17.1；相同 native SHA256 `9f8da65e0e55…` 与 probe SHA256 `98f7ee1f656d…`。manifest 中 `git_head` 是 parent/candidate 标签；以源码 hashes 确认谱系：559 个可比 parent Python 源文件均与 `initial_manifest.json` 的基线 `c5c8b28e8838132c30b6074c0690f029a87edabe` 一致。
- parent/candidate 仅四个生产源文件 hash 改变：`tools/features.py`、`tddft/execution.py`、`tddft/response.py`、`tddft/casida.py`。**GW/BSE 生产源码完全相同**；已核对 candidate manifest 的全部现存 Python 文件与当前候选一致。
- H2：0.74 Angstrom，Cartesian STO-3G，PBE/grid0，GW `nw=16`。水：O(0,0,0)、H(0,±0.757,0.587) Angstrom，Cartesian def2-SVP（25 AO），PBE0/grid3，`nw=32`。均为 full SCF，GW 使用真实 SCF exact-pair spectral factors（tol=1e-12），没有供应或冻结 QP/screening poles。
- SCF energy/density/orbital-gradient 阈值为 1e-11 Ha/1e-10/1e-9；GW outer tol=1e-7 Ha、max_cycle=80、damp=.3；scGW beta=20 Ha^-1、mixing=.5；BSE Davidson 1 root、tol=1e-9。每种计算包含首次调用和一次 fresh-object warm 重复。

证据：[SCF/TD 对照](comparison-scf-td.json)、[H2 GW/BSE 对照](comparison-gw-bse.json)、[水 GW/BSE 对照](comparison-water-gw-bse.json)，及 `raw/{parent,candidate}-*.manifest.json`、`*.process.json`。

## 最强的 GW/BSE 结果

下表取 candidate warm 记录；parent 与首次调用的相应数值相同。能量与残差单位均为 Ha。

| 体系/方法 | 最大已记录 QP 残差 | BSE-TDA 最低根 | full BSE 最低根 | 最大 BSE 残差 |
| --- | ---: | ---: | ---: | ---: |
| H2/PBE → G0W0 | 1.5534e-12 | 0.9937300677584563 | 0.9561325943682494 | 1.5919e-16 |
| H2/PBE → evGW | 9.8253e-8 | 0.9847018473753023 | 0.9533415736028890 | 4.9648e-16 |
| H2/PBE → qsGW | 未提供该诊断 | 0.9847018473753023 | 0.9533415736028890 | 4.9648e-16 |
| 水/PBE0 → G0W0 | 2.5843e-9 | 0.2799177651932867 | 0.2781340077459504 | 1.6109e-10 |

前三种 H2 GW 的全部 2 个 QP、以及水全部 25 个 QP，computed/converged masks 均为 true；G0W0/evGW 已提供的 branch masks 也全部为 true。上述全部 BSE roots 的 converged/stable/response_valid/screening_valid 为 true。G0W0 记录的是实际 KS screening spectrum，evGW 记录的 screening spectrum 与其更新 QP 一致；没有把初始 KS 水平假充已计算 QP。

H2 Matsubara scGW 真实前向收敛：44 iterations，fixed-point residual=9.912610573939688e-8 Ha，particle-number error=0，total energy=-1.136829584446466 Ha；`mo_energy=null`。这证明有限温 imaginary-axis 求解路径仍可运行，不提供静态 QP/BSE 谱。

六种 H2 SCF 均收敛，HF 三种总能量约 -1.116759307396426 Ha，PBE 三种约 -1.151021650280414 Ha；RHF/UHF/RKS/UKS 的 TDA/full TD 前向完成。该 runner 的 TD residual 字段均为 null，因此这里只报告能量与 convergence parity，不声称测得了 TD 算符残差。

## 导数与训练凭据

`raw/ad-families.xml` 精确结果为 **93 passed、2 failed、0 skipped**，pytest 1191.107 s、process returncode=1。两项失败均是 `test_neural_scf_force_training[implicit/explicit]` 在 `runpy` 读取 `examples/training/train_neural_scf_forces.py` 时 FileNotFoundError，未进入数值验证。

已实际通过的相关门包括：

- `test_molecular_gw_hybrid.py`：18/18，包括 PBE0、wb97xd × G0W0/evGW/qsGW/scGW 的 8 个完整 relaxed-KS/native-geometry/G/W JVP/VJP 测试；每个测试在 3e-4、1e-4 Bohr 位移重新求解并检查有限差分（atol=3e-6、rtol=4e-5），同时检查 JVP/VJP 一致性与平移和规则。其 scGW 断言包含 particle error<1e-9、fixed-point residual<1e-8；XML 不记录实际最大导数误差，不能把断言容差写成测得误差。
- `test_response_external_memory.py`：23/23，包括 restricted 三种训练模式与 unrestricted 两种占据 × 三种模式（共 9 个 fixed_density/explicit/implicit 训练门），以及完整四层 AO/global-grid NN Hessian action 与 direct J/K 消费者选择。
- `test_scf_orbital_native_geometry.py`：4/4，UKS/GKS × implicit/unrolled，native 几何 VJP 对 1e-4 Bohr 五点差分；implicit 分支另含二阶曲率检查。
- `raw/policy-green.xml`：10/10，包括 auto+tol0、预算拒绝 packed cache 后的 direct 程序复用，direct custom_xc 参数 JVP/VJP/FD、动态更新与 opaque fallback；不能由前向性能对照替代这些门。
- `raw/integration.xml` 中新 `test_workflow_reuse_integration.py` 模块 14/14，通过真实 GW/BSE 来源、已有 SCF 状态保持、screening 与能力边界检查；该整批总体是 51 passed、2 failed，另外两项是 streamed preparation 的旧计数断言，不能称整批全绿。

独立重跑凭据已返回：`raw/force-training.xml` 为 **5 passed / 52.02 s**，
`raw/force-training.process.json` 为 returncode=0（2 force-training、2 NNAO direct-energy、
1 methane basis-gradient）。此前 missing-fixture 两项已由独立重跑覆盖，原始失败记录保留。
旧 `raw/final-regressions.process.json` 的 returncode=-11 继续保留，不计为通过；
拆分新进程重跑已通过：streamed25、真实工作流14、BSE/fusion17、SCF/GW/BSE reuse20；另新增共享direct坐标AD2项通过。原混合进程崩溃的根因仍未确定。

## 支持边界与计时解释

- GHF/GKS 仅完成广义 SCF。两种 TD 调用均真实拒绝 `Generalized spinor response is not implemented`；8 个配对 capability rows 不是 TD 结果。
- G0W0/evGW/qsGW 的静态 TDA/full BSE 链已在上述真实数据上完成。scGW 两种 BSE 调用真实拒绝缺少 validated analytic continuation/QP coverage；4 个配对 capability rows 不计作 BSE 或完整光谱成功。
- 四种 GW 的前向/导数综合接受限于这些小分子、指定 exact-pair approximation 与显式频率网格；此批没有新的 benzene GW/BSE、广义 spinor GW/TD 或 scGW→BSE 结果。
- 每个条件只有**一个 warm fresh-object 样本**，首次调用还混合编译、执行及前序 case 的缓存状态。相同 CPU/affinity 不能消除时序负载、频率和运行环境漂移；未改动的 GW 阶段也广泛变快，H2 G0W0/evGW/qsGW 的 warm candidate/parent 时间比约 .61/.60/.61。因此不能把这些时间差归因于 GW 算法优化。
- 水的 warm G0W0→BSE 完整流程为 92.1797→80.3781 s，full BSE 为 .359623→.358476 s；这只是一次配对观测，**本报告不主张 GW/BSE 加速**。水累计 process peak RSS 为 2.76273→2.80269 GiB，H2 GW/BSE 为 1.43171→1.44313 GiB；这是各进程累计峰值，不是单独阶段新增内存，也没有显著省内存证据。

数值保持已有明确证据；missing-fixture 重跑已通过；崩溃批次的独立重跑结果另见主报告。
