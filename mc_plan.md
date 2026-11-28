# GradSCF `mc` v1 固定 TDD

基线 `a3555de5`（`release/v1.0.0`），开发分支 `feat/mc-casscf`。这是实现前冻结的测试契约。RED 必须指向预期缺失的行为。GREEN 应同时运行本阶段与上游依赖的回归。参考值、容差、测试体系和方法范围发生变化时，应单独说明证据和原因，经确认后再更新本计划。

## 1. v1 交付与接口

v1 交付分子实轨道、单态、固定总自旋的 `gradscf.mc.CASCI` 和 `gradscf.mc.CASSCF`。CASCI 是固定轨道的基础接口。CASSCF 在给定 AO 基组下共同优化空间 MO 系数 \(C_{\mu p}\) 和活性空间的 CSF 系数 \(c\)。覆盖 RHF 与 ROHF 参考，包含 singlet（包括开壳层单重态）、doublet 和 triplet 的小体系验证。v1 的活动空间由用户指定索引，省略索引时按非活性、活性、外部的连续分区构建。状态平均、自动选空间、多组态自洽的 RAS/GAS、点群及复数轨道在独立阶段扩展。

```python
from gradscf import mc

obj = mc.CASSCF(mf, ncas=2, nelecas=(1, 1), spin2=0, active=(1, 2))
obj.run()
result = obj.result
```

- `ncas` 是活性空间轨道数。`nelecas=(nalpha, nbeta)` 指活动电子数，`spin2=2S` 指目标总自旋。输入中 \(2M_S=n_\alpha-n_\beta\)，须满足 \(|2M_S|\le 2S\)、奇偶一致及有限空间的占据约束。
- 非活性轨道保持双占据，CASSCF 中可以与活性或外部轨道旋转。显式冻结轨道将来作为独立约束。非连续 `core` 和 `active` 的声明顺序保留，重叠、重复和越界索引报错。
- `run()` 沿 GradSCF facade 习惯返回求解对象。把总能量、活性根、`mo_coeff`、`ci_coeff`、分区、自旋、CI 残差、轨道梯度范数和迭代数存入结果。`make_rdm12()` 提供活动空间自旋求和 1-/2-RDM，`make_rdm1s()` 提供活动空间分自旋 1-RDM，后处理需要全体系 RDM 时由非活性密度与活动 RDM 合成。
- eager facade 从收敛 RHF/ROHF 读取一次参考快照并校验源。数值核心接收数组和静态空间拓扑，结果拥有自己的轨道、CI 和 RDM。跨几何复用参考时的轨道匹配与选根规则属于显式输入契约。

## 2. 参考实现与分层

| 层 | 主要参考和落点 |
| --- | --- |
| CSF 图、σ、RDM | OpenMolcas 后续提交 `fdcaaa2de76e99a956ac3b3620c8703df285dc31`。`proc_inp.F90` 的 `DoSGCI` 分派和 `CI_Lib_Select` 选择直接 SGUGA，`CSFG_Make`、`SG_Init` 准备弧权重、片段和耦合表，`ci_lib_core` 对接 graph σ、对角与密度。参考 [后端选择](https://github.com/Molcas/OpenMolcas/blob/fdcaaa2de76e99a956ac3b3620c8703df285dc31/src/rasscf/proc_inp.F90#L4136-L4222)、[图初始化](https://github.com/Molcas/OpenMolcas/blob/fdcaaa2de76e99a956ac3b3620c8703df285dc31/src/sguga_util/sguga_setup.F90#L30-L83)、[直接 σ 与 RDM](https://github.com/Molcas/OpenMolcas/blob/fdcaaa2de76e99a956ac3b3620c8703df285dc31/src/ci_lib_util/ci_lib_core.F90#L1939-L2033)。 |
| 轨道主算法 | OpenMolcas `v26.10`（`32a60dfd712fb951e66e775fd4a515b5b65a5ac3`）的 RASSCF 宏迭代、Super-CI 和轨道旋转。该标签在 CSF 坐标求 CI，默认 σ 借助 Lucia 的行列式接口。较新的在线手册标注 `v26.10-1057-gfdcaaa2de`，对应上一行的直接 SGUGA。参考 [v26.10 CI 分派](https://github.com/Molcas/OpenMolcas/blob/v26.10/src/ci_interfaces_util/ci_interfaces.F90#L89-L151)、[宏迭代](https://github.com/Molcas/OpenMolcas/blob/v26.10/src/rasscf/rasscf.F90#L919-L995)、[手册版本](https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/rasscf.html)。 |
| Python 组织和数值对照 | PySCF `v2.9.0` 提供 CASCI/CASSCF facade、初始轨道排列、可插拔 CI solver、RDM 接口和独立数值对照。轨道主步按本计划的 OpenMolcas Super-CI 路线设计。参考 [CASCI](https://github.com/pyscf/pyscf/blob/v2.9.0/pyscf/mcscf/casci.py#L564)、[CASSCF facade 与 kernel](https://github.com/pyscf/pyscf/blob/v2.9.0/pyscf/mcscf/mc1step.py#L333)。 |
| GradSCF 复用 | `src/gradscf/fci/reference.py` 的 `fold_core` 和 `_active_from_orbitals` 已有固定轨道 CAS Hamiltonian；`ao2mo.blocks.transform_eri_blocks` 可批量请求指定 MO 块；`solvers.eigen.solve_hermitian`、`LinearOperator`、`attach_root` 为求谱和响应入口。原 determinant FCI 在小空间作独立对照。 |

### 2.1 总自旋适配 CI

1. 活动轨道按顺序耦合自旋。DRT 的节点携带层、累计电子数及中间自旋，弧对应空、双占及两种单占自旋耦合步。有终点约束的路径对应一个 CSF。先计算每节点的可达路径数和弧权重，得到确定顺序的 rank/unrank。大空间只保留图、必要的局部耦合和系数向量。
2. 以自旋自由生成元
   \(E_{pq}=\sum_{\sigma}a^\dagger_{p\sigma}a_{q\sigma}\)、
   \(e_{pq,rs}=E_{pq}E_{rs}-\delta_{qr}E_{ps}\) 表示 Hamiltonian。对照 OpenMolcas 的 split-graph 片段和耦合规则实现局部作用，CI solver 通过 `matvec/matmat` 获得 \(\sigma=H_Ac\) 和对角预条件器。小空间显式矩阵只用于独立验证或明确的 dense 选择。
3. 1-/2-RDM 与 σ 共用自旋自由生成元及 CI 路径约定。通过迹、缩迹、对称性和能量重建验证索引。能量使用化学家记号及自旋求和、**有序**的二体 RDM，其中 \(\mathrm{Tr}\,D=N\)、\(\sum_{pr}\Gamma_{pp,rr}=N(N-1)\)。
4. DRT 精确固定 \(S\)。闭壳层 singlet、破键 singlet、奇电子 doublet 和 triplet 使用同一图流程。组合增长仍体现在 CSF 数和每个 Davidson 向量的长度，预算需计入多根、子空间和 RDM 工作区。

### 2.2 AO→MO 和轨道–CI 宏迭代

非活性密度 \(D_c=2C_cC_c^\mathsf{T}\)，活动单电子矩阵
\[
h^A=C_A^\mathsf{T}\bigl(h+J[D_c]-K[D_c]/2\bigr)C_A.
\]
由同一 \(D_c\) 计算非活性能量，核排斥作为显式标量，只加一次。首先复用当前固定空间 CAS 折叠路径；把复用范围抽到 `ao2mo` 的中性接口时，原 `fci` 调用也须通过其既有测试。

- 每次旋转后构造活性 `aaaa` 和轨道梯度/Super-CI 所需的混合块。先按消费式列出索引请求，例如 \(paaa\) 型请求按实际使用的置换复用，再批量调用已有 AO2MO。用同一个 C 和积分快照生成一体与二体数据。预算从活性大小、块输出和消费者状态得出。native-direct 和 DF 路径按选定块计划，密集全 AO/MO 路径作为小规模 oracle。
- 非冗余轨道角取非活性↔活性、非活性↔外部、活性↔外部。\(C'=C\exp(\kappa)\)、\(\kappa^\mathsf{T}=-\kappa\)。同组旋转是规范自由度，由约化坐标去掉。输入/输出校验 \(C^\mathsf{T}SC=I\)。
- 前向宏迭代按 OpenMolcas 流程分工：在当前 \(C\) 上准备积分，求精确目标自旋的 CI 根和 RDM，装配轨道梯度与 Super-CI 模型，求旋转试探步并重新评估，再开始下一宏步。Super-CI 参照 `SXHAM → DAVCRE →` 用参考分量归一旋转参数 `→ ROTORB`。小空间保留显式模型 oracle，大空间按算子求低根。参考 [Super-CI 构造与求步](https://github.com/Molcas/OpenMolcas/blob/v26.10/src/rasscf/sxctl.F90#L456-L577)、[指数旋转](https://github.com/Molcas/OpenMolcas/blob/v26.10/src/rasscf/rotorb.F90#L266-L282)。
- v1 的主步是 Super-CI。能级移动、步限制和可选 `QUNE` 加速遵照独立的前向策略。`QUNE` 参考低秩逆 Hessian 历史、线搜索与能量上升的阻尼。输出逐宏步的能量、CI 残差、轨道梯度和旋转幅度。最终收敛同时检查 CI、轨道驻点和能量变化。OpenMolcas 的数值默认值和迭代次数用于诊断，验收以指定物理误差为准。 [QUNE 参考](https://github.com/Molcas/OpenMolcas/blob/v26.10/src/rasscf/qune.F90#L249-L308) · [收敛检查](https://github.com/Molcas/OpenMolcas/blob/v26.10/src/rasscf/rasscf.F90#L1367-L1393)

### 2.3 AD 合同

显式物理输入至少包括 \(S,h,g,E_{\rm nn}\)，积分也可由同一密度拟合因子或 native 数据提供。用随动度规的轨道框架表达 \(C^\mathsf{T}SC=I\)，使 \(\delta S\) 和电子积分的变化参与导数。固定 `(N, ncas, spin2, core, active, root)` 的局部分支上，定义
\[
R(z;p)=\left(
 \partial_{\kappa} E(C,c;p),\,
 (H_A(C;p)-\mathcal E)c,\,
 \tfrac12(c^\mathsf{T}c-1)
\right),\quad z=(\kappa,c,\mathcal E).
\]
\(\mathcal E\) 是活动 Hamiltonian 的本征值，总能量另加非活性能量和核排斥。用图 σ 表示 CI 部分，将轨道梯度对 CI 的依赖和 CI 方程对轨道的依赖一起保留。通过 `jax.linearize` 和转置作用构造耦合的 \(Jv,J^\mathsf{T}v\)，复用 `attach_root` 的隐式 JVP/VJP。前向 Super-CI、试探步、加速和 Davidson 选择属于数值求解；导数由已收敛的物理方程决定。

- **能量导数和状态导数分开验收。** 能量可利用驻点性质。CI、1-/2-RDM 和后续 PT/PDFT 的响应必须含轨道和 CI 的耦合变化。相位用与经验证的参考态重叠为正的局部规范固定，比较物理 RDM 或 AO 量时消去组内轨道规约。
- **按后端显式约定微分能力。** 在数组接口验收 \(S,h,g,E_{\rm nn}\) 的 JVP/VJP。DF 因子作为活数组验收。原生路径至少验收 MO 系数和固定拓扑的核坐标响应；核基参数梯度在已有可微 AO 数组来源上贯通。原生 selected AO2MO 当前的指数/收缩系数响应是单独的能力闸门，调用不匹配的物理方向时明确报出。所有交付的能量与 RDM 路径要在相应能力内保留响应。
- 固定空间的孤立、收敛根允许隐式微分。非法电子/空间计数、未收敛、根空间切换和奇异响应均有可检查诊断。第一版按固定单态分支验收。后续态平均进入前先固定根集合、权重和规范，处理等权隔离子空间及近简并问题。

## 3. 冻结参照

### 3.1 本次已运行的独立参照

PySCF `2.9.0`，全电子，`cart=False`，`symmetry=False`，电荷为 0，坐标单位 Å。`spin=0` 用 RHF，其余用 ROHF。HF 参数 `conv_tol=1e-13`、`conv_tol_grad=1e-9`、`max_cycle=100`。CASSCF 参数 `conv_tol=1e-12`、`conv_tol_grad=1e-8`、`max_cycle_macro=100`，先从同一 HF 轨道计算固定轨道 CASCI，之后从该 HF 初始框架独立优化，MO 索引从 0 开始。默认活性列为从 `ncore` 起的 `ncas` 列。

| ID | 确切输入 | 非活性 / CAS / \(2S\) | \(E_{\rm CASCI}\) | \(E_{\rm CASSCF}\) |
|---|---|---|---:|---:|
| H2-S | H \((0,0,0)\)、H \((0,0,0.74)\)，STO-3G | 0 / (2e,2o) / 0，`nelecas=(1,1)` | −1.137283834488502 | −1.137283834488502 |
| H2-T | 同 H2-S，ROHF `spin=2` | 0 / (2e,2o) / 2，`nelecas=(2,0)` | −0.530773357001458 | −0.530773357001458 |
| H2-R | H \((0,0,0)\)、H \((0,0,1.4)\)，6-31G | 0 / (2e,2o) / 0，`nelecas=(1,1)` | −1.051606680216211 | −1.065638305056641 |
| LiH-C | Li \((0,0,0)\)、H \((0,0,1.6)\)，STO-3G | 1 / (2e,2o) / 0，`nelecas=(1,1)` | −7.862128833372132 | −7.881045251314651 |
| H3-D | H \((0,0,0)\)、H \((0,0,0.90)\)、H \((0.85,0,0.31)\)，6-31G，ROHF `spin=1` | 0 / (3e,3o) / 1，`nelecas=(2,1)` | −1.510677772022706 | −1.524105411664464 |

能量单位 \(E_h\)，包括核排斥。H2-S、H2-T、H3-D 的 \(\langle S^2\rangle\) 分别为 \(0,2,0.75\)。H2-R 移动第二个 H 的 \(z\)，PySCF 解析 CASSCF 梯度为 \(+0.067936230549270\,E_h/a_0\)。完全重收敛的中心差分，步长 \(2\times10^{-3}\) 和 \(10^{-3}\,a_0\)，分别为 \(+0.067936320232942\) 和 \(+0.067936333740581\,E_h/a_0\)。Å/Bohr 参考换算为 `0.52917721092`。

H3-D 的 PySCF 解析梯度固定为
\[
\begin{pmatrix}
-0.015587542014807&0&-0.028379625416768\\
 0.094323792061573&0&-0.042777051454901\\
-0.078736250046767&0& 0.071156676871669
\end{pmatrix}E_h/a_0.
\]

### 3.2 OpenMolcas 发布版交叉参照

官方 [`test/standard/000.input`](https://github.com/Molcas/OpenMolcas/blob/v26.10/test/standard/000.input#L12-L34) 定义 H₂ 在 \(x=\pm0.35\) Å、`h.DZ....` 基组、原始对称性和 CAS(2,2)。同一输入的 [校验行](https://github.com/Molcas/OpenMolcas/blob/v26.10/test/standard/000.input#L70-L72) 给出 \(E_{\rm RASSCF}=-1.141134852262\,E_h\)。将这一输入作为独立交叉复现。对齐 OpenMolcas 导出的收缩基组和初始活性轨道后，比较 GradSCF 的同一 singlet 总能量，容差 \(2\times10^{-7}\,E_h\)。本轮证据是固定发布版的已核对官方记录。实际交叉运行时另保存输入、OpenMolcas 日志及交换的 S/h/ERI；它与 3.1 的 PySCF 对照分别报告。

### 3.3 现有 GradSCF 基线

从本工作树 `a3555de5` 运行 `tests/fci/test_response.py`，本机 JAX 0.8.1 的结果是 **5 passed、2 failed**。两个已有 `test_energy_and_density_jvp_vjp` 参数（dense、Davidson）都在 Hermitian eigenpair 响应的 `lax.map` 转置过程中报 `ValAccum` 类型错误。B00 将这一结果记录为实现前闸门。用已经具备的 JAX/JAXLIB 0.10.0 Linux CPU float64 环境复测；若仍复现，按共享 eigenpair 响应缺陷独立解决，再开展依赖它的 CASSCF 状态微分。能量通过与 CI/RDM 响应通过分别验收。

## 4. 固定验收矩阵

误差均为绝对误差，除非条目另列相对量。微分与对照测试用 CPU float64，基态根采用固定确定性输入。实现时先写对应的真实用例，观察 RED，再写满足它的最小实现。

### G. CSF 图、σ 与 CASCI

| ID | 固定输入与独立 oracle | GREEN 条件 |
|---|---|---|
| G01 | CAS(2,2) 的 singlet/triplet，CAS(3,3) 的 doublet/quartet，CAS(4,4) 的 singlet/triplet/quintet。用 \(d_S=d(M_S=S)-d(M_S=S+1)\) 独立计数。 | 路径数分别为 3/1、8/1、20/15/1。各小空间穷举 rank/unrank 双射。边界包含 \(S=0\)、\(S=N/2\) 和不允许的偶奇/占据组合。 |
| G02 | 固定实对称合成模型。令 \(p,q=0,\ldots,m-1\)，\(P=0,\ldots,2m-1\)，\(h_{pq}=\delta_{pq}(-1.2+0.43p)+0.11/(1+p+q)\)，\(L_{P,pq}=0.07\cos[1+(P+1)(p+q+2)]+0.03/(1+\lvert p-q\rvert+P)\)，\(g_{pqrs}=\sum_P L_{P,pq}L_{P,rs}\)。取 G01 的 \(m=2,3,4\)。测试侧通过 determinant 空间的 \(S^2\) 和逐级 Clebsch–Gordan 耦合建立独立 oracle，明确排布与相位。 | 在相同自旋坐标中逐列 σ 和对角与 oracle 的最大差 \(\le 2\times10^{-11}\,E_h\)。最低根和小空间完整谱在 \(10^{-10}\,E_h\) 内一致，各自旋块 Hermitian 缺陷 \(\le 2\times10^{-12}\,E_h\)。 |
| G03 | 用 G02 的收敛态比较图 1-/2-RDM 和测试侧 determinant RDM。 | 元素最大差 \(\le10^{-10}\)，电子数迹、2-RDM 缩迹、能量重建、\(\langle S^2\rangle=S(S+1)\) 各差 \(\le10^{-10}\)。 |
| G04 | CAS(12,12), singlet，验证 DRT 弧计数 \(d=226512\) 和索引 \(0,\lfloor(d-1)/31\rfloor,\ldots,d-1\) 的均匀 32 个取样点 \(i_k=\lfloor k(d-1)/31\rfloor\)。CAS(8,8) singlet 用 G02 的同一模型检查一个覆盖全部分量的 σ 作用，向量为 \(\sin(1+i)\) 后归一化。 | 大空间图构建和 rank/unrank 报告静态图节点/弧数量。CAS(8,8) action 的 JAX 数组形状与工作区检查确认只依赖图/耦合项、输入输出向量及有界块，行列式全空间和 \(d_\mathrm{CSF}^2\) 矩阵只出现在独立数值 oracle 路径。用显式适用的 `solver="davidson"` 重跑小谱，CI 残差 \(\le10^{-9}\,E_h\)。 |
| G05 | 3.1 的 H2-S、H2-T、H2-R 与 H3-D 的 CASCI，沿用参考轨道分区。 | 总能量差 \(\le2\times10^{-9}\,E_h\)，\(\langle S^2\rangle\) 差 \(\le10^{-9}\)。singlet 的 H₂ 在拉伸后仍使用同一 \(S=0\) 图。 |

### A. AO2MO、非活性折叠与驻点轨道

| ID | 固定输入与独立 oracle | GREEN 条件 |
|---|---|---|
| A01 | G02 合成积分取 \(m=7\)，`core=(0,3)`，`active=(4,1,5)`，`external=(2,6)`，`enuc=0.27`。以 L 的各列因子建立同一份 ERI/packed/DF。\(S=I+0.03uu^\mathsf{T}\)、\(u_p=1/(1+p)\)。\(C=L_S^{-\mathsf T}\exp K\)，\(L_S=\operatorname{chol}(S)\)，\(K_{pq}=0.09\sin(1+p+q)\) 当 \(p>q\)，斜对称补全。用全 AO→MO 加现有 `fold_core` 作 oracle。 | active 列顺序保持不变；dense、s4、同因子 DF 的 \((h_A,g_{AAAA},E_c)\) 最大差 \(\le2\times10^{-11}\,E_h\)。分清核排斥计数。 |
| A02 | 3.1 的 LiH-C 基组与坐标，在同一 AO 模型上比较 exact MO 变换与 native selected AO2MO。DF 路径对照自己的重建 ERI。 | exact/s4/direct 的活性积分、非活性能量及混合块 paaa、pqaa（p、q 遍历当前 MO）最大差 \(\le2\times10^{-10}\,E_h\)。对 native production path 的数组形状审计确认保留的是所需块和有界工作阵列。 |
| A03 | 对 A01 中物理数组和 A02 中的 MO 系数取确定方向 \(v_i=\sin(i+1)/\|v\|\)，依照对称性投影后归一化。标量投影种子取 \(w_i=\cos(i+1)/\|w\|\)。FD 步长 \(2\times10^{-4},10^{-4}\)。 | 数组路径在 \(S,h,g,C,E_{\rm nn}\) 以及 DF 因子上 JVP/VJP 点积误差 \(\le 2\times10^{-9}\)，与独立双边 FD 的标量方向导数差 \(\le2\times10^{-6}\)。native selected 路径核对 MO C 和核坐标的活输入响应，坐标比对使用重建积分。 |
| C01 | H2-R。以 `3.1` 初始 CASCI 作为宏迭代入口。 | 显式满足 CI 无穷范数残差 \(\le10^{-9}\,E_h\)、非冗余轨道梯度无穷范数 \(\le10^{-7}\,E_h\)、两次最终接受步的总能量差 \(\le10^{-10}\,E_h\)。能量与独立 CASSCF 差 \(\le10^{-8}\,E_h\)，比固定轨道 CASCI 能量至少低 \(0.01\,E_h\)。 |
| C02 | LiH-C，Li \(1s\) 为非活性双占据轨道并参加非冗余优化。 | 与参考总能量差 \(\le2\times10^{-8}\,E_h\)，驻点指标同 C01。轨道正交误差 \(\|C^\mathsf{T}SC-I\|_\infty\le10^{-10}\)。初始 CASCI 可独立复现，原 `mf.mo_coeff/mo_occ/e_tot/scf_result` 的数值与对象所有权保持一致。 |
| C03 | H3-D，ROHF doublet 输入。 | 目标 \(\langle S^2\rangle=0.75\)，活动电子数为 (2,1)，总能量差 \(\le2\times10^{-8}\,E_h\)，CI 与轨道收敛标准同 C01。 |
| C04 | 在 A01 的 CAS(2e,3o)、nelecas=(1,1)、spin2=0 的三个分区上分别乘以固定组内旋转 \(\exp K_b\)，\(K_b\) 为本组上同 A01 构造、系数改为 0.17 的反对称矩阵。重求同一孤立根。另以 LiH-C 的起始轨道和 CASCI 根，在小模型中显式构造参考与单次轨道作用生成的 Super-CI 子空间。 | 三种组内旋转的总能量差 \(\le2\times10^{-9}\,E_h\)，回到 AO 的 1-RDM 最大差 \(\le2\times10^{-8}\)。Super-CI 模型的重叠、Hamiltonian action 和低根按同一规范与小空间显式投影比较，最大差 \(\le2\times10^{-9}\)；受限步旋转后的 \(C^\mathsf{T}SC\) 达标。 |
| C05 | 用重叠、重复、越界活动索引，错误的 \(2S/2M_S\)，人工简并的 CAS(2,2) singlet（h=g=0）、无效 `max_cycle=0` 以及 LiH-C 的 `max_cycle=1` 配额。 | 输入错误报告具体字段。未收敛、根间隙不充分及不支持的后端导数在相应输出上给出可区分的失败诊断。 |

### D. 物理响应和跨版本回归

| ID | 固定输入与独立 oracle | GREEN 条件 |
|---|---|---|
| B00 | 基线 `tests/fci/test_response.py`，记录 JAX/JAXLIB、平台、提交和命令。 | 在主验收环境 JAX/JAXLIB 0.10.0 Linux CPU float64 下通过，尤其是已有 dense 与 Davidson 的 eigenpair/RDM JVP/VJP。与 3.3 的本机失败对应，禁止用只测能量的用例代替这一闸门。 |
| D01 | H2-R 的全程导数。移动第二个 H 的 z 核坐标，用 \(h=2\times10^{-3},10^{-3}\,a_0\) 重新构造积分并重优化。对照 3.1 解析梯度和两档中心差分。 | AD JVP 与 VJP 结果差 \(\le2\times10^{-8}\,E_h/a_0\)，两者各与参考差 \(\le3\times10^{-6}\,E_h/a_0\)。两原子力和对应的平移导数 \(\le2\times10^{-7}\,E_h/a_0\)。 |
| D02 | H3-D 的允许分支，逐原子 XYZ 响应对照 3.1 中从 ROHF 初猜得到的 CASSCF 解析梯度；沿归一化方向 \(v_{I\alpha}=\sin(1+3I+\alpha)\) 取同 D01 的两档重新优化 FD。 | 逐分量最大误差 \(\le10^{-5}\,E_h/a_0\)。方向导数与 FD 各差 \(\le10^{-5}\,E_h/a_0\)，两档位移所得分支满足同一自旋和收敛条件。 |
| D03 | H2-R 同一几何方向对 AO 1-RDM 与活动 2-RDM 做状态响应。每个位移作相同参考框架的轨道对齐，再比较活动 RDM 或使用变换后的物理量。以 \(w_i=\cos(i+1)\) 归一化投影。 | VJP/JVP 点积差 \(\le2\times10^{-8}\)，与完全重优化 FD 的标量方向量差 \(\le8\times10^{-6}\)。电子迹的一阶导数 \(\le10^{-8}\)。包含只验证能量将漏掉的 CI 与 MO 松弛回归。 |
| D04 | 用 H2-R 的显式 `(S,h,g,E_nn)` 与轨道种子运行 eager、`jit`、`jit(value_and_grad)`。对每个输入按 A03 构造对称方向，分别做完全重收敛的两档 FD。种子随度规重正交后求解。每阶段复跑已有 FCI 和对应的 AO2MO 测试。 | eager/JIT 能量差 \(\le10^{-10}\,E_h\)，同一方向的能量导数差 \(\le10^{-8}\)，对每个输入的重收敛 FD 差 \(\le3\times10^{-6}\)（以归一化输入方向计）。重复输入得到一致分支。`vmap` 或 GPU 覆盖只有在相同测试实际通过的平台上列入交付说明。 |

## 5. 动态相关路线和后处理合同

后续方法读取同一 CASSCF 结果快照，包含活性定义、总自旋、根、非活性与外空间轨道、已验证的 1-/2-RDM、按需的高阶收缩入口以及明确的导数有效性。CASSCF 首版以第 4 节验收。后处理每次选择一种方法并报告所用的 \(H_0\)、冻结空间、根、能量约定和响应范围。

### 5.1 多参考 PT 调研

| 方法族 | 物理和实现要点 | 验证迁移顺序 |
|---|---|---|
| **SC-/PC-NEVPT2，NEVPT3** | Dyall \(H_0\) 与外空间激发类定义是核心。强收缩按类聚合，部分收缩保留更多自由度。高阶活性 RDM 或相应直接收缩按需提供。SC、PC 和三阶需要分别锁定工作方程。 | **第一个 MRPT 目标是 SS SC-NEVPT2**。先对 PySCF 和 OpenMolcas 对齐各激发类能量，再验总能和响应。PC-NEVPT2 以 OpenMolcas 为参考，三阶进入新的阶段。 [SC/PC 原文](https://doi.org/10.1063/1.1515317) · [PySCF 参考](https://pyscf.org/user/mrpt.html) · [OpenMolcas NEVPT2](https://molcas.gitlab.io/OpenMolcas/sphinx/tutorials/tut_nevpt2.html) · [SC-NEVPT3](https://pubmed.ncbi.nlm.nih.gov/16468852/) · [PC-NEVPT3 方程及 C₂ 对照](https://annali.unife.it/iuss/article/download/330/283) |
| **CASPT2 / CASPT3** | 广义 Fock 零阶分区、内部收缩度规、近零分母与 level/imaginary/IPEA shift 独立记录。SS 到 MS 或 XMS 引入有效 Hamiltonian、态权重及旋转。CASPT3 的方程与 CASPT2 幅度复用关系单独验证。 | SS CASPT2 对照 OpenMolcas 同一 shift；再用公开的 RS2→RS3 参考做 CASPT3 对照。固定条件后分别测近 intruder 的诊断与固定分支的 JVP/VJP。 [CASPT2 原文](https://doi.org/10.1063/1.462209) · [OpenMolcas CASPT2](https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/caspt2.html) · [CASPT3 原文](https://doi.org/10.1080/002689796173967) |
| **MRMP2 / MRMP3 与通用阶数** | Hirao 型 MRMP2 与三阶的不同分区方案各有定义。对同一已明确的 \(H_0,W,Q\)，才评估用现有 `mp.series` 的“逐阶源项 + 同一零阶 resolvent”模式扩展。 | 优先验证二阶幅度和能量分项。等 DRT、外空间投影、RDM 收缩及根响应闭合，再对已选分区推进高阶。 [Hirao MRMP2](https://doi.org/10.1016/0009-2614%2892%2985354-D) · [三阶分区参考](https://dspace.library.uu.nl/bitstream/handle/1874/727/chapter2.pdf) · [GradSCF 级数引擎](src/gradscf/mp/series.py#L351) |
| **MRCI / MRCC** | MRCI 从 CAS 参考构造外部组态或内部收缩空间，继续使用 Hamiltonian action 与本征响应。MRCC 需另定参考态、参数化和根响应；独立列方法变体。 | 先固定 CAS 参考快照接口和外空间约定，随后分别对照 OpenMolcas GUGACI 与所选 MRCC 方案。 [OpenMolcas GUGACI](https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/gugaci.html) |

### 5.2 MC-PDFT 的翻译和 AD

本地 Wiki 的 `sources/Multiconfiguration Pair-Density Functional Theory.md` 指向 `.wiki/raw/Papers-exported-PDFs-4/J. Chem. Theory Comput.-2014.pdf`。原件 SHA-256 为 `87113cb08cf6a2e82d9d1f0a14714dc549d727b923afa9f198f79fb1d826a27b`。公式取原件第 2–4 页及其 on-top translation。非单重态的原论文数值需同时检查勘误。 [原论文](https://doi.org/10.1021/ct500483t) · [PySCF on-top 实现](https://pyscf.org/_modules/pyscf/mcpdft/otfnal.html) · [ft 原文](https://doi.org/10.1021/acs.jctc.5b00609)

纯 MC-PDFT 用全体系 \(D\) 计算
\[
E_{\rm MC\text{-}PDFT}=E_{\rm nn}+\mathrm{Tr}(hD)
+\tfrac12\mathrm{Tr}(J[D]D)+E_{\rm ot}[\rho,\Pi],
\]
其中 \(\Pi\) 是**无序** on-top pair density。有序自旋求和 \(\Gamma\) 收缩到同一格点时乘 \(1/2\)。可用活动 cumulant 加全体系 \(\rho_\alpha\rho_\beta\) 重建。单行列式闭壳层极限 \(\Pi=\rho^2/4\)，单电子极限 \(\Pi=0\)。现有 GradSCF XC 点核返回每单位体积能量时，格点贡献按 \(w_g e_g\) 累加。

- \(R=4\Pi/\rho^2\)，\(\tilde\rho_\pm=\rho(1\pm\zeta)/2\)。
- **t-GGA** 用 \(\zeta=\sqrt{1-R}\) 于 \(R<1\)，其余为零。传入 GGA 的梯度特征定义为 \((1\pm\zeta)\nabla\rho/2\)。空间梯度中省去的 \(\nabla\zeta\) 项，与参数 AD 对 \(\zeta(R)\) 的导数是两件事。
- **ft-GGA** 在 \(R<0.9\) 用平方根，在 \(0.9\le R<1.15\) 用 \(Ax^5+Bx^4+Cx^3\)，\(x=R-1.15\)，其余为零。采用固定实现系数 \(A=-475.60656009\)、\(B=-379.47331922\)、\(C=-85.38149682\)。梯度特征额外加 \(\pm\rho\zeta'(R)\nabla R/2\)，\(\nabla R=4\nabla\Pi/\rho^2-2R\nabla\rho/\rho\)。接点检查值、一阶和二阶导数。
- 后续首测 `tPBE` 与 `ftPBE`。冻结同一个 grid、AO 和 CAS RDM，对照 PySCF 的 \(\rho,\Pi,R,\tilde\rho,\tilde g,E_{\rm ot}\) 和纯 MC-PDFT 总能。固定参考低密度护栏 \(\rho<10^{-15}\) 的分支，低密度取 R=1，\(\Pi=0\) 的格点取 R=0，受保护区域的空间导数为零。校验物理零密度极限。t 的 \(R=1\) 用函数值和单侧约定检查，AD 数值对照选离开接点的输入。ft 在接点做连续性和双边 FD。后处理的几何导数同时验证 AO/网格、CI 和轨道响应。

## 6. RED → GREEN 顺序

1. **基线与测试约定。** 先记录 B00；落下 G01–G04 的测试和显式 oracle。首个 GREEN 是固定总自旋的 CSF 图、矩阵自由 σ 和 RDM。
2. **接活性 Hamiltonian。** A01–A03 与 G05 先 RED，再复用 `ao2mo` 和已有 core folding 接通 CASCI。dtype、AO 来源、冻结索引及预算分别验收。
3. **轨道宏迭代。** C04 的小模型与不变量先通过，再接通 C01–C03 的 Super-CI 宏迭代和 C05 诊断。此阶段对比能量、驻点和轨道框架，OpenMolcas 交叉证据单独记录。
4. **耦合响应。** 写 D01–D04 后，以同一物理 `R(z;p)` 挂接隐式响应。最终 GREEN 包含能量与状态/CAS RDM 检查。每次改变收敛器、内存路径或编译边界，都复跑对应的数值与响应测试。
5. **后处理。** CASSCF 验收后，先按 5.1 和 5.2 的方法顺序建立各自的数值基线和 AD 验收，再启用具体接口。

计划中的测试文件按 `tests/mc/test_csf.py`、`test_casci.py`、`test_ao2mo.py`、`test_casscf.py`、`test_response.py` 落地。回归命令固定为：

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 \
  PYTHONPATH=src python -m pytest -q tests/mc tests/fci/test_response.py tests/ao2mo
```

native 阶段按仓库现有命令先构建 `PYTHONPATH=src python -m gradscf.integrals._native.build`，再运行选中的 native 案例。随结果记录源码提交、JAX/JAXLIB、设备、dtype、活性空间、求解器、阈值和测试退出状态。
