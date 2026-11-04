# SCF 分阶段性能诊断

2026-11-04。生产起点 `d8c9c183`，隔离分支 `perf/bse-tddft-grid3-20261101`。
本轮只新增诊断工具及报告，未修改生产数值内核、积分近似、网格或正式矩阵。

**本例最明确的前向成本来源是 GKS conventional packed 交换 K 收缩。**
同输入 warm K 中位数16.294s，J为2.557s，XC为1.001s，spinor对角化含前后变换为0.128s。
将独立 K 探针的工作区由64MB改为2048MB后，K为9.812s，最大数值差6.66e-16Ha。
这是局部调度收益；没有对生产 SCF 实施优化，也没有端到端加速或 AD 等价验收结论。

## 1. 资源、版本与实际执行

原矩阵在521个原始配置完成后暂停，确认无孤立日志或活动科学 worker，没有中断案例。
诊断作业115855依赖原115757结束；后继115856依赖诊断结束，三者串行。
始终只有一份 NV5090×1、64逻辑CPU、256GiB 分配在运行。

诊断分配从17:30:04至17:58:35，约28分31秒，时限30分钟。
四子进程实测墙钟合计1710.332s；此前同一工程资源内的smoke为12.221s，总计1722.554s。
原矩阵已在115856恢复。GPU仅属于分配资源，所有本次数值计算均为 CPU；不报告GPU性能。

环境为 n006 / AMD EPYC9654、Python3.11.15、JAX/jaxlib0.10.0、NumPy2.4.4。
生产源码来自 `candidate-roundoff-20261104`，四进程记录的86个相关Python源码哈希一致。
实际加载库为该目录下 `src/gradscf/integrals/_native/lib/libgradscf_integrals.so`，SHA256：

```text
ce51d885d4b65e753b5f56863470f1f1bb9c41d4b0762e47df5128e0d0b8feb8
```

主要8逻辑CPU进程实际绑定CPU0..7，分别为8个物理核；OMP/BLAS为1，libcint请求8个workers。
旧正式矩阵的CPU8 step绑定0..3和192..195，是4物理核的8个SMT线程。
因此旧709.015s与本次559.886s不属于相同资源拓扑，不能称为优化收益。
CPU64对照绑定0..31和192..223，并请求64个native workers；其SCF未在600s内返回完整结果。

所有时间均以秒记录并同步结果；MB/GB为十进制，分配和RSS保护阈值的GiB为二进制。
RSS guard为200GiB，单step预留240GiB，软件 `max_memory=180000` MB；三者含义不同。

## 2. 体系与计时协议

主体系是对苯醌阴离子 C6H4O2−，charge−1、spin1，Cartesian def2-TZVP，PBE0，grid_level3。
空间AO312、spinor维数624、电子数57，网格152104点。初猜hcore，最多200周期，
能量/密度RMS/轨道梯度门槛分别1e-10Ha、1e-8、1e-7Ha。

| 阶段 | 实际任务 | 结果 |
|---|---|---|
| GKS8 clean | 未插桩、完整conventional SCF | 25周期收敛，facade559.885869s |
| GKS8 diagnostic | 显式固定5周期、cProfile、CPU trace、准备阶段同步hooks及微核 | facade161.748993s，全部诊断约453.43s |
| UKS8 RI diagnostic | 真正weigend 2c/3c RI、完整SCF，带同步hooks/cProfile | 25周期收敛，facade85.826772s |
| GKS64 clean | 原GKS任务，仅改线程/绑核配置 | 600s墙钟超时，未得到完整SCF验收结果 |

GKS clean能量−381.2340642402313Ha，轨道梯度8.26933e-8，通过原门槛。
fixed5只覆盖5次有效advance，另有scan前的初始Fock/XC与对角化；它不是总共5次Fock。
其能量−378.9197030735884Ha、梯度1.79496，明确未收敛、`scientific_pass=false`。
原始与有效输入分别保存，没有修改正式配置或把这个探针纳入收敛/速度比较。

每个facade时间包含对象构造、积分/网格准备、首次编译和求解，排除imports、basis-dimension
preflight、证书与状态I/O。外层guard覆盖整个进程。clean没有hooks/cProfile/trace。
diagnostic同步点改变调度、融合或等待落账，不能直接与clean取比值。
微核以固定5周期返回态为输入，独立lower/compile/首次执行/warm，所有数组作为动态参数。

UKS使用RI及实数分自旋表示、流式网格；GKS使用conventional及复数spinor表示、常驻AO。
两者能量分别为−381.2336077809417和−381.2340642402313Ha，不是相同积分近似，
不将85.8/559.9或旧117/709称作隔离的GKS/UKS实现倍率。

## 3. 积分与准备阶段

以下均为诊断运行的**同步inclusive跨度**，包含相应首次编译/绑定；父子行不能相加。
例如auxiliary factors已包含2c/3c，GKS输入组装已包含ERI和网格/AO。

| 边界 | GKS conventional / s | UKS true RI / s |
|---|---:|---:|
| 输入组装总计 | 34.785332 | 8.257115 |
| `_build_common` | 34.404505 | 7.850294 |
| 基组准备 | 0.336718 | 0.340709，另有辅助基准备0.149586 |
| 网格/AO context | 2.612996 | 0.569073 |
| 其中分子网格生成 | 0.128298 | 0.120648 |
| 常驻AO及一阶导数求值 | 2.032024 | 流式，不在此处一次性求值 |
| overlap | 2.540375 | 2.514191 |
| kinetic | 0.332528 | 0.307127 |
| nuclear attraction | 0.460444 | 0.433773 |
| s4四中心ERI | 27.428933 | 不使用 |
| 辅助基factor准备 | 不使用 | 2.391897 |
| 其中2c/3c积分 | 不使用 | 1.619570 |
| RI factor unpack | 不使用 | 0.436179 |
| dipole | 0.383816 | 0.360293 |

overlap是较早触发FFI/JIT的入口，其2.54s不是纯重叠积分算术时间。
本轮不把native积分生成、分配、布局和首次编译进一步伪拆为不存在的计时。
SCF没有四指标AO→MO变换；该项明确为not_applicable，而不是测得0秒。

## 4. SCF与固定输入收缩

| 同步边界 | GKS fixed5 / s | UKS RI完整25周期 / s |
|---|---:|---:|
| solver外层总跨度 | 126.932679 | 77.548795 |
| 初始energy/Fock | 23.382934 | 5.323715 |
| 其中初始generalized JK | 19.433703 | 未以该接口测量 |
| 其中初始J-only | 2.767500 | 未独立测量 |
| 其中初始四spin K | 16.506863 | 未独立测量 |
| 其中初始XC | 3.801381 | 与流式求值融合 |
| **直接`solve_scf`跨度** | **101.902090** | **67.708556** |
| 末次raw Fock重评估 | 不另列 | 3.556829 |

两列周期数和数学算法不同，不横向计算速度比。初始评估含首次编译，不是warm。
`solve_scf`是直接同步测量，不是从facade相减推测；但它仍包含编译、迭代与等待。

### 4.1 显式编译后的微核

大项3次warm，小矩阵10次；样本完整同步。下面不是生产scan可直接相加的时间分区。

| 固定输入边界 | warm中位数 / s | 样本数 | 独立compile / s |
|---|---:|---:|---:|
| packed J-only | 2.556542 | 3 | 0.096207 |
| 四spin blocks K，生产默认调度 | **16.293958** | 3 | 0.068078 |
| K，2048MB及block_size上限16 | **9.812284** | 3 | 0.064467 |
| generalized JK | 18.831254 | 3 | 0.140095 |
| XC能量及势 | 1.000868 | 3 | 0.592599 |
| spinor正交化 | 0.002415 | 10 | 0.099340 |
| spinor对角化含前后变换 | 0.128255 | 10 | 0.036922 |
| AO DIIS commutator | 0.042497 | 10 | 0.026783 |
| 密度投影 | 0.007625 | 10 | 0.031710 |
| 占据加权MO残差 | 0.014921 | 10 | 0.047153 |
| 完整energy/Fock | **19.751787** | 3 | 0.657524 |

11个实测微核全部完成目标warm数，12项数值一致性检查通过。
完整energy/Fock相对fixed5返回态：E、Exc、Fock最大差均为0；分项重构Fock最大差3.55e-15Ha。
真正DIIS历史外推未独立测量，因为返回态不保留ring history；没有捏造代表性历史。
已测的是其误差commutator。实际历史外推及小矩阵求解仍包含在scan跨度中。

### 4.2 已确认的调度限制

[`generalized_jk`](../../../src/gradscf/scf/gks.py) 没把facade预算传入公共packed contractor。
[`build_jk_from_packed`](../../../src/gradscf/integrals/molecular/jk.py) 因而仍用64MB额外工作区。
该接口的工作区预算排除了ERI、密度与调用者其他数据，不能盲目把总预算180000MB原样转发。

本例s4为48828×48828，共2,384,173,584元素；CPU native streaming目前接收s8，
s4走portable(1)。元素数超过2^31，J从小buffer的直接einsum切为分块gather；
该保护分支从304AO触发，但本轮没有303/304相邻点对照，不能量化所谓性能悬崖。
native s8还受dtype及workspace判定约束；本例四spin complex K的native工作区估计约62.3MB，
超过默认64MB预算保留20%后的可用量，因此只切换为s8也不保证激活native路径。

默认K实际`p=1,q=9`，约10920个tile；2048MB对照实际`p=1,q=312`，约312个tile。
所以不能把收益称为“16行批量”：16只是上限，实际p没有增大。
同ERI、同密度、同线程与精度下，K由16.293958降至9.812284s，减少6.481674s、约39.78%，
仅该微核的时间比为0.602204（1.6606×）。最大差6.66e-16Ha、RMS9.44e-18Ha。
未改变基组、积分近似、轨道窗口或丢弃spin块。

## 5. 编译、trace与内存

| 诊断 | 精确backend_compile_and_load Python调用数 | self / s | cumulative / s |
|---|---:|---:|---:|
| GKS fixed5 | 278 | 5.553344 | 5.553547 |
| UKS RI | 298 | 6.035976 | 6.036219 |

这是该函数调用次数，不是独立executable数，也不包括所有tracing/lowering成本。
GKS cProfile中`synchronize` self约121.98s，是异步执行等待的落账位置，不能叫同步器算法瓶颈。
后续warm收缩仍需16s，而独立K编译仅约0.07s，证据支持先优化执行调度。

Chrome trace有100,516条带module/op/run/device标识的运行期HLO候选；但恰好1,000,000条X事件，
所有新事件只覆盖到trace约59.09s，SCF scan约59.76s才开始，且没有`solve_scf` annotation或
`jit_scan` HLO事件。文件可正常解析，疑似事件上限截断，不能认为HLO覆盖了完整161.75s。
长父span结束时间不等于后续事件被记录。本次**不能给scan内JK/XC/DIIS运行期百分比**。
原始操作统计仅适用于已覆盖的准备/初始评估；xplane是否保留更多事件未另行解码。
CPU线程事件inclusive时长不是wall-time分区；父子、线程、module不可直接相加。

| 对象或指标 | bytes | 含义 |
|---|---:|---|
| GKS s4 ERI | 19,073,388,672 | 单个积分数组逻辑容量 |
| 常驻AO values | 379,651,584 | 152104×312 float64 |
| 常驻AO deriv1 | 1,518,606,336 | 4×152104×312 float64，含value分量 |
| UKS RI factors | 417,411,072 | 536×312×312 float64；不等价于conventional积分 |
| GKS clean进程峰值RSS | 25,458,413,568 | ru_maxrss，约25.46GB |
| GKS诊断含全部微核峰值RSS | 28,710,629,376 | 含profiler、缓存与延长inputs生命周期 |
| UKS RI诊断峰值RSS | 3,481,735,168 | 其自身算法和流式布局 |
| 默认K compiler temporary | 14,020,104 | buffer accounting，不是RSS |
| 调度对照K compiler temporary | 485,961,352 | 更快的工作区有内存代价 |
| XC compiler temporary | 2,658,777,944 | 内核分析值，不是进程独立峰值 |

UKS resident AO点维度为0代表流式网格，权重仍有152104点，不是省略了DFT网格。
diagnostic捕获inputs供micro复用，延长ERI生命周期，但不深拷贝或落盘19GB张量。
每个独立进程有0.2s RSS sampler和ru_maxrss；短峰值可能漏采。没有每微核的独立RSS峰值。
CPU64在600s触发墙钟保护、峰值约26.34GB，未发生记录中的OOM；无终态、周期或可接受速度比。
不能凭它断定加核普遍更慢，也不能承诺更多线程会解决当前调度问题。

## 6. 按证据排序的后续优化

| 优先级 | 最小改动方向 | 本轮证据与验证要求 |
|---|---|---|
| 1 | 贯通消费者的总预算、常驻数据预留与packed J/K工作区；测试有界CPU调度 | 同输入K减少39.78%；先做完整收敛SCF与原算子比较，再跑非对称/batch/complex、JVP/VJP和已有几何/NN回归。不能直接把全部总预算当工作区，也不外推端到端倍率 |
| 2 | 复用现有native s8能力，或让CPU消费者直接有效消费s4；避免大量小gather | 当前s4没有走native streaming，K有10920个tile；最小实验应对齐同一ERI/density并独立计转换成本，保留pair权重、spin-offdiagonal与积分/密度高阶AD。不是新增生产积分器或改为RI |
| 3 | 结合AO规模、tile和实际物理核拓扑管理native/JAX/BLAS线程 | 8物理核clean已完成，64逻辑CPU对照未在600s内完成；下一步先固定准备数据分离J/K线程收益，区分收敛轨迹、native线程与JAX pool，不做无界调参 |
| 4 | 再处理重复准备、编译固定成本和属性按需构建 | 冷编译入口约5.55s，dipole约0.38s，优先级低于16.29s热K；同签名动态输入/几何/模型参数更新必须正确失效，不能缓存旧收敛态 |

目前没有生产优化收益验收；没有在收敛态上复测调度对照或测试其AD。
未对LDA/GGA/RSH、neural XC、GPU、核梯度或训练步做本轮性能推广。
没有把本次前向数据用来否定AD或宣称调用libcint便能解决全部问题。

## 7. 文件与复跑

- [run_plan.json](run_plan.json)、[phase_results.json](phase_results.json)：预算及实际状态。
- [cases/](cases/)：完整几何/基组/XC/阈值配置。
- [stage_measurements.csv](stage_measurements.csv)、[kernel_measurements.csv](kernel_measurements.csv)：原始逐边界/逐微核汇总。
- [compile_counts.json](compile_counts.json)、[environment.json](environment.json)：编译调用语义、加载路径、源码及原始文件SHA256。
- 本地 `raw/`（Git忽略）保留逐条events、samples、pstats及Chrome gzip；155MB xplane保留远端。
- 权威原始路径：`jy_gpu_2:~/private/homefile/gradscf-scf-solver-20261102/results/stage-profile-115855/`。
- [commands.txt](commands.txt) 给出独立clean、诊断与纯元数据分析命令。

本次三项诊断/完整SCF成功完成，条件CPU64测试超时；这不替代60体系验收。
