# GradSCF libcint / NNAO / AD 对称性优化交接

本轮已完成原 libcint 单生产接口计划及追加的 S9 导数对称性工作，统一按有界科学验收收口。工作树为 `/Users/jiaoyuan/.codex/worktrees/dcdb/GradSCF`，分支 `refactor/libcint-nnao-perf-20261026`。起点是原 HEAD `70aea47` 加当时必要的81个 dirty/untracked 源码与配置文件，保存为独立快照提交 `7d2e859`；未用旧审阅 checkout 替代当前源码。后续代码、测试、探针按独立提交保留，`commits.txt` 列出记录文档前的提交链。

生产 Gaussian 一电子、Coulomb 四中心及显式辅助基 RI 统一走 libcint。六个纯 JAX Gaussian 引擎文件及其生产 dispatch 已删除，旧 `cpu/native/jax/jax_reference/pyscf` selector 不作为别名。普通 HF/DFT 构造器不再选择积分引擎。JAX 仍负责归一化、投影、网格 AO、XC、数组代数与响应；周期 FFT/GTH 保留。必要的 PySCF 填充/ECP C 源码有单独来源和许可证，不是 Python runtime fallback。下载的 libcint6.1.3 上游归档与45个使用的 vendor 文件逐字核验，未修改 vendor。

核心改造的计时在 n005/Xeon6430，追加对称性前后比较在 n006/EPYC9654。Python3.11.15、JAX/JAXlib0.10.0、PySCF2.13.0，物理计算为 float64。S9 使用一张 RTX5090、最多64逻辑 CPU、512GiB 分配；GPU模型和数组运算配合 CPU libcint。正式 CPU1/CPU8 配对保持同硬件、输入、输出与收敛条件；宽方法验证前后分别CPU1/CPU8，禁止由其诊断墙钟计算速度比。所有量化计算在远端 Slurm 中完成，本地仅检查源码和解析已有数组。

水／Cartesian def2-SVP 为25AO、5占据、20虚轨道，PBE网格21952点。核心新对象 warm 中位数：PBE3.368→0.319s，CCSD6.418→0.250s；显式 Weigend-RI GW→TDA/full-BSE114.869→57.685s。普通四中心谱因子 GW 的旧6000s作业完成了两个结果后超时，新版四次均完成；保留有效数值对照，但不为不完整旧计时生成正式速度比。RI与四中心谱压缩始终分列。

S9 的系数一阶保留四个导数位置，二阶保留十二个有序交叉项；VJP先累加窗口内所有逆置换 cotangent，再逐自由腿归约。重复 shell 的发射去重与导数重数分开，一般收缩的 primitive/contracted 维度分开。几何核携带导数腿和分量，保留四个同腿、两个同电子对、四个跨电子对 Hessian 模式。调度只保存二次方 shell-pair 元数据，scratch缓存每driver最多4096条，不缓存数值积分或收敛态。

同输入 CPU1 微内核：系数 VJP0.1283→0.0272s，系数二阶VJP0.3756→0.0653s，指数VJP0.2825→0.0513s，核坐标二阶VJP0.4034→0.2482s。窄窗口回退52.4→61.6μs，也保留在表中。16对微内核输入逐位相同，结果最大差7.105e-13。另有12个公共 PBE0/TDA/TDDFT 进程的48次结果，实际基组、Cartesian表示、网格、五个态及收敛状态匹配；相对PySCF激发能误差约1.26e-8eV。

阅读计时时先筛选计算口径。微内核首次时间是降低、编译和首次执行，载库与准备另列；正式工作流从几何与新方法对象开始，含所请求的求解和性质。两类时间不能直接相除。每个配对都有三个同步完成的warm原样本，未复用最终能量或收敛态。统一定义time_ratio为GradSCF除以参考程序，大于一表示更慢；after/before是另一列。参考不足、超时或物理近似不同的行不补速度比。软件内存预算也不等于进程硬限制，实际RSS与采样频率另存，编译峰值和warm峰值不能混称。

验证覆盖11个代表方法组，前后各12案例均成功，包括SCF、R/U响应、CI/FCI、MP2/RI-MP2、递归MP4、CC/(T)/EOM、GW/BSE、OFDFT、NNAO及周期对照。各组包含不同的完整几何响应或固定输入偏导，不能解释为所有方法的全套测试。新产品41项、既有高阶/packed/direct/range128项、最终公共与复用44项分别通过，不相加夸大总覆盖。N₂严格π简并的MP2完整梯度/HVP在CPU/5090一致。

干净进程曾暴露 bounded几何AD首次调用未注册handler的问题，诊断预加载会掩盖它。12个新进程用例先失败，修复后与packed回归合计30项通过；最终无PySCF安装环境的12个首次调用再通过。最终安装的486个Python/C++/基组资源文件逐项匹配当前源码，模块和实际库均位于隔离前缀。

NNAO保留c与log-exponent能量训练、归一化、mask、优化器与checkpoint语义。四组CPU/GPU前后运行均完成四步及最终验证；直接读NPZ并安全解析optimizer验证count=4、冻结/闲置参数与梯度，CPU前后逐位一致，GPU差约1e-15。这里是seed3、小型真实MACE架构的两个H₂随机fixture，未提供历史训练权重；未验证完整7171态训练构造器或真实活动恢复。当前目标为重收敛RHF能量均值，不是FCI监督精度或力损失。

缓存分为编译程序与数值输入，两者验收不同。改变指数、几何或辅助基参数会改变固定积分签名；仅改变收缩系数可复用真正独立于系数的primitive数据，实际积分产品仍重新求值。单个零系数的梯度没有被前向筛选裁掉，整列非法归一化与接近线性相关状态仍应拒绝。当前训练检查重叠最小本征值及收敛、稳定性和能量重建误差。不要把固定辅助metric的系数训练解释成几何微扰中辅助中心不动，也不要为了缓存命中冻结外部模型输出。

剩余能力边界没有隐瞒：RI几何/辅助参数、指数Hessian、NNAO基组与几何混合力损失仍不支持；ECP保留固定primitive系数路径，contracted变指数RI/ECP明确拒绝。TDA新对象仍有性能差距：独立profile第二次forward有4次backend编译累计4.982s，response累计6.872s，reference准备约0.051s；这些嵌套时间不相加，也不是backward成本。后续应先功能化动态算符payload，避免冻结分子或盲目套外层JIT。MOLGW上游CH₄ scissor小例有限且正常结束，但与水的非线性GW任务不匹配，旧water/restart NaN未在此轮闭环，不报其速度比。

统一入口为 `implementation_plan.md`、`implementation_report.md`、`migration_matrix.md` 和 `before_after.csv`，后者56行；历史表的字节副本为 `phase1_before_after.csv`。`s9_results.json` 保存348项原始artifact哈希，原始数据归档在 `evidence/s9/`。复跑使用 `tools/libcint_migration/products.py`、`worker.py`、`probes.py` 与 `run_cases.py`；诊断计数另建 `--diagnostics` 库。先读最新集群手册并申请新分配，不复用已结束的109318。

原工作区 HEAD/index、81个迁移文件及1901个起始源码条目均未变；共享Git对象新增属于隔离分支的正常提交。109318已正常COMPLETED、exit0:0，实际03:46:59后释放。当前只做本地提交，未合并或推送。
