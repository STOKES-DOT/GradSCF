# CPU 积分准备与 GPU GKS 试跑

2026-11-04。隔离工作树1b2c，分支 `perf/bse-tddft-grid3-20261101`，起点0d344b92。
生产源码未修改；只新增显式staging诊断工具与本报告。

**312 AO 的 conventional GKS 在 RTX 5090 上完成并通过与CPU的数值一致性验收。**
单次脚本入口到SCF验收为194.525s，历史CPU观察为561.443s，GPU/CPU耗时比0.34647。
这是一组同配置、同主CPU affinity、同节点上的单次观察，非三次重复的统计性能结论。

## 1. 执行与资源

复用jy_gpu_2的Slurm115856，一张NV5090、64逻辑CPU、256GiB，没有申请第二张GPU或超出CPU额度。
大矩阵父controller在当前worker结束后暂停；GPU主测量独占该分配，结束后自动SIGCONT恢复。
仅很小的CUDA健康检查曾与 `phenanthrene-def2-tzvp-uks-pbe-gradscf` 重叠；该CPU案例已追加
计时排除，保留科学结果，不生成速度比。原始日志、配置和测量不覆盖。

首次launcher从home目录提交，被站点homefile规则拒绝，未执行数值；修正工作目录后另开r2输出。
失败日志仍在远端 `results/gpu-profile-20261104/`，不与成功r2记录混合。

实际GPU为NVIDIA GeForce RTX5090，Slurm可见CUDA0。健康检查PID对应GPU UUID与大例preflight一致：
`GPU-aff7b45c-2ef8-cb7a-3c39-c899769851c3`。健康检查真实运行float64 matmul和complex128 eigh，
验证输出设备、精度和数值，并逐TID检查无affinity越界。不是仅用nvidia-smi判断GPU可用。

JAX/jaxlib及CUDA12 plugin/PJRT均0.10.0，NumPy2.4.4，SciPy1.17.1，Python3.11.15。
主要进程affinity0..7，即8个不同物理核，native积分workers8，OMP/BLAS1；其余预留CPU没有当作
“CPU64测试”报告。设置 `JAX_PLATFORMS=cuda,cpu`、x64、PREALLOCATE=false、MEM_FRACTION=0.9。
launcher使用进程局部 `HWLOC_THISSYSTEM=0` 避免历史NUMA重绑故障；未修改系统配置或Slurm限制。

## 2. 数据路径与物理一致性

```text
CPU libcint生成积分、CPU计算grid/AO
  → 同步准备结果并检查CPU驻留
  → 显式上传一次七个输入数组
  → 输入驻留GPU，调用生产run_gks_from_integrals
  → 返回少量能量/梯度标量到host验收，矩阵仍在GPU
```

Gaussian积分没有在GPU上重算，也没有把CPU回退标为GPU执行。准备在`jax.default_device(cpu)`内，
后续数值在`jax.default_device(gpu)`内，检查全部矩阵输出及输入的实际设备与dtype。
没有stop_gradient、轨道窗口截断、精度降为float32或新的SCF循环。

主体系：对苯醌阴离子C6H4O2−、charge−1、spin1、Cartesian def2-TZVP、PBE0、grid3、
312空间AO、624spinor MO、152104网格点。conventional s4，不使用辅助基RI。
GPU配置与CPU参考input逐字段一致，86个相关生产Python文件和native SHA完全相同。
两侧均hcore初猜；GPU未载入收敛轨道。native SHA256为：

```text
ce51d885d4b65e753b5f56863470f1f1bb9c41d4b0762e47df5128e0d0b8feb8
```

最多200周期，能量/密度RMS/轨道梯度门槛为1e-10Ha/1e-8/1e-7Ha，CPU配对能差门槛1e-7Ha。

| 指标 | CPU参考 | GPU试跑 |
|---|---:|---:|
| 有效SCF周期 | 25 | 25 |
| 总能量 / Ha | −381.2340642402313 | −381.23406424024995 |
| 轨道梯度 / Ha | 8.26933168e-8 | 8.26932459e-8 |
| 收敛验收 | 通过 | 通过 |

绝对能差1.86446e-11Ha，远低于门槛。CPU参考summary文件SHA已与本地保存记录重新计算核对。
周期数相同是本次观察，不是GPU验收必须条件。

## 3. 分开计时边界

| 边界 | GPU时间 / s |
|---|---:|
| 启动、导入、设备初始化及准备前元数据合计 | 51.282755 |
| CPU几何/基组/积分/网格准备 | 31.655688 |
| 显式上传约20.974GB | 1.027559 |
| GPU生产SCF，包含首次编译及同步 | 110.489307 |
| CPU准备开始至GPU结果验收，含阶段间开销 | **143.241947** |
| 脚本入口至SCF验收，包含启动 | **194.524702** |

CPU参考的脚本入口至SCF验收为561.442629s，故该单次冷观察的GPU/CPU时间比为0.34647298，
反比约2.886倍。CPU facade559.885869s只含设备已初始化后的构造/准备/求解。
**不能将CPU facade559.9除以GPU solver110.5作为端到端加速。**
前51.28s没有进一步分离为纯GPU初始化；不能全部归因于CUDA。

两次观察时间不同、节点与其他用户共享、进程局部NUMA策略也已披露；不是同时独占整节点的
重复统计基准。双方主亲和性0..7相同；旧正式矩阵4物理核8SMT的709s不作为本轮分母。

### 3.1 收敛态上的GPU微核

| GPU固定输入微核 | lower / s | compile / s | 首执行 / s | 3次warm中位数 / s |
|---|---:|---:|---:|---:|
| J-only | 0.016650 | 4.407605 | 0.832262 | **0.817913** |
| 四spin blocks K-only | 0.019115 | 0.168464 | 2.434141 | **2.429865** |
| XC能量与势 | 0.072609 | 1.338292 | 0.140376 | **0.139018** |
| eig及前后变换 | 0.008427 | 0.114278 | 0.026273 | **0.025653** |

数组动态传参，计时同步完整输出，核验输出驻留CUDA0且有限；此表只报告GPU绝对计时。
此前CPU微核使用未收敛fixed5快照，本次使用GPU收敛快照，**不计算严格共享输入的CPU/GPU微核速度比**。
独立JIT边界不能相加当作整轮SCF的精确分区。微核的finite检查也不替代逐元素CPU oracle；
独立数值证据由下节小体系/AD测试及大例完整能量/残差验收提供。

## 4. 显存、主机内存与传输

| 指标 | bytes | 口径 |
|---|---:|---|
| 一次显式上传的输入 | 20,974,420,936 | 七个数组的逻辑容量 |
| 其中s4 ERI | 19,073,388,672 | 输入矩阵48828×48828 float64 |
| JAX活动分配峰值 | 24,488,574,208 | allocator `peak_bytes_in_use`，约22.807GiB |
| JAX分配池峰值 | 30,311,998,208 | `peak_pool_bytes`，约28.230GiB |
| 结束时JAX活动分配 | 21,002,466,816 | 输入及返回态仍存活 |
| 主机进程RSS峰值 | 24,649,781,248 | ru_maxrss，约22.957GiB |

不能把JAX活动分配峰值叫作整个GPU物理占用峰值：另有内存池空闲保留和CUDA上下文等。
`bytes_reserved=0`也不代表没有分配池。没有采集大例NVML进程显存峰值序列，不以0填补。
0.5s采样的JAX内存计数与整个进程RSS记录保留在原始JSONL中。

上传前同时检查物理free和JAX pool剩余额度；预估约27.12GB，小于当时可用约30.31GB。
这是保守预检，不是任意shape的OOM保证。CPU报告的临时空间没有当作GPU空间上限。
诊断保留CPU输入引用，主机内存与显存分别占用，不能宣称整体内存下降。
本例已使用较多显存，更大basis/conventional积分仍受O(N^4)存储约束。

## 5. 数值与AD验收范围

- CUDA健康检查：实际float64矩阵乘法和complex128 Hermitian eigh通过。
- CH4/STO-3G/PBE/grid3、9AO staged GKS：CPU/GPU均10周期，能差2.84217e-14Ha，科学通过。
- 312AO/PBE0大例：完整hybrid GKS能量、收敛与轨道梯度通过，见第2节。
- 8个GPU参数化array AD case全部通过，0skip、0fail，约57.09s。

8个case覆盖GPU×s4/s8×J+K/J-only/K-only/neither。5AO、seed2781、(2,3,5,5)随机非对称complex密度，
对照独立dense oracle验证值、联合ERI/密度JVP、两者的标量loss反向梯度。
值/JVP atol/rtol3e-11；reverse atol3e-9、rtol3e-11。
**未覆盖HVP、几何导数、高阶导数、NN训练、GKS/SCF反传或312AO大索引AD。**
CPU准备在本轮为前向staging步骤，不能据此宣布完整几何/NNAO的GPU backward已打通。

## 6. 结论与下一步

本例显示GPU值得使用：在不降低精度、不改积分近似/轨道空间的条件下，完整目标能量通过，
单次冷观察较历史CPU更快。初始化和CPU积分准备仍是实际成本；GPU并没有替换libcint积分生成。

本次是显式CPU准备→GPU核心求解的诊断路径，没有修改高层默认值或将全部facade自动改为GPU。
后续应优先统一CPU准备/显式搬运/设备驻留的生产边界，并把工作区预算按实际显存与常驻数据扣减，
验证GKS几何/NN所需后向能力，再拓展更多方法与体系。不能直接将CPU总预算180GB当作GPU工作区。
GPU仍沿用默认64MB packed workspace，312AO四spin K的实际p1/q9没有被本轮调大；调度优化仍有待测。

原60体系矩阵已在115856恢复；本轮不替代其全量验收，也未处理已有开壳层驻点能量差异。

## 7. 证据与复跑

权威原始目录：`jy_gpu_2:~/private/homefile/gradscf-scf-solver-20261102/results/gpu-profile-20261104-r2/`。
本地 `raw/`（Git忽略）有逐阶段日志、显存、Slurm guard、JUnit和完整summary/runtime。
[environment.json](environment.json) 保留源码/native/参考及原始文件SHA，
[validation.json](validation.json) 保留测试节点与small验收，full验收见[summary.json](summary.json)，
[gpu_kernel_measurements.csv](gpu_kernel_measurements.csv) 保留每个微核的样本范围。
[cases/](cases/) 与 [commands.txt](commands.txt) 给出输入与命令。
