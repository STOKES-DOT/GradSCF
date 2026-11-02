# 乙烷 C–C 拉伸：SCF 分支修正后的九曲线对照

## 修正原因

旧版采用的 hcore 初猜可收敛到较高能、但内部稳定的受限 HF 解。内部稳定性不保证全局最低能量。
在 2.75 Å，同一冻结 MACE 基组的原生能量从 −78.737672894991 Eh 降至
−79.002031128907 Eh，差 0.264358233916 Eh（165.887296 kcal/mol）；两个解均通过内部稳定性检查。
因此旧图的突跳不能用于判断网络表达能力或基组极限。

原始数据与第一版图仍在各自目录及 `archive-hcore-only/` 中保留。本目录当前图表采用修正后的数据。

## 方法与实际完成范围

- 同一批 56 个刚性几何，C–C 距离 1.525976–8 Å；中性闭壳层、球谐 AO、float64、
  DF-RHF / def2-universal-jkfit。甲基内部几何不松弛，基组指数与内层固定。
- 冻结 MACE update 3351，原始收缩系数不变；全部 56 点重新比较 hcore 与中性原子布居初猜的 SCF 解。
- 自由收缩在 2.70–8 Å 的 **32 点重新优化**，采用相邻已收敛系数及原归档系数两个起点。
  其余 **24 点重新核对能量、稳定性和归一化系数梯度后沿用**。
- 内层系数不参与优化。自由参数作可逆 primitive 重叠度量预处理，不改变基组空间或能量目标。
  内部优化仍使用现有 **implicit SCF** 和积分收缩 VJP；初猜密度视为算法输入，不对它求物理响应。
- 每个优化起点及终点比较 hcore/原子布居解，在满足 SCF 与内部稳定性要求的解中选较低者。
  单个内层优化步只执行选定初猜下的一次融合 SCF forward/backward；没有有限差分。
- 传统基组为 7×56=**392 个几何–基组组合**。比较归档 hcore、MINAO、前一点投影密度、
  中性原子布居初猜；102 点比旧版获得更低的能量。每点可能执行多次 SCF，不能将 392 当作实际 SCF 调用次数。

## 五个原未收敛点

系数梯度在每个可训练径向函数的原始系数归一到欧氏范数 1 后检查，接受阈值仍为 1e-5。
受控 callback 可在达到梯度目标后终止 SciPy；日志中的该终止状态不等于物理终点未收敛。

| C–C / Å | 修正后 MACE 能量 / Eh | 自由收缩能量 / Eh | 自由收缩梯度范数 |
|---|---:|---:|---:|
| 2.70 | −79.010218786 | −79.023228751 | 4.40e-6 |
| 2.75 | −79.002031129 | −79.015858289 | 4.62e-6 |
| 2.80 | −78.994009028 | −79.008735279 | 2.71e-6 |
| 2.85 | −78.986167264 | −79.001857522 | 4.60e-6 |
| 2.90 | −78.978527370 | −78.995221868 | 3.72e-6 |

五个点另完成未预处理参数的双起点优化，均达到相同物理梯度标准；两起点最大能量差为 3.01e-11 Eh。
这支持当前局部极小解的一致性，不能证明收缩系数的全局变分极限。

## 全范围结果

相对能量分别使用每条曲线自己的 E(R₀)，R₀=1.525976376756 Å；绝对能量图不作平移。

| 基组／模式 | AO 数 | 有效点数 | E(8 Å)−E(R₀) / kcal mol⁻¹ |
|---|---:|---:|---:|
| STO-3G | 16 | 56 | 304.973 |
| STO-6G | 16 | 56 | 303.108 |
| STO-6G + C(d), H(p) | 44 | 56 | 301.942 |
| 3-21G | 30 | 56 | 243.878 |
| 6-31G(d,p) | 58 | 56 | 241.128 |
| cc-pVDZ | 58 | 56 | 235.847 |
| cc-pVTZ | 144 | 56 | 231.921 |
| NNAO / MACE | 44 | 56 | 256.854 |
| NNAO / 自由收缩 | 44 | 56 | 245.392 |

自定义 STO-6G 极化层沿用此前固定设置：C 的 d 指数 0.8、H 的 p 指数 1.1 Bohr⁻²，
原始系数均为 1；不是对唯一标准化 STO-6G** 的命名主张。

## 验证与计算环境

- 新版共 **504/504 个有效能量点**，已取消原五点缺口，没有插值或平滑修补。
- 原生 NNAO 的 56×2=**112/112 个冻结基组能量**通过独立 CPU PySCF 对照，
  最大差 1.137e-12 Eh。独立对照中一个候选初猜触发 PySCF DIIS/LAPACK 异常，已单列；
  同一点其他初猜得到有效、内部稳定的对照解，没有跳过任何最终案例。
- 最终自由收缩梯度均小于 1e-5；32 个新优化点的最大值为 4.874e-6。
- 24 项相关 CPU 回归测试通过；4 项不在本轮范围的真实 MACE/有限差分测试未运行。
- 显式初猜入口本次只作了 CPU 回归，未对 GPU 后端作专项验证。
- 本地 macOS arm64、12 个逻辑 CPU；GradSCF/JAX 0.8.1 CPU 后端和原生积分器；
  独立参考为 PySCF 2.9.0。仅缓存 S、h 与 packed DF 因子，没有完整四指标 ERI。
- 原生积分线程上限为 4；PySCF 的 BLAS 线程环境请求上限为 4，该构建没有 OpenMP。
  32 点优化脚本墙钟为 1123.34 秒，含准备/编译/端点审查，并与其他本地复核进程并行，
  不构成与旧 V100 计算的公平速度对比。峰值内存未独立测量。

这是受限 RHF 压力测试，仍不代表物理均裂解离极限或真实键解离能。
多初猜、内部稳定性和两个系数起点均不保证严格全局最低解；cc-pVTZ 也不是精确解。

## 文件和复现

- `ethane-cc-critical-repaired.png/.pdf`：五点局部能量与重算梯度。
- `ethane-cc-relative.png/.pdf`、`ethane-cc-absolute.png/.pdf`：九条完整曲线。
- `carbon-radial-functions-repaired.png/.pdf`、`carbon-radial-response-repaired.png/.pdf`：
  用修正后的全部 56 点重做的径向函数与响应图，共 224 个归一化径向函数。
- `curves.csv`、`summary.csv`：全部点与 AO/能量汇总；`nnao-results.json`：新版原生结果。
- `traditional-results.json`、`basis-library.json`：多初猜参考与冻结传统基组。
- `provenance.json`、`verification.json`：数据哈希、数值与绘图核对。
- `../nnao-ethane-dissociation-repair-20261022/`：全部修正驱动、轨迹、32 个优化点、
  56 个归档点复核、独立对照及最终冻结基组；旧扫描原始数值保持不变。

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python artifacts/nnao-ethane-dissociation-repair-20261022/repair_scan.py
python artifacts/nnao-ethane-dissociation-repair-20261022/publish.py
python artifacts/nnao-ethane-dissociation-benchmark-20261022/analyze.py
python artifacts/nnao-ethane-dissociation-repair-20261022/render_critical.py
python artifacts/nnao-ethane-dissociation-benchmark-20261022/verify.py
```

数值重算依赖相应原生库和已经归档的候选池、系数及参考结果；`run.py` 和
`archive-hcore-only/` 是旧 hcore-only 协议，不用于覆盖新版多初猜结果。
