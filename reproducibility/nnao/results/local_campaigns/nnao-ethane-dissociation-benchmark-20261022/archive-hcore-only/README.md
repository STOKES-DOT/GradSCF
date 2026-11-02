# 乙烷 C–C 拉伸：NNAO 与传统基组对照

## 计算方法与来源

沿用 `nnao-ethane-dissociation-20261019` 的全部 56 个刚性几何，C–C 距离
1.525976–8 Å；甲基内部结构不松弛。所有曲线均为中性、闭壳层、球谐 AO 的
DF-RHF / def2-universal-jkfit，float64。

- NNAO 两条曲线复用原 MACE **update 3351** 及同池自由收缩结果，未重新推理或优化。
  指数与内层固定；自由收缩由 MACE/参考系数两个起点独立优化。它是有限预算结果，
  不等于严格全局变分极限。
- 传统基组新增 **7×56=392** 个本地 CPU PySCF 2.9.0 计算，全部通过收敛及内部稳定性检查。
  C/H 基组原样复用此前扭转比较的冻结基组库；没有重新选择 primitive 或极化指数。
- 自定义 `STO-6G + C(d), H(p)` 使用 C 的 d 指数 0.8、H 的 p 指数 1.1 Bohr⁻²，
  原始系数均为 1；它不代表唯一标准化的 STO-6G** 基组。
- 新计算使用初始 hcore 密度；必要时采用 0.2 Eh level shift 后回到零 shift，
  再分析受限内部稳定性。能量阈值 1e-12 Eh，轨道梯度阈值 1e-9，最多 200 SCF 步；
  AO 对易子范数须小于 1e-7。只使用 DF，不构造完整四指标 ERI。

## 数值结果

| 基组／模式 | AO 数 | 有效点数 | E(R₀) / Eh | E(8 Å)−E(R₀) / kcal mol⁻¹ |
|---|---:|---:|---:|---:|
| STO-3G | 16 | 56 | −78.306060095 | 304.973 |
| STO-6G | 16 | 56 | −79.065274925 | 303.108 |
| STO-6G + C(d), H(p) | 44 | 56 | −79.107642302 | 301.942 |
| 3-21G | 30 | 56 | −78.793497403 | 243.878 |
| 6-31G(d,p) | 58 | 56 | −79.237948410 | 241.128 |
| cc-pVDZ | 58 | 56 | −79.234952355 | 235.847 |
| cc-pVTZ | 144 | 56 | −79.259695973 | 231.921 |
| NNAO / MACE | 44 | 56 | −79.216446507 | 256.939 |
| NNAO / 自由收缩 | 44 | 51 | −79.216513372 | 245.574 |

R₀=1.525976376756 Å；相对能量分别减去各自的 E(R₀)。绝对能量图不作平移。
新增传统计算的最大 AO 对易子范数为 9.366e-9；初始几何处与旧 GPU 计算的最大能量差
为 1.223e-12 Eh。原 NNAO 在 8 Å 处的 MACE−自由收缩差为 11.406679 kcal/mol。

## 解释边界与绘图处理

这是受限 RHF **压力测试**，不是物理均裂解离曲线，也不能将表中的 8 Å 拉伸能称为真实键解离能。
各点为局部稳定受限 SCF 解，内部稳定不保证全局最低解或距离扫描中的分支连续性。
原 NNAO 在约 2.7 Å 附近的突变，以及传统基组出现的分支变化，均保留原值；没有平滑修补。

自由收缩的 2.70、2.75、2.80、2.85、2.90 Å 五个点未达到系数梯度标准，因此曲线断开；
这些点的 MACE SCF 有效，仍保留。CSV 同时保存原始能量与 `accepted`/`status`；
`plot_energy_hartree` 和相对能量列在未接受点留空。灰色带只标记上述自由收缩失败区间。
cc-pVTZ 是比较基组，不能作为精确解；相同 AO 数也不意味着 primitive 数、内层或计算成本相同。

## 本次补算与复现

硬件：本地 macOS arm64，12 个逻辑 CPU；未申请额外 GPU。BLAS 线程环境上限请求为 4；
该 PySCF 构建不含 OpenMP，OpenMP 的 4 线程请求不生效，不能宣称使用了 4 个 OpenMP 线程。
392 个比较点累计脚本耗时 **105.49 秒**，包含可复用的 21 个短验证点；未单独测量峰值内存。
原 NNAO 数据来自 V100/32 CPU 的历史计算，运行时间不能与本次 CPU 补算直接比较。

```bash
/opt/anaconda3/bin/python artifacts/nnao-ethane-dissociation-benchmark-20261022/run.py
/opt/anaconda3/bin/python artifacts/nnao-ethane-dissociation-benchmark-20261022/analyze.py
/opt/anaconda3/bin/python artifacts/nnao-ethane-dissociation-benchmark-20261022/verify.py
```

- `ethane-cc-relative.png/.pdf`：各自初始能量为基准，左图全部九条，右图极化基组与 NNAO。
- `ethane-cc-absolute.png/.pdf`：不平移的 Hartree 总能量，同样的两组视图。
- `curves.csv`：504 个原始点及接受状态，其中 499 个用于曲线。
- `summary.csv`：AO 数、初始／末端能量及拉伸能。
- `traditional-results.json`、`basis-library.json`、`run.log`：新增计算与冻结基组。
- `structures.json`：与原扫描逐字节一致的几何输入副本。
- `provenance.json`、`verification.json`：输入哈希、处理规则和独立核对记录。
