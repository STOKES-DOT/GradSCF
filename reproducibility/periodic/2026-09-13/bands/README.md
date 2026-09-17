# Si 与金刚石能带对比

已分别完成 GradSCF 与 PySCF 的自洽计算，再使用各自收敛的密度进行非自洽路径查询。两套结果共用 PySCF 在采样路径上的价带顶作为零点；误差由原始 Hartree 能量直接计算。

## 计算设置

- 体系：金刚石结构 Si（常规晶格常数 5.430 Å）和 C（3.567 Å），各用两原子原胞。
- 方法：PBE；基组 GTH-SZV；赝势 GTH-PBE；两套实现使用相同参数。
- 自洽 k 网格：2×2×2；FFT 网格：41³；Ewald precision：1e-10。
- SCF：能量阈值 1e-11 Ha、GradSCF 密度阈值 1e-9、轨道梯度阈值 1e-7；最大 150 步。
- 路径：Γ–X–W–K–Γ–L，每段 12 个区间，共 61 个点；每个点比较全部 8 条空间能带。
- 路径使用常规笛卡尔倒空间坐标：Γ=(0,0,0)、X=(0,1,0)、W=(0.5,1,0)、K=(0.75,0.75,0)、L=(0.5,0.5,0.5)，单位 2π/a。
- 远程 c20 CPU，亲和性 0–3；OMP/MKL/OPENBLAS 线程数为 1；float64/complex128。完整软件版本与源码哈希在 metadata.json。

## 数值结果

| 体系 | GradSCF 路径带隙 / eV | PySCF 路径带隙 / eV | 最大能带差 / eV | RMS 能带差 / eV | 总能量差 / Ha |
|---|---:|---:|---:|---:|---:|
| silicon | 2.0625090804 | 2.0625090803 | 8.4493e-11 | 1.6698e-11 | 4.4054e-13 |
| diamond | 6.1503262056 | 6.1503262056 | 1.4311e-10 | 2.8018e-11 | 2.5580e-13 |

两个体系均通过预设比较门槛：最大能带差 <1e-3 eV、总能量差 <1e-6 Ha。所得差异远小于该门槛。
**这些带隙对应当前最小基组、粗 SCF k 网格和有限路径采样，尚未做基组与 k 网格收敛；不应直接作为收敛的材料带隙预测。**

## 可视化与数据

- band_comparison.png：能带叠图和放大的逐带误差。
- band_comparison.pdf / band_comparison.svg：矢量版本。
- silicon/bands.csv、diamond/bands.csv：逐点、逐带原始能量及差值。
- 各目录的 bands.npz：完整数组；summary.json：各体系指标和分阶段耗时。
- band-tests.log：远程能带接口测试，2 passed。comparison.log：实际计算日志。

曲线由真实采样点直接连线，未拟合平滑；误差面板单位为 neV（1e-9 eV）。已检查 PNG 版式，标注不覆盖曲线。

## 运行与限制

新增 mf.get_bands(kpts, chunk_size=4) 支持纯 LDA/GGA；使用固定 SCF 密度，不在路径上重新运行 SCF。HF/杂化泛函的任意路径交换处理尚未实现。
远程接口回归 2 项通过；本地基础回归 2 项通过、1 项因缺少 jax_xc 跳过。未运行全库测试或 GPU 验证。

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python tests/comparisons/compare_periodic_bands.py --output artifacts/bands-20260913
python tests/comparisons/plot_periodic_bands.py artifacts/bands-20260913
```

两套程序均按 chunk_size=4 查询路径。耗时包含 JAX 编译及各自的分块势构建，未据此作通用性能排名。

| 体系 | GradSCF SCF / s | PySCF SCF / s | GradSCF 路径 / s | PySCF 路径 / s |
|---|---:|---:|---:|---:|
| silicon | 15.86 | 100.74 | 18.27 | 286.02 |
| diamond | 5.41 | 124.94 | 14.68 | 313.96 |

远程使用隔离目录 /tmp/gradscf-bands-20260913，未替换远程生产源码；未提交或推送。

参考接口：[PySCF 能带计算说明](https://pyscf.org/user/pbc/scf.html#band-structure-calculations)。
