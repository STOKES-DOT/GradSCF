# 周期硅 Γ 点 TDDFT 吸收强度谱对比

分别使用 GradSCF 与 PySCF 独立完成 SCF、TDDFT 和速度规范振子强度计算，再采用相同的单位面积高斯展宽。结果使用原始振子强度，没有按各自峰值或电子数重标定。

## 量的定义与范围

- 硅金刚石结构两原子原胞，常规晶格常数 5.43 Å；Γ 点采样。
- PBE / GTH-SZV / GTH-PBE；FFT 网格 41³；Ewald precision=1e-10。
- 当前基组空间全部 16 个单重激发，完整 TDDFT；SCF 能量阈值 1e-11 Ha，响应阈值 1e-8。
- 速度算子为 p - i[r,V_NL]，包含 GTH 非局域修正。用固定 AO 傅里叶系数下非局域投影算子对均匀倒空间位移的导数实现，独立对照 PySCF 的解析矩阵。
- f = 2/(3 omega) * sum_xyz |v_transition|²，omega 用 Hartree。限制性 X/Y 先采用单位辛范数，再包含 sqrt(2) 自旋因子。
- Gaussian FWHM=0.30 eV；谱 S(E) 的单位为每原胞振子强度/eV，积分等于总振子强度。
- 这是 Γ 点约定下的周期原胞强度谱；未计算 bulk k 积分、宏观吸收系数或介电函数，也未做基组/k 采样收敛验证。

## 已验证结果

- 最大激发能误差：6.96009386e-11 eV。
- 简并能级组总强度的最大误差：1.57260871e-10。
- 最大绝对谱差：1.19656676e-08 eV^-1。
- max|S_GradSCF-S_PySCF| / max(S_PySCF)：2.09657179e-10。
- 总强度：GradSCF 22.880200218538，PySCF 22.880200218694。两条离散谱的数值积分均与对应总强度一致。
- 总能量差：4.12381240e-12 Ha。

简并子空间内的单个本征矢可任意旋转，因此强度以能级组求和进行比较。分组容差为 1e-5 eV；原始逐态数据仍全部保留。

| 能级组能量 / eV | 简并度 | GradSCF 总 f | PySCF 总 f |
|---:|---:|---:|---:|
| 2.60306181 | 3 | 1.647672193e-25 | 9.071371597e-21 |
| 2.84851721 | 3 | 18.22555949 | 18.22555949 |
| 3.58360332 | 2 | 7.739328133e-27 | 9.056485497e-21 |
| 4.35802604 | 3 | 4.617197701 | 4.617197701 |
| 4.84972526 | 1 | 2.967983703e-26 | 1.713122619e-23 |
| 16.62644543 | 3 | 0.03744302428 | 0.03744302428 |
| 18.11576447 | 1 | 7.171855309e-29 | 1.667328766e-27 |

最低的 2.6031 eV Γ 点激发组为暗态；明亮组位于 2.8485、4.3580 和 16.6264 eV。插图放大高能弱峰，纵轴仍为相同物理单位。

## 验证与复现

- 远程算子/归一化测试：2 passed，7.03 s。算子检查包含非零的非局域赝势修正，并检验 Hermitian 对称性。
- 本地：1 passed，1 skipped；本地 PySCF 2.9 没有新周期强度 API，实际参考使用远程 PySCF 2.13。
- 完整比较任务通过，耗时 216.22 s。远程 c20 CPU，亲和性 0–3，OMP/MKL/OPENBLAS 线程数 1，float64/complex128。软件版本与精确源码哈希见 metadata.json。
- 已核对远程测试源码与当前工作区哈希一致；PNG 经过视觉检查；未运行全库/GPU回归。

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python tests/comparisons/compare_periodic_absorption.py --output artifacts/absorption-20260913
python tests/comparisons/plot_periodic_absorption.py artifacts/absorption-20260913
```

新增接口：td.oscillator_strength(gauge="velocity")、td.transition_velocity_dipole()、pbc.optics.broaden_spectrum(...)。强度接口目前仅接受限制性 Γ 点参考；无效、过期或未收敛响应会明确报错。
GradSCF 返回物理复数跃迁速度，PySCF 的对应函数返回其虚部约定；整体本征矢相位不用于比较，强度与简并组总量才是比较对象。

## 文件

- absorption_comparison.png/pdf/svg：展宽谱、组强度棒谱与误差；包含高能弱峰放大图。
- spectrum.csv / spectrum.npz：完整谱网格和数组。
- transitions.csv：全部 16 态的能量与强度；简并态应结合 summary.json 的 groups 解释。
- optics-tests.log、comparison.log：实际运行日志。

参考：[PySCF 周期速度规范强度实现](https://pyscf.org/_modules/pyscf/pbc/tdscf/rhf.html)。PySCF 同样对其周期振子强度的解释范围给出限制，此处未将其当作宏观吸收系数。

远程使用 /tmp/gradscf-absorption-20260913 隔离源码；没有生产 PySCF 调用、提交或推送。
