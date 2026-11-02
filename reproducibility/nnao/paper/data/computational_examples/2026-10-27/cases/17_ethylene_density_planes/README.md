# 乙烯六套基组：分子平面与垂直截面的总电子密度

## 结果与图

**6/6 套 DF-RHF 计算通过，两个截面共 12 张密度图。**

- 分子平面 `xy, z=0`：[六图 PNG](figures/ethylene-density-xy.png) · [PDF](figures/ethylene-density-xy.pdf)。
- 垂直平面 `xz, y=0`，通过 C=C 键轴：[六图 PNG](figures/ethylene-density-xz.png) · [PDF](figures/ethylene-density-xz.pdf)。
- `figures/xy-*.png/pdf` 和 `figures/xz-*.png/pdf`：12 个单独面板，色标与组合图一致。

每组顺序为 MACE、自由收缩、STO-6G、STO-6G+P、cc-pVTZ、cc-pVDZ。
总密度非负，因此采用顺序蓝色色图与统一对数色标，而非带正负号的轨道/差分密度色标。
两组共用 `10⁻⁴–125 e/bohr³` 的范围、相同等密度轮廓和视窗，不对每幅图单独缩放。
低于下限的区域为白色；原始数组保留所有数值，不做密度平滑。

## 构型、方法与基组

构型直接取 GradSCF `examples/localization/ethylene_boys.py` 中的固定示例，单位 Å：

```text
C  -0.6695   0.0000   0.0000
C   0.6695   0.0000   0.0000
H  -1.2321  -0.9289   0.0000
H  -1.2321   0.9289   0.0000
H   1.2321  -0.9289   0.0000
H   1.2321   0.9289   0.0000
```

C=C 距离 1.339 Å，电荷 0，自旋 0，16 个电子。没有逐基组优化几何。
六组均为全电子 **DF-RHF / def2-universal-jkfit**、spherical AO、float64。
SCF 阈值：能量 1e−12 Eh、密度 1e−10、轨道残差 1e−9；报告额外检查内部 RHF 稳定性。

| 方法 | AO 数 | 总能量 / Eh |
|---|---:|---:|
| NNAO / MACE update 3351 | 36 | −78.002485762294 |
| NNAO / 自由收缩 | 36 | −78.002939718722 |
| STO-6G | 14 | −77.826648460672 |
| STO-6G+P | 36 | −77.866495218932 |
| cc-pVTZ | 116 | −78.063220692713 |
| cc-pVDZ | 48 | −78.039721267762 |

NNAO 沿用乙烷实验的 L-SZP：C `[2s1p1d]`、H `[1s1p]`。
MACE 使用历史 update-3351 权重；在当前实现上重新预测历史乙烷构型，逐壳层重现归档基组，
容差 1e−11。参数 SHA256 为
`2f4386c12cb4637b0b1f92682c86dba2b298b1e9c27f1926f1c2e24a1f6f1ade`。
权重沿用已有共享模型，未重新训练，也未复制另一份 81.7 MB 参数文件。

自由收缩固定相同指数、C 内层及 AO 数，每个原子独立优化所有可变径向收缩系数，m 分量共享。
使用项目现有 implicit SCF 梯度，经固定 primitive 积分投影反传；没有计算指数梯度、有限差分或完整四指标 ERI。
优化采用 primitive 重叠度量的可逆白化和 L-BFGS-B，分别从 MACE、参考收缩出发。
两个端点梯度范数为 **1.33e−6、2.99e−6**，均满足 1e−5 标准，能量相差 **1.56e−11 Eh**。
优化器的 `callback raised StopIteration` 是达到梯度阈值后的主动停止；数值接受依据不是 `optimizer_success`。
这是有限预算局部优化结果，不声称全局极限。

STO-6G+P 是明确的自定义对照：STO-6G 加 C(d, α=0.8)、H(p, α=1.1)，各一个 primitive、原始系数 1，
指数单位 bohr⁻²。与此前乙烷极化对照一致，不泛指任意软件的同名基组。

## 密度、截面与网格

使用所有占据轨道构造自旋求和总电子密度：

\[
\rho(\mathbf r)=\sum_{\mu\nu}D_{\mu\nu}\chi_\mu(\mathbf r)\chi_\nu(\mathbf r)
              =\sum_{i\in\mathrm{occ}}n_i|\psi_i(\mathbf r)|^2.
\]

包括 C 1s 内层，未做内层扣除、密度差、轨道相位着色或沿垂直方向积分。
每个截面为 641×561 点，x ∈ [−2.6,2.6] Å，另一坐标 ∈ [−2.2,2.2] Å。
全部 12 幅图采用完全一致的网格位置及显示尺度。
精确使用的单位转换 `1 Å = 1.8897261245650618 bohr` 已保存在配置和 NPZ 中。

乙烯 π 轨道在分子平面具有节面，`xy` 图主要反映内层与 σ 部分。
`xz` 截面包含平面上下的 π 密度贡献，但仍然展示**总密度**，没有单独分解 π 轨道。
垂直截面只有两个 C 核落在该平面上，所以不将四个平面外 H 核投影成截面上的原子。

## 定量比较的边界

相对 cc-pVTZ 的截面密度 L1 差定义为

\[
\epsilon_{\mathrm{plane}}=
\frac{\sum_{g\in\mathrm{plane}}|\rho_g-\rho_g^{\mathrm{cc-pVTZ}}|}
     {\sum_{g\in\mathrm{plane}}\rho_g^{\mathrm{cc-pVTZ}}}.
\]

| 方法 | xy 截面 | xz 截面 |
|---|---:|---:|
| MACE | 1.1194% | 1.2318% |
| 自由收缩 | 1.1427% | 1.2825% |
| STO-6G | 3.2578% | 3.7310% |
| STO-6G+P | 2.4611% | 3.0819% |
| cc-pVDZ | 0.9454% | 0.6007% |

这是所选有限窗口的**二维截面**指标，包括内层；不是三维密度误差或缺失电子数。
cc-pVTZ 是同方法基组参考，不是精确密度。
自由收缩比 MACE 降低 0.453956 mEh，但这两个截面上的参考密度差没有随之减小。
能量变分目标不保证每个密度指标单调改善。不能只凭热图肉眼接近就判定基组等价。

## 数值验证

- 六组 SCF 均收敛且通过 GradSCF 内部 RHF 稳定性检查；`Tr(DS)=16`，误差小于 3e−14。
- 独立 PySCF 2.9.0，使用逐原子冻结基组、相同辅助基，能量最大差 **3.13e−13 Eh**。
- 网格 AO 用本实验的 NumPy 实球谐 GTO 解析评估，s/p/d/f 次序和归一化分别与 PySCF 核对，最大误差 1.67e−16。
- 实际六套基组的 206 个三维 AO 探针、两截面的 **24576 个抽样密度点**通过交叉检查；抽样密度最大差 **3.13e−13 e/bohr³**。
- 密度矩阵与占据 MO 重构一致，所有保存的网格密度有限、非负。
- 没有进行有限差分梯度验证；验证不独立重复自由收缩的整个优化。

## 文件与复现

- `inputs/ethylene.xyz`、`structure.json`：固定构型。
- `inputs/frozen-bases.json`：六套逐原子完整基组；`mace-coefficients.npz`、`optimization/*-final.npz`：收缩参数和梯度。
- `states/*.npz`：六组密度矩阵、MO、占据和重叠；同名 JSON 保存 SCF/PySCF 诊断。
- `arrays/plane-densities.npz`：网格及 12 个总密度数组。键如 `xy_mace`、`xz_free`、`xy_cc-pvtz`。
- `summary.csv`、`results.json`：能量、AO 数、电子数与截面指标。
- `configuration.json`、`environment.json`、`model-lineage-check.json`、`verification.json`：完整条件、来源和验证。
- `plot.py`：只依赖 NumPy/Matplotlib，可在本目录重新出图，不运行 SCF。
- `verify.py`：需要 NumPy/PySCF，核对保存状态和抽样网格，不重跑 SCF。
- `run.py`、`density.py`：原始计算/解析密度驱动。SCF/积分及 implicit 优化使用 GradSCF，PySCF 仅用于实验交叉对照。

完整计算命令、依赖和本地 CPU 环境见 `environment.json`。本轮完整计算（含模型/JIT、积分准备、双起点优化、六组状态与两平面网格）
耗时 **72.18 秒**；补齐依赖、重建 native、绘图与后续验证不包含在该计时中。
大积分缓存和编译产物不随数据归档。重跑完整物理计算需要 GradSCF 源码及已有 update-3351 共享模型；
重画图只需本目录的数据。

本案例为数据与图的补充，不修改论文正文、历史结果或正在运行的训练。
