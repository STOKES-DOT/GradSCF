# 乙烷扭转：STO-6G 固定极化基组与 NNAO 对照

在原 25 个刚性构型（0–120°，间隔 5°）上，新增 STO-6G、STO-6G+C-d、STO-6G+C-d+H-p 三套固定基组，分别为 16、26、44 AO。C-d 指数 0.8，H-p 指数 1.1 Bohr^-2，均复制自原 benchmark 的 6-31G(d,p) 极化壳层。STO-6G 部分原样使用 PySCF 内置数据。

这是明确指定参数的自定义对照，不宣称存在唯一标准化 STO-6G* 或 STO-6G**。

| 基组 | AO | 0° 总能量 / Eh | 扭转势垒 / kcal mol-1 | 相对曲线最大差 / kcal mol-1 | RMS差 / kcal mol-1 |
|---|---:|---:|---:|---:|---:|
| STO-6G | 16 | -79.0652749251 | 3.043763 | 0.172113 | 0.102041 |
| STO-6G + C(d) | 26 | -79.0958458733 | 3.239255 | 0.023378 | 0.014904 |
| STO-6G + C(d), H(p) | 44 | -79.1076423017 | 3.205074 | 0.010803 | 0.005565 |
| NNAO / MACE | 44 | -79.2164465067 | 3.651135 | 0.435258 | 0.256013 |
| NNAO / free contraction | 44 | -79.2165133720 | 3.495155 | 0.279279 | 0.168139 |
| NNAO / C p2 / free | 50 | -79.2202526591 | 3.133362 | 0.082515 | 0.046675 |
| 6-31G(d,p) | 58 | -79.2379484101 | 3.223413 | 0.008373 | 0.006181 |
| cc-pVDZ | 58 | -79.2349523555 | 3.458704 | 0.242828 | 0.146354 |
| cc-pVTZ | 144 | -79.2596959729 | 3.215877 | 0.000000 | 0.000000 |
| STO-3G | 16 | -78.3060600952 | 3.049505 | 0.166371 | 0.098928 |
| 3-21G | 30 | -78.7934974028 | 3.022156 | 0.193720 | 0.112683 |

差值以 HF/cc-pVTZ 为比较；它不是精确极限。势垒来自固定几何的刚性扫描。

本例中固定 44 AO 对照的相对扭转曲线优于当前 44 AO NNAO 的 MACE 和自由收缩端点，但 NNAO 的绝对总能量更低。总能量下降不保证相对能量误差同步减小。不同基组的误差抵消需要在其他任务上另外验证；本例不能推导一种基组全面优于另一种。

NNAO 曲线沿用 update 3351，不是后来的 epoch 596 检查点。50 AO 曲线仅是新增第二套 C-p 的自由收缩结果。相同 AO 数仅匹配矩阵维度，primitive 池、内层和归一化后径向函数均不同；总能量差不能全部归于 MACE 的环境自适应。

新增计算：GPU4PySCF，任务 101856，Tesla V100-SXM2-32GB，32 CPU；75/75 SCF通过，数值循环耗时 39.85 秒（不含 Python 导入和 Slurm 排队）。最大轨道残差 5.92e-10。

计算为 DF-RHF/def2-universal-jkfit，float64，spherical AO；能量收敛阈值 1e-12 Eh，轨道梯度阈值 1e-9，最终残差 <1e-7。未新做有限差分或完整稳定性扫描。

文件：`ethane-relative-comparison.png/.pdf`、`ethane-absolute-comparison.png/.pdf`，`all-curves.csv`（11×25 点）、`comparison-summary.csv/.json`、`basis-library.json`。

计算命令：`sbatch artifacts/nnao-ethane-sto6g-polarized-20261020/run.slurm`。
绘图命令：`/opt/anaconda3/bin/python artifacts/nnao-ethane-sto6g-polarized-20261020/analyze.py`。
