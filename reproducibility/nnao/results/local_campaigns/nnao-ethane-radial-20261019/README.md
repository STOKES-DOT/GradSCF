# 乙烷扭转过程中 C0 的价层径向函数

比较原 44 AO NNAO 的 MACE update 3351 与逐构型自由收缩。固定指数和内层，DF-RHF/def2-universal-jkfit；25 个刚性构型，0–120°，间隔 5°。
C0 是输入的第一个碳，坐标 (0, 0, +0.7629881884) Å；H5/H6/H7 绕 C–C 轴旋转。
价层 s/p 各有 6 个 primitive，分别收缩为 1 个径向通道。固定 core s 不在图中。

## 定义与约定

`chi_lm = R_l(r) Y_lm`，球谐函数归一化；`R_l = r^l sum_p w_p exp(-alpha_p r^2)`，且 `integral r^2 |R_l|^2 dr = 1`。w 包含 primitive 和收缩归一化。采用与生产积分器相同的归一化公式，所有曲线对齐到 MACE 0° 的正主瓣。
`<r> = integral r^3 |R_l|^2 dr`。半径为 Bohr (a0)，R 为 a0^(-3/2)。CSV 同时提供 rR 和径向概率 r^2|R|^2。p_x/p_y/p_z 共用该径向部分。
这是基函数形状，不是占据分子轨道或电子密度；s 保持原始价层基函数，没有额外与 core 正交化。

## 结果

| 通道 | 方法 | <r>(0°) / a0 | <r>(60°) / a0 | 差值 / a0 |
|---|---|---:|---:|---:|
| s | MACE | 1.3260715183 | 1.3220247659 | -0.0040467524 |
| s | Free | 1.3278398184 | 1.3291899635 | +0.0013501451 |
| p | MACE | 1.5554656614 | 1.5464414505 | -0.0090242109 |
| p | Free | 1.5572746798 | 1.5584604095 | +0.0011857297 |

本构型扫描中 MACE 向 60° 预测径向收缩，自由变分端点的平均半径略增。该差异说明本例的基函数环境响应不同，不能仅凭径向曲线把能量差归因到一个 C 的 s 或 p 通道。

## 文件与复现

- `carbon-radial-functions.png/.pdf`：0/30/60° 的 R 和相对 0° 的变化。
- `carbon-radial-response.png/.pdf`：全部 25 个角度的平均半径。
- `carbon-radial-interactive.html`：拖动角度，比较 R、rR 或径向概率，并显示对应分子。
- `radial-functions.csv/.npz`、`radial-coefficients.csv`、`radial-metrics.csv`：函数、系数和指标。
- `summary.json`：源文件 SHA256、检查点、校验误差和本地绘图耗时。

命令：`/opt/anaconda3/bin/python artifacts/nnao-ethane-radial-20261019/plot_radial.py`。
本地 CPU float64 后处理；没有重新运行 SCF、MACE、训练或有限差分。
归一化最大误差 4.44e-16；解析矩与独立数值积分最大差 8.88e-16。
原始自由收缩是有限预算优化；多起点一致不构成全局最优证明。

额外校验：`normalization-check.json` 记录 100 个壳层与生产 JAX
`normalized_shell_coefficients` 的对照，最大权重差为 6.66e-16，CPU float64。
两张 PNG 已检查。交互页面的脚本通过 `node --check`；浏览器安全策略阻止
打开本地 `file://` 页面，因此 HTML 的浏览器交互效果尚未实测。
