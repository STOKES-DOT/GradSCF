# 当前500体系模型的历史相对能量推理

冻结参数：epoch 335 / update 3351；训练保持运行。

| 类别 | 模型/基组 | 有效条目 | MAE vs HF/cc-pVTZ | MAE vs GMTKN55 |
|---|---|---:|---:|---:|
| reaction | minao | 6/6 | 23.1310 | 24.5959 |
| reaction | 3-21g | 6/6 | 3.1782 | 3.9080 |
| reaction | 6-31g(d,p) | 6/6 | 1.3586 | 2.4205 |
| reaction | def2-svp | 6/6 | 1.8416 | 2.6333 |
| reaction | cc-pvdz | 6/6 | 0.9276 | 2.7565 |
| reaction | NNAO-200-best | 6/6 | 12.6039 | 10.6491 |
| reaction | NNAO-500-initial | 6/6 | 20.2719 | 21.7368 |
| reaction | NNAO-500-current | 6/6 | 6.5926 | 5.0935 |
| barrier | minao | 6/6 | 12.6725 | 10.6751 |
| barrier | 3-21g | 6/6 | 9.8803 | 11.0536 |
| barrier | 6-31g(d,p) | 6/6 | 1.4657 | 6.7354 |
| barrier | def2-svp | 6/6 | 1.3251 | 7.0643 |
| barrier | cc-pvdz | 6/6 | 0.7588 | 6.5919 |
| barrier | NNAO-200-best | 6/6 | 10.2104 | 16.4748 |
| barrier | NNAO-500-initial | 6/6 | 13.2716 | 14.8718 |
| barrier | NNAO-500-current | 6/6 | 9.1013 | 15.4006 |
| noncovalent | minao | 6/6 | 2.5045 | 1.7186 |
| noncovalent | 3-21g | 6/6 | 4.1431 | 3.3747 |
| noncovalent | 6-31g(d,p) | 6/6 | 0.9636 | 0.8082 |
| noncovalent | def2-svp | 6/6 | 0.9707 | 0.9253 |
| noncovalent | cc-pvdz | 6/6 | 0.9091 | 0.9222 |
| noncovalent | NNAO-200-best | 6/6 | 1.5124 | 2.0140 |
| noncovalent | NNAO-500-initial | 6/6 | 5.2952 | 4.8246 |
| noncovalent | NNAO-500-current | 6/6 | 0.7915 | 1.5545 |

单位：kcal/mol。

- 18 historical relative energies; not full GMTKN55/WTMAD2.
- HF/cc-pVTZ is a finite-basis anchor, not a correlated or CBS reference.
- NCI uses monomer sum minus complex, positive binding, no CP/BSSE correction.
- New versus old models differ in training data, exponent pool, core asset and architecture.
- No training-geometry overlap at pair-distance rounding 1e-5 Angstrom; not chemical-family or calibration independence.

GradSCF与独立PySCF同基组最大能量差：7.673861546209082e-12 Eh；参考后端：Tesla V100-SXM2-32GB。

## 三个历史例子

| 例子 | 当前500模型 | 原200模型 | HF/cc-pVTZ | GMTKN55 |
|---|---:|---:|---:|---:|
| CO2 + H2 -> HCOOH (G2RC/2) | 0.506653 | -20.389549 | -0.666039 | -2.260000 |
| HCN -> HNC forward barrier (BH76/37) | 71.995167 | 73.877536 | 47.551951 | 48.140000 |
| Formic acid dimer binding (S22/3) | 19.447157 | 22.562591 | 16.239128 | 18.753000 |

相比同池初始化，39/39 个结构的总能量降低；这不保证每个相对能量的误差同时降低。
例如 HCN 势垒由同池初始化的62.867564升至71.995167 kcal/mol，仍高于HF/cc-pVTZ的47.551951。

结构前向计算耗时486.91s，独立参考耗时45.70s；不含排队、导入和参考脚本重试。
训练任务99005在测试期间保持运行。最初参考脚本将GPU4PySCF默认level_shift=None误判为未解除移位；修正为显式0后参考78项全部通过，原误判记录单独归档。
