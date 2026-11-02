# NNAO 相对能量测试

固定模型：200 体系训练 epoch 629；18 个相对能量，39 个结构。
反应能和势垒按原始符号；非共价条目为正值表示结合的解离能，未做 CP 校正。

| 类别 | 基组 | 共同有效条目 | MAE vs HF/cc-pVTZ (kcal/mol) | MAE vs GMTKN55 (kcal/mol) |
|---|---|---:|---:|---:|
| reaction | minao | 6/6 | 23.1310 | 24.5959 |
| reaction | 3-21g | 6/6 | 3.1782 | 3.9080 |
| reaction | 6-31g(d,p) | 6/6 | 1.3586 | 2.4205 |
| reaction | def2-svp | 6/6 | 1.8416 | 2.6333 |
| reaction | cc-pvdz | 6/6 | 0.9276 | 2.7565 |
| reaction | cc-pvtz | 6/6 | 0.0000 | 2.5841 |
| reaction | NNAO-200-initial | 6/6 | 15.3650 | 16.8299 |
| reaction | NNAO-200-best | 6/6 | 12.6039 | 10.6491 |
| barrier | minao | 6/6 | 12.6725 | 10.6751 |
| barrier | 3-21g | 6/6 | 9.8803 | 11.0536 |
| barrier | 6-31g(d,p) | 6/6 | 1.4657 | 6.7354 |
| barrier | def2-svp | 6/6 | 1.3251 | 7.0643 |
| barrier | cc-pvdz | 6/6 | 0.7588 | 6.5919 |
| barrier | cc-pvtz | 6/6 | 0.0000 | 6.8209 |
| barrier | NNAO-200-initial | 6/6 | 13.7647 | 15.8371 |
| barrier | NNAO-200-best | 6/6 | 10.2104 | 16.4748 |
| noncovalent | minao | 6/6 | 2.5045 | 1.7186 |
| noncovalent | 3-21g | 6/6 | 4.1431 | 3.3747 |
| noncovalent | 6-31g(d,p) | 6/6 | 0.9636 | 0.8082 |
| noncovalent | def2-svp | 6/6 | 0.9707 | 0.9253 |
| noncovalent | cc-pvdz | 6/6 | 0.9091 | 0.9222 |
| noncovalent | cc-pvtz | 6/6 | 0.0000 | 1.6238 |
| noncovalent | NNAO-200-initial | 6/6 | 5.3069 | 5.0133 |
| noncovalent | NNAO-200-best | 6/6 | 1.5124 | 2.0140 |

GPU：Tesla V100-SXM2-32GB；CPU：32。
独立 GPU4PySCF 同基组核对最大误差：5.158540261618327e-12 Eh。

## 解释范围

- HF/cc-pVTZ is a finite-basis comparison, not CBS.
- GMTKN55 error contains both basis and HF method error.
- NCI values are uncorrected for BSSE; independently predicted monomer and complex contractions.
- Selected 18-reaction exploratory panel, not full GMTKN55 or WTMAD2.
- Category comparisons use the same complete-reaction intersection for every basis.

详细逐条数值见 relative-energies.csv；失败条目及版本信息见 summary.json。
