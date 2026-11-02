# 当前 NNAO 基组：相对能量测试

## 固定对象

- 当前 200 体系训练任务：96984；不停止、不修改该任务。
- 冻结最佳 checkpoint：epoch 629，平均训练能量 −383.86174763141355 Eh。
- checkpoint SHA-256：`fa8a17cb149ee9c3bc1e672402df8ac1e97bfb712a625e9c47c3fbb913ce27bf`。
- 科学源码 SHA-256：`d7a4e0ace93ad1faccae713c7ff5cb4b67cbbbd39035ba8e9ee2a42baf286090`。
- 基组 `szp663_direct`，内层 6 primitive 固定收缩；MACE 元素表 H/C/N/O/F；使用 checkpoint 自带的已标定指数。
- 初始化对照使用相同 seed、网络结构和固定指数，重新生成训练前 MACE 参数。

## 数据与范围

GMTKN55 v2 固定提交 `ccabc16c8c4677fc60c19870ae96ac25405844a2`，读取本地 TF 盘的结构与反应索引。
18 个相对能量条目、39 个结构，最大 17 个原子，均为中性闭壳层。

| 类别 | 条目 |
|---|---|
| 反应及异构化能 | G2RC/2、5、12、17；ISO34/1、14 |
| 势垒 | BH76/37、38；BHPERI/1、7；BHROT27/1、7 |
| 非共价作用 | S22/1、2、3、9、10、16 |

保持原始结构、化学计量、符号和参考标签，不进行几何优化。先固定选择，后计算能量，不按结果筛选。
相对当前 200 训练结构：ID 重叠 0；按元素配对距离（保留到 1e-5 Å）检查的几何重叠 0；同分子式结构 9。
此检查不保证化学家族、片段、初始指数标定或历史模型选择的独立性。
BH76/37 和 /38 共用过渡态，不能将所有条目视为统计独立。

## 计算与比较

- DF-RHF，spherical AO，float64；辅助基 `def2-universal-jkfit`。
- SCF：最大 200 次，能量阈值 1e-12 Eh，轨道梯度阈值 1e-9；保存残差、收敛与稳定性诊断。
- GradSCF native contracted 积分 + GPU SCF，计算初始化和训练后两种 NNAO；导出原子环境相关的固定基组，再由 GPU4PySCF 独立核对。
- 传统基组：minao、3-21G、6-31G(d,p)、def2-SVP、cc-pVDZ、cc-pVTZ。
- 同基组 GradSCF/GPU4PySCF 总能量差要求 ≤1e-7 Eh。
- 不做训练、参数更新、完整四指标 ERI 或 force/backward 计算。

相对能量为 `sum(nu_i * E_i) * 627.5094740631 + energy_add`，单位 kcal/mol。
先保存每个单点，再按原始反应组装；缺失或失败的任一组分会令该基组对应的整条反应缺失。
分类统计使用所有基组共有的完整条目，报告覆盖率、MAE、带符号平均误差和最大绝对误差。

两种参考分开报告：HF/cc-pVTZ 是有限基组 HF 对照，非 CBS；GMTKN55 参考值体现总体误差，包含 HF 方法缺少电子相关的影响。
非共价条目遵循 `单体和 − 复合物` 的正结合能符号；各真实结构独立预测 MACE 收缩，未做 CP 校正，包含基组自适应变化的影响，不能直接称为 BSSE。
本面板不构成完整 GMTKN55 或 WTMAD2 评测。

## 执行与产物

Slurm 任务 97621：V100 ×1、32 CPU、128 GiB 主存、2 小时时限；提交后运行节点 n003。
命令入口：`sbatch artifacts/nnao-relative200-20261014/run.slurm`。
远程 checkout：`/home/user_yuanjiao/private/homefile/GradSCF_resume443_stability_20261013`。

输入见 `snapshot.json`、`test-selection.jsonl`、`reactions.jsonl`、`selection-manifest.json`。
运行结果依次为 `native-results.json`、`frozen-bases.jsonl`、`gpu4/comparison.json`。
汇总结果为 `relative-energies.csv`、`category-metrics.csv`、`summary.json` 和 `README.md`。
每个阶段保存实际硬件、版本、时间、诊断及失败；文件存在不等于整个测试通过。
