# NNAO：前四周期主族 1000 体系

本版本按 2026-10-13 的最新要求，用于九组 S/M/L × SZP/DZP/TZP 新训练。
原 L-SZ/200 体系任务 96510 继续运行，不被替换。

- 1000 个体系：800 GMTKN55 v2、200 QH9Stable。
- 原子总数包含氢，最多 30；全部中性、spin=0、偶电子。
- 范围为前四周期主族元素与卤素，Z≤36，不含过渡金属。
- 实际覆盖 25 种元素：H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Ge As Se Br Kr。
- Ga 模板可用，但现有合格结构中没有 Ga；未人为构造其训练数据。
- 相比前一版，58 个含 Sn/Sb/Te/I 的体系被移除，另补入 58 个可核验结构。
  Pb/Bi 在上一版已因缺少当前模板所需原始数据而排除。
- 原 200 体系中满足原子数上限的 181 个继续保留。没有新增验证集；本集合不是独立泛化测试集。

`references.json` 是训练输入，`pilot.json` 包含覆盖全部训练元素的 26 个显式
标定参考。九组均重新标定候选指数，再从 seed 0 初始化 MACE 和 Adam，学习率
1e-4，目标 1000 epoch。各组的初始指数不能用原五元素候选直接代替。

内层采用 `reference` 策略：Li-Ca 使用原有六 primitive 6-31G 内层，更重的
支持元素保留 BSE 原始 3-21G v1 内层收缩。无 ECP、无缺失元素的参数拟合或补数。
默认 DF 辅助基保持 def2-universal-jkfit；SCF、稳定性和梯度门控仍须通过。

`manifest.json` 保存排除/替补 ID、数据哈希、元素频率和实验配置。
`train.jsonl` 与 `systems.csv` 是结构明细；`structures.extxyz` 保存同一批几何。
训练前还需完成 GPU 标定集检查及各自的 1000 体系首批编译、热调用和梯度检查。
任务已提交并不代表这些检查或正式训练已经完成。

来源与电子态沿用父数据集的逐条记录：GMTKN55 使用其电荷/自旋标签；QH9
使用官方生成代码的中性单重态约定，SQLite 本身没有独立的 charge/spin 列。
不使用反应能或 DFT Hamiltonian 作为监督标签。
