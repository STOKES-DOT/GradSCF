# NNAO 主族元素扩展：1000 体系候选训练集

准备日期：2026-10-13。**仅准备数据，未修改现有训练、队列、基组或检查点，未运行 SCF。**

## 范围与组成

- 1000 个独立几何：800 个 GMTKN55 v2 体系、200 个 QH9Stable 分子。
- **最多 30 个原子，包含氢**；999 个分子/复合物，另含一个明确标记的 He 原子参考。
- 中性、spin=0，偶电子；计划首先使用全电子 RHF 变分能量训练，不使用原数据的反应能或 DFT Hamiltonian 标签。
- 覆盖 31 种主族元素及卤素：H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Ge As Se Br Kr Sn Sb Te I Pb Bi。
- 563 个体系仅含 CHNOF，437 个体系含其他元素；共 838 种分子式，同式最多 4 个几何。
- 原 200 体系中符合 30 原子上限的 181 个全部保留；其余 19 个被排除，ID 见 manifest.json。
- 所有记录初始权重为 1；未设置新的 validation/test 划分。这不是独立测试集，不能用于宣称对 GMTKN55 或 QH9 的无泄漏泛化成绩。

| 原子数（含 H） | 体系数 |
| --- | ---: |
| 1（He 原子参考） | 1 |
| 2–10 | 416 |
| 11–20 | 458 |
| 21–30 | 125 |

## 元素覆盖

下面统计含该元素的体系数，一个体系可以同时计入多列。

| 元素 | 体系数 | 元素 | 体系数 | 元素 | 体系数 |
| --- | ---: | --- | ---: | --- | ---: |
| H | 885 | He | 1 | Li | 23 |
| Be | 6 | B | 64 | C | 736 |
| N | 410 | O | 415 | F | 154 |
| Ne | 8 | Na | 10 | Mg | 8 |
| Al | 36 | Si | 40 | P | 93 |
| S | 78 | Cl | 75 | Ar | 7 |
| K | 1 | Ca | 1 | Ge | 1 |
| As | 1 | Se | 1 | Br | 37 |
| Kr | 3 | Sn | 1 | Sb | 9 |
| Te | 10 | I | 43 | Pb | 9 |
| Bi | 9 | | | | |

K/Ca/Ge/As/Se/Sn 各只有一个几何，He 仅一个原子参考。覆盖元素不等于覆盖化学环境，不能据此声称这些元素已可可靠训练或泛化；需要后续补充独立来源的小分子环境。Ga/In/Rb/Sr/Xe 等未从本次可核验的结构源中选入，没有通过替换元素或猜测坐标制造样本。

## 来源与电子态

1. GMTKN55 来自 TF 盘已保存的官方 v2 快照，commit `ccabc16c8c4677fc60c19870ae96ac25405844a2`。沿用上游电荷、自旋标记和原始坐标，要求被保留反应引用。数据归属与 CC-BY-4.0 说明见附带的 GMTKN55 来源文件。
2. QH9Stable 从 TF 盘 SQLite 数据库以只读方式读取 `id,N,Z,pos`，不读取 `Ham`。固定 seed=0 抽取 3000 个 ID 作为候选，再选取 200 个以补充分子式多样性；候选 ID 全部保存。
3. QH9Stable 的 `pos` 是 Angstrom；`Z` 是 little-endian int32、`pos` 是 little-endian float64，依据[官方读取代码](https://github.com/divelab/AIRS/blob/main/OpenDFT/QHBench/QH9/datasets.py)。[官方生成代码](https://github.com/divelab/AIRS/blob/main/OpenDFT/QHBench/QH9/dataset_generation_examples/QH9_stable.py)使用默认中性单重态 Mole 和 RKS。数据库没有独立 charge/spin 列，因此记录明确区分“数据集生成约定”与 GMTKN55 的逐体系标签；本次未重新确定电子基态。
4. TF 盘上的 QM9_GW_BSE、OpenMolcas 示例和 agent 工作目录也已检查。前者没有增加元素覆盖；后两者的电子态/来源记录不一致，未混入这批训练数据。

## 筛选与验证

优先保留现有训练的合格体系，再保留稀有元素几何，给其他新增元素争取至少 40 个样本（不足时保留可用样本），最后兼顾分子式、子集和尺寸补齐。GMTKN55 与 QH9 跨源统一去重。

去重使用带元素标签的逐原子距离指纹，距离量化到 `1e-5 Angstrom`，对平移、旋转和原子排列不敏感，镜像合并。不同构象可以保留；这不是完整分子图同一性判定，也不能严格排除罕见等距离非同构结构。每条记录还保留原始来源和几何。

已检查全部 1000 条记录：原子上限、有限坐标、电荷/自旋与电子数、无距离小于 0.2 Angstrom 的核对、指纹唯一性、与原始来源坐标精确一致、XYZ 往返精度。`validation.json` 是数据层检查结果，**不是 SCF/梯度验证结果**。

## 当前模型支持边界

`references.json` 故意设置 `training_ready=false`、`basis_family=null`，避免被当前 CHNOF 工作流误当作可直接开跑的输入。

- 437 个体系需要扩展九基组模板、MACE 元素列表、指数通道与参数初始化。
- 114 个体系含 Z>20 元素，现有六 primitive 内层数据尚需验证；需要同时核查全电子 Hamiltonian 和辅助基。没有引入 ECP。
- 17 个体系含 Pb/Bi，现有 NNAO 旧模板也缺失，需要新增。
- He 原子参考需要单独验证零边图 MACE 与 SCF 路径。
- 所有体系仍需在具体 SZP/DZP/TZP 配置下做 SCF 稳定性、implicit 梯度、条件数和显存检查。30 原子上限不是显存保证。

本批数据不要求改变现有九基组训练。后续训练前应固定扩展后的基组、Hamiltonian、辅助基和元素配置，再建立缓存；不得静默跳过不支持的元素。

## 文件

- `train.jsonl`：1000 条标准化结构及逐体系来源/支持状态。
- `references.json`：同一数据的 JSON 容器；尚未绑定具体基组。
- `systems.csv`：可筛选的体系清单。
- `element-coverage.json`：元素频率与稀疏元素标记。
- `manifest.json`、`source-audit.json`、`deduplication.json`：规则、来源摘要、哈希和排除情况。
- `xyz/`、`xyz-index.json`：1000 个 Angstrom XYZ 文件及 ID 映射。
- `validation.json`：独立数据完整性检查。
- `prepare_dataset.py`、`validate_dataset.py`：对应的准备与核对脚本副本；运行时需要按原工程目录布局定位上游数据。
- `SHA256SUMS`：发布目录文件校验和。

训练目标仍是自洽变分能量，不需要收集监督标签。能量改善和偶极矩、极化率、BSSE 的改善应分别验证。
