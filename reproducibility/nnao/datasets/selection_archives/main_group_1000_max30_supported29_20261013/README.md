# NNAO 1000 体系候选集：29 元素支持版

日期：2026-10-13。本版本取代 `main_group_1000_max30_20261013` 作为后续元素扩展的候选输入；旧版保留用于溯源。

## 本次修改

按用户要求，仅使用有原始参考数据的当前 NNAO 模板。已查询 Basis Set Exchange 官方 API：3-21G v1 不包含 Pb/Bi，两次独立元素请求均返回 404。其他基组确实有 Pb/Bi 数据；本次排除原因是缺少符合当前紧凑模板的原始参考，而非整个 BSE 没有这些元素。

未对 Pb/Bi 拟合、补数、拼接或生成替代模板。删除原候选集中含 Pb/Bi 的 17 个体系，从满足约束的 GMTKN55 原始结构中补回 17 个。983 个原有体系继续保留。

## 数据规模

- **1000 个体系，800 GMTKN55 v2 + 200 QH9Stable。**
- **最多 30 个原子，包含氢；中性、spin=0、偶电子。**
- 元素：H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Ge As Se Br Kr Sn Sb Te I。
- 999 个分子/复合物和 1 个 He 原子参考；821 种分子式，同式最多 4 个几何。
- 580 个体系仅含 CHNOF，420 个含其他支持元素。
- 原正在训练的 200 体系中，符合尺寸限制的 181 个全部保留。
- 尺寸分布：He 原子 1；2–10 原子 408；11–20 原子 466；21–30 原子 125。
- 每条记录权重为 1，不设新的验证集。目标是收敛 SCF 能量的变分最小化，不使用反应能或 Hamiltonian 监督标签。

## 基组与模型

新的 29 元素 MACE 配置见 `model-config.json`。九个 SZP/DZP/TZP × S/M/L 配置支持显式元素表。

`core_primitives='reference'` 保留 Li-Ca 已有六 primitive 内层；更重的支持元素使用 BSE 3-21G v1 原始内层收缩，不强行填充到六项，不引入 ECP。BSE 原始文件和哈希随代码保存。

`references.json` 仍保留 `training_ready=false`、`basis_family=null`，因为本次只完成结构筛选与元素/基组基础支持。正式使用前须选择九个配置之一，并完成覆盖新元素的 A/B 池标定、SCF 稳定性、implicit 梯度及辅助基精度检查。尤其 Sn/Sb/Te/I 的全电子 DF 精度尚未因此获得验证。

## 数据验证与局限

全部 1000 份坐标与 GMTKN55 原始记录或 QH9 SQLite 几何逐项核对。带元素标签的距离指纹（量化到 1e-5 Angstrom）无重复，对旋转、平移和原子排列不敏感，镜像合并；它不是完整分子图判定。没有重新进行 SCF，也没有更改现有训练或队列。

GMTKN55 沿用逐体系电荷和自旋标签。QH9 SQLite 没有单独 charge/spin 列，采用其官方生成代码中的中性单重态 RKS 约定，并在每条记录中注明。该约定不是本次新计算的电子态结论。

He/K/Ca/Ge/As/Se/Sn 的样本仍很少，元素出现不代表已具备充分的环境覆盖；详细计数见 `element-coverage.json`。本集合与现有训练数据重叠，不能当作独立泛化测试集。

## 文件

- `train.jsonl`、`references.json`：结构数据。
- `systems.csv`、`element-coverage.json`：体系和元素清单。
- `exclusions.json`：17 个排除 ID 和 17 个替补 ID。
- `manifest.json`：本版本及父版本哈希、规则和来源摘要。
- `validation.json`：数据完整性检查，不是 SCF 验证。
- `xyz/`、`xyz-index.json`：Angstrom XYZ 与体系 ID 映射。
- `source-audit.json`：继承的原始数据源审查；本版选择数量以 manifest 为准。
- `SHA256SUMS`：目录文件校验和。

来源：[GMTKN55](https://github.com/grimme-lab/GMTKN55)、[QH9 官方读取代码](https://github.com/divelab/AIRS/blob/main/OpenDFT/QHBench/QH9/datasets.py)、[QH9 生成代码](https://github.com/divelab/AIRS/blob/main/OpenDFT/QHBench/QH9/dataset_generation_examples/QH9_stable.py)、[BSE 3-21G v1](https://www.basissetexchange.org/api/basis/3-21g/format/json/?version=1)。
