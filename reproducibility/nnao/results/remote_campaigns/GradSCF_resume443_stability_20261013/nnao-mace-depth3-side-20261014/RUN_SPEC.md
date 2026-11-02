# 24 体系 MACE：64 通道、2 → 3 层

本轮在 side conversation 中获准增加层数。仅将 num_interactions 从 2 增至 3，保持 channels=64、max_ell=2、correlation=2、cutoff=5 Å、径向网络和收缩输出头设置不变。

使用原 24 体系、固定的 epoch 629 指数池、szp663_direct/core6、全电子 DF-RHF、implicit SCF、seed=0、Adam 学习率 1e-4、全局梯度范数裁剪 1，重新初始化训练 200 epoch。没有继承两层模型权重或优化器状态。

对照：已完成的 `nnao-mace-width64-side-20261014/wide`，科学源码和输入完全一致。该轮 epoch 200 平均能量为 −114.1250568411208 Eh。
仅创建本实验目录，不改动主线程源码、旧实验或正在运行的任务。

训练前校验科学源码哈希、上一轮完成状态、几何/指数文件的字节一致性，以及所有 24 个结构的初始参考收缩。随后复用原训练入口的完整冷/热梯度重放和逐结构参数方向有限差分检查。

资源：一张 V100、32 CPU、128 GiB 主存，最多 4 小时。输出：`deep/` 检查点和日志、`comparison.json` 同 epoch 深度对照、`loss-curves.csv` 与 `per-molecule.csv`。
这是固定 seed 和训练预算下的深度对照；结论不直接等于充分训练后的表达能力极限或测试集泛化能力。
