# 24 体系 MACE：32 → 64 通道对照

用户在 side conversation 中明确要求进一步增大参数，进行相同训练。
本轮仅将 channels 从 32 增至 64；max_ell=2、num_interactions=2、correlation=2、cutoff=5 Å、径向 MLP=(32,32) 保持不变。

原 24 体系结构、固定指数池、szp663_direct/core6、seed=0、Adam 学习率 1e-4、梯度裁剪 1、implicit SCF、等权平均 RHF 总能量目标、200 epoch 均沿用上一轮。
重新初始化网络，不继承 32 通道权重或 Adam 状态。比较已完成的默认 32 通道结果；8 通道结果也列入报告，但其 max_ell=1，应独立解释。

复用经过核验的 fixed-pool 训练入口。训练前检查原 32 通道实验完成状态、数据与指数文件哈希、公共科学源码哈希，以及新模型在全部 24 个结构上的参考收缩初始化。
随后执行训练入口原有的完整梯度冷/热重放和参数方向有限差分检查，不降低阈值或跳过失败分子。

一张 V100、32 CPU、128 GiB 主存、最多 4 小时。使用独立目录及缓存，不修改主线程源码、旧实验配置或运行中的训练。
所有对照来自相同源码 checkout `GradSCF_resume443_stability_20261013`。

结果：`wide/` 保存训练轨迹、检查点与数值检查；`comparison.json` 汇总匹配 epoch 的能量与耗时；`loss-curves.csv` 和 `per-molecule.csv` 保存三组曲线及逐分子比较。
单 seed、训练集结果不等于泛化效果；需要区分有限训练预算优势和充分优化后的表达能力限制。
