# 优化版固定指数 NNAO：从 epoch 22 续训

用户授权停止原版并从断点使用最新优化代码重启。
原作业 96071 已停止；新作业 **96305** 于 2026-10-13 在 n004 运行。

## 已保存的状态

- 原 `last.npz` / `best.npz` 均为 epoch 22，平均能量 **−383.3513154811086 Eh**。
- 原检查点 SHA256：`f6fb68873d42cb0d98f487b06533d9a124c5a7f584e3812ee5ae1ea9c9496a71`。
- 归档目录 `archive/`，归档与原文件逐字节一致。
- 模型参数 7171 个，已接受梯度和 114818 字节的 Adam 状态完全保留。
- 实际反序列化检查通过，Adam 计数为 **22**，未重置优化器。
- 迁移后的 last/best SHA256：`e1a9cf61f9d4d95a4cc08f69a477cb8052747827e5fd80fc172b814890b8bd0d`。

迁移仅改变源码身份及添加来源记录；所有训练数组和历史保持不变。
源码 SHA256 `a8e825583cf5cb13b16b17eb9f206d36b2962d1bcf91e7c789047a28b6c92007`
与完成完整 200 体系验证的优化版相同。

## 重启配置与门槛

- 阶段 C：固定候选指数，只训练 MACE；原 200 个结构、等权总能量、无 heldout。
- 学习率 1e-4，目标为**总计 1000 epoch**，下一次更新为 epoch 23。
- A800 ×1、32 CPU、256 GiB 主存、72 小时；预分配关闭。
- 复用原有 200.78 GiB mmap 数值缓存及优化验证的持久编译缓存；是否命中由 JAX 校验。
- 重放全部 200 个分子：逐分子能量误差 ≤1e-8 Eh、平均误差 ≤1e-9 Eh，
  平均梯度 atol1e-7/rtol1e-5；还执行原工作流的热调用与方向导数检查。
- 门槛通过前不进行新 Adam 更新；失败保留 epoch 22 的有效状态。

## 初始化模板身份检查

初次仅按优化验证任务的 identity 迁移被严格检查拒绝：初始化模板哈希不同。
该尝试未写入新检查点，记录保存在 `initial-identity-check.json`。
随后在实际续训 GPU 上重建模板，得到的哈希与原训练完全一致；实际迁移没有修改
初始化指纹。验证任务的模板值不会用于替换训练参数。
具体见 `template-check.json`、`migration.json` 和 `optimizer-check.json`。

## 运行位置

服务器：`/home/user_yuanjiao/private/homefile/GradSCF/artifacts/fixed-pool200-restart-20261012/`。

- `resume.py`：限制源码身份变更、数组保真迁移和逐分子重放验证。
- `resume.slurm` / `slurm-96305.log`：实际配置与日志。
- `restart-gates.json`：完整重放通过后生成。
- `training/phase.json` / `batch_progress.json` / `gate-progress.json`：初始化和验证进度。
- `training/progress.json` / `last.npz` / `best.npz`：新任务的训练进度和检查点。

2026-10-13 00:12 的确认状态：迁移与 Adam 检查完成，重放门槛尚未完成，未确认 epoch 23。
本地迁移保护测试 3 passed，涵盖数组/状态保留、拒绝其他身份变化及逐分子误差抵消检查。
