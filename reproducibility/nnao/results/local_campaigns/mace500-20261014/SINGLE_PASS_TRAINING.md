# 固定指数小批次训练：一次能量与梯度计算

2026-10-16：根据用户要求，将当前 NNAO 小批次流程改为每个 batch 只评估一次，再更新 Adam 并保存。实现及验证均在本地完成，本轮未操作远程训练任务。

## 行为

- 每个分子每 epoch 一次能量/implicit 梯度调用；500体系、batch_size50对应500次分子级计算和10次Adam更新。
- 不执行启动整集冷热重放、候选参数复算、轮末整集评分或方向差分。首次JIT发生于实际需要的batch。
- 当前batch SCF、稳定性或梯度失败时停止，不提交该batch更新。非有限优化器提议也不提交。
- `last.npz` 保存更新后的参数及Adam状态；梯度来自更新前的参数，记录 `gradient_update_step`，新参数标记 `parameters_evaluated=False`。
- 每轮 `loss_kind=online_pre_update_mean`，汇总不同参数时刻的batch能量，不解释为最终参数的整集能量，不自动生成或更新 `best.npz` 和 `last_full_epoch.npz`。
- 保留历史整集最佳检查点；从旧版本轮中恢复时只汇总剩余batch，并记录真实样本数和完整覆盖标志。已经完成全部batch的旧游标可直接结束该轮，缺失loss记为null。
- 未指定batch_size的小样本整集验证路径保留原有行为；CLI支持 `--batch-size 50`。

## 验证

旧实现对4个样本训练两轮调用28次评估器；新实现严格8次。相关本地测试32项通过，包含调用次数、确定性shuffle、Adam/学习率状态恢复、失败输入快照、旧检查点游标、在线loss和不伪造最佳模型。真实H2+MACE断点恢复两步严格调用两次实际能量/梯度评估。合入当前工作区后12项针对性复核通过，diff检查通过。

没有运行有限差分验证，没有测量新流程的远程epoch耗时。旧小样本全批次测试中包含差分的3项本轮未运行。

## 切换

当前运行任务98645保持原工作状态。切换时需要先归档最新检查点，保留参数、Adam和batch游标；确认只有训练控制源码改变后迁移源码身份，再启动新流程。用户此前要求不动工作状态，因此本轮没有停止、替换或重启该远程任务。

## 用户确认后的远程切换

用户随后明确授权重启。旧任务98645已于2026-10-16 16:57:12 CST取消；冻结的最新断点为update40、epoch3、batch_cursor10，保留第4轮全部小批次更新。旧轮末评分未完成，新流程不补算，直接进入第5轮。

检查点和旧源码已归档到远程同目录 `single-pass-archive-20261016T085842.487126Z/`。核对仅 `training/minibatch.py` 和 `workflows/fixed_pool.py` 两处科学源码变化；native、模型配置、数据、参数/梯度/Adam数组字节和其余检查点元数据保持一致。旧整集best和last_full_epoch保留原样。

新源码SHA256：`d86bd78ed807b3f5aae93bcf63988739311edc1858efe4c6c4b72b0a132a095b`。新任务99005，A100_80G×1、32CPU、512GiB、72h；2026-10-16 17:02 CST查询为PENDING(Resources)，Slurm预测17:09:55启动，实际状态以实时查询为准。完整记录见 `single-pass-switch.json` 与 `restart-single-pass-submitted.json`。
