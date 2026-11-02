# Kr3 稳定性修复与无差分续训

2026-10-16：按用户要求重启500体系训练，关闭有限差分梯度验证。解析 implicit SCF backward 仍用于训练；SCF收敛、稳定性、梯度有限性、冷热重放与检查点身份检查保留。

## 失败与修复

任务98201完成epoch2，在epoch3第10批的候选update30上停止。Kr3的SCF已收敛，轨道残差5.53e-12；两次Davidson后特征对残差5.90e-7超过1e-7，因此稳定性尚未判定。恢复点是已接受的update29、epoch2、batch_cursor9；失败候选不晋升为恢复参数。

Kr3有69 AO、54个占据轨道，轨道Hessian维度810。V100对同一失败参数的原路径在本次运行中达到2.15e-8，说明原迭代的数值收敛对执行路径敏感；直接解析Hessian对照的三个低根为4.1126616165、4.1136189239、4.1136189239，残差不超过1.89e-12。

新增后备路径仅在两次Davidson均未收敛、轨道维度不超过1024时执行。逐列应用解析Hessian算子，避免扩大DF中间量；只存有界轨道Hessian，不保存完整四指标ERI。仍使用原1e-7特征对残差阈值，负曲率或未收敛结果仍拒绝。正常SCF路径与implicit固定点定义不变。

## 无差分检查

- 本地稳定性/DF回归14项通过；跳过差分的mock工作流测试7项通过。
- 合入当前工作区后新增功能测试12项通过。
- 远程任务98641：22项回归通过；失败Kr3，以及last/best各自的Kr3、CaO、BrBr_OPH3共7条前向、稳定性、解析梯度有限性检查通过。best能量按原记录重放。
- 本轮没有执行任何有限差分梯度验证。`check_parameter_gradients=False`、`gate-progress.json`及`gate.json`明确记录`skipped_by_request`；旧检查进度会被重置。

## 恢复与来源

仅迁移三个检查点的源码身份字段，原参数、梯度、Adam数组成员及其他元数据逐字节保留，旧文件完整归档。恢复仍使用batch_size50、每epoch10次更新、初始lr=1e-3且每100epoch减半、目标1000epoch。数据、基组指数、积分缓存及native库不变。

- 父源码：`0a5510b43d00e003eb6d0f16258d48d14a5553d05bfda5b960c6467da85fefc5`
- 新源码：`55b628b381fe84ee226c2882a1b095bce02ba0e32fc9634dec9865b93ad65e7f`
- 入口：`restart_kr3.py` / `restart_kr3.slurm`，1×A100_80G、32 CPU、512GiB、72h。
- 证据：`kr3-diagnostic.json`、`kr3-restart-verification.json`、`kr3-restart-gate-98641.log`；迁移和提交状态以各自JSON记录及Slurm实况为准。

小样本检查通过不代表恢复后的500体系整集重放已完成。
