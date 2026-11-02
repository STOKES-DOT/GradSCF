# CaO update-20 修复与续训

## 失败与修复

任务 98020 在第 2 轮的整集评估中于 `GMTKN55-v2/ALKBDE10/cao` 失败。已接受 20 次小批次更新，但只有第 1 轮完整通过；`last.npz` 为 epoch=1、batch_cursor=10、update_step=20。

任务 98193 使用相同更新20参数、固定积分和阈值复现失败。此前只有从 core 初猜执行移位重启，原子初猜分支没有移位重启。新增的最后救援分支为：原子初猜的普通 SCF 失败后，以同一初猜和 level_shift=0.2 重新迭代，再以无移位配置精修。正常成功路径保持不变，稳定性及 implicit 固定点检查保留。没有更换数据、指数、辅助基、电子态、学习率或阈值。

同时修复 mini-batch 异常记录：`progress.json` 明确标记 failed，失败快照记录实际参与失败评估的参数。已接受参数、Adam 状态及游标的提交逻辑不变。两个 JIT trace 计数测试显式清理自身 LRU 起点，消除测试顺序依赖。

## 验证

- 本地隔离工作区相关回归 47/47 通过；合入当前工作区后的新增救援/失败恢复测试 11/11 通过。
- 远程任务 98197：CPU 回归 47/47 通过，随后使用 Tesla V100-SXM2-32GB、JAX float64 验证实际检查点。
- 更新20 CaO：E = −751.1809142167732 Eh，轨道残差 3.6351e−10，最小稳定性曲率 0.0760650；系数3个方向、2个步长的最大 FD 误差 2.6490e−8，完整 MACE 参数方向误差 1.0171e−9。
- 旧最佳参数在 CaO、BrBr_OPH3、Ge2H6、Kr2、ADIM6/AM2 上能量重放通过，最大差 9.0950e−13 Eh。
- 此前任务 98195 的6项失败来自把 CPU native FFI 单测错误放到 CUDA 默认设备；修正验证脚本后由98197完整重跑通过。

精修成功不代表救援便宜：该 CaO 仍累计经历934次 SCF 迭代，其中最后移位32次、无移位精修2次。此次修复保证正确收敛，没有改变常规求解顺序以追求加速。

## 检查点与提交

`migration-result.json` 记录归档位置及校验和。只修改三个检查点 `identity.scientific_source_sha256`，参数、梯度、Adam 的 NPY 成员字节和其他元数据全部不变；原文件完整归档。

- 父源码：`37f917b3d144c8545f0c024910e0f5fa47ae3c753cf0d0ddbae4b85b8bb88ca6`
- 新源码：`0a5510b43d00e003eb6d0f16258d48d14a5553d05bfda5b960c6467da85fefc5`
- native 未变：`bdbf4a3ac6f1538dfbd154d970461bd1bf0f10415abd65af8ad45bd08e1e248f`
- 续训任务：98201，1×A100_80G、32 CPU、512 GiB、72h。
- 2026-10-15 17:27:12 CST 状态：PENDING(Resources)，Slurm预计19:03:21启动；预测可能变化。

启动入口 `restart_cao.py` 从更新20恢复，执行完整500体系冷热重放和网络方向梯度检查，再补齐第2轮整集评估，随后才继续Adam更新。本次小样本验证不代表更新20已完成全500体系验证。

主要证据：`cao-update20-diagnostic.json`、`cao-restart-verification.json`、`cao-restart-gate-98197.log`、`migration-result.json`。
