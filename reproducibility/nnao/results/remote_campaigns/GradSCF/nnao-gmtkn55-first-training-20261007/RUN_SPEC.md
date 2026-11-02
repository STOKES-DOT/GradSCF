# GMTKN55 首批共享 NNAO 训练

2026-10-07 用户授权：将本轮 implicit 线性求解容差设为 `1e-8`，完成验证后训练 **1000 epoch**。

## 数据与模型

使用已固定的 GMTKN55 v2 小样本选择：24 条 CHNOF、中性、spin=0 结构，包括16个核心、6个扩展、2个配套单体。全部属于本轮训练样本，没有独立测试集；不据此评价泛化。

一个共享 MACE 同时处理全部结构，统一元素词表 `(H,C,N,O,F)`。配置：channels=8、2个interaction、max_ell=1、correlation=2、zero_init=True、seed=0。基组 `szp663_direct/core6`，球谐AO，无ECP；固定几何、primitive指数和内层。

## 目标与更新

\[
L(\theta)=\frac{1}{24}\sum_i E_{\mathrm{RHF,DF}}(R_i,c_\theta(R_i)).
\]

每个结构先收敛SCF，再对密度固定点做implicit backward。每epoch在同一参数处累积24个梯度，再执行一次Adam更新。学习率 `1e-4`，global gradient clip=1。参考能量不进入loss；没有额外正则项。

前向SCF保持 `max_cycle=150`、能量容差 `1e-12 Ha`、密度容差 `1e-10`、梯度容差 `1e-9`。implicit设为 `tolerance=1e-8`、max_iter=100、restart=40、regularization=0。

## 启动条件与保存

原失败的甲烷二聚体先验证，再复测全部24条：方向有限差分、5步Adam、SCF与基组诊断全部通过后才能训练。训练中任何不合格结构都会停止该步，保留最后有效检查点，不静默丢样本。

`last.npz` 和 `best.npz` 保存参数、完整Adam状态、epoch、对应参数处的loss与梯度；允许严格续训。逐epoch记录均值能量和每结构诊断。native固定积分采用既有CPU DF缓存，GPU执行收缩、SCF和AD；不保存完整ERI。

Slurm使用home分区、单张V100、16 CPU、64 GiB。验证时限1小时，训练时限4小时。实际硬件、软件版本、源码/数据/缓存哈希由运行记录保存；执行状态见对应日志、summary和status文件。
