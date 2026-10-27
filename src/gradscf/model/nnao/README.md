# NNAO 模型模块

NNAO 使用 MACE 预测收缩系数，并通过 GradSCF 的基组组装、libcint 积分和
implicit SCF 计算变分能量与梯度。此目录包含 GradSCF 的实现；原始 MACE
项目隔离在 [`vendor/mace-jax/`](vendor/mace-jax/)，由其 `UPSTREAM.json`
记录版本和逐文件哈希，许可证保留在上游目录。

## 文件归属

- `encoder.py`、`graph.py`：网络和输入图。
- `initialization.py`：原有初始指数与参考收缩模板；本次整理不改变数值。
- `basis.py`、`grimme.py`、`data/`：基组组装和物理资产。
- `integrals/context.py`：orbital／auxiliary 计划与固定 primitive 数据准备。
- `integrals/fixed_exponents.py`：固定指数下的 JAX 收缩及其自动微分。
- `integrals/variable_exponents.py`：native contracted 积分及系数／指数 VJP。
- `integrals/cache.py`：primitive 缓存身份、NPZ 与只读 mmap。
- `integrals/fixed_pool.py`：选定候选指数后的混合池缓存，保持内层已收缩。
- `training/objective.py`：`BasisRHF` 能量与 SCF 有效性诊断。
- `training/trainer.py`、`checkpoints.py`：接受状态、优化器与续训。
- `training/alternating.py`、`native.py`：现有两种指数更新实现。
- `training/validation.py`：数值检查；`runtime.py`：数据读取与来源校验。
- `workflows/`：固定指数训练、交替标定、native 联合训练及检查。

## 运行

扩展元素配置与原始基组来源见 [ELEMENTS.md](ELEMENTS.md)。九个基组配置可使用
显式元素表和 `core_primitives='6-31g'`：内层 s/p 使用原始六 primitive 收缩，
Ga–Kr 的 3d 只取原始三 primitive 单收缩，保持内层 single-zeta。
`reference` 是这一策略的别名；运行中的旧 CHNOF 训练保持其冻结源码。

Primitive 池与自由收缩的 A/B 参考流程见
[池标定说明](POOL_CALIBRATION.md)，入口为
`python -m gradscf.model.nnao.workflows.pool_calibration --help`。
候选池确定后的阶段 C 入口为 `workflows.fixed_pool`，仅优化 MACE 参数。
指定 `batch_size`（CLI `--batch-size 50`）时，每个分子每轮仅进行一次能量与
implicit 梯度计算，batch 平均梯度后更新 Adam。训练不执行启动冷热重放、
更新后的候选复算、轮末整集评分或方向差分；SCF不收敛、稳定性未通过或梯度
非有限时停止，保留最近检查点和失败输入。未指定 batch_size 的小样本整集验证
入口仍保留独立验证流程。

训练 loss 是不同更新时刻的 batch 能量均值，记录为 `online_pre_update_mean`，
不能解释为最终参数的整集能量，也不用于自动选择 `best.npz`。旧的整集最佳
检查点保留原样。`last.npz` 保存更新后的参数和 Adam；其梯度属于前一次参数，
由 `gradient_update_step` 标明，`parameters_evaluated=False` 表示新参数尚未复算。
断点恢复不重放已处理 batch；旧版本在轮中接续时，仅统计剩余 batch 的真实
样本数，缺失的训练 loss 不补算、不伪造。

在已有依赖环境中，从仓库根目录运行。使用本地上游快照时：

```bash
export PYTHONPATH="$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax"
python -m gradscf.model.nnao.workflows.validate --help
python -m gradscf.model.nnao.workflows.train --help
python -m gradscf.model.nnao.workflows.fixed_pool --help
python -m gradscf.model.nnao.workflows.joint --help
python -m gradscf.model.nnao.workflows.joint_ad --help
python -m gradscf.model.nnao.workflows.optimize --help
```

固定指数小批次训练支持 `early_stop_window=10, early_stop_tolerance=1e-4`，
或对应的 `--early-stop-window 10 --early-stop-tolerance 1e-4`。
阈值单位为 Hartree，统计最近十次完整 epoch 在线平均能量变化的绝对值之和；
至少需要十一轮完整记录，残缺轮次不参与判定。默认关闭，策略和历史随断点保存。
判定复用已有 forward/backward 的能量，不增加 SCF 评分；在线均值并非最终保存参数上的整集能量。

生产共享训练的来源校验要求 MACE 来自本地快照。公共模型的可选安装入口
仍为 `GradSCF[nnao]`；源码快照不作为嵌套 Python 包打进 GradSCF wheel。
`tools/train_nnao_*.py` 等保留原命令行参数，仅调用这些模型内工作流。

训练主体可直接导入，不依赖仓库 `tools/`：

```python
from gradscf.model.nnao.training import BasisRHF, NativeJointEngine, JointAdam
from gradscf.model.nnao.integrals import prepare_native_context, native_projection
```

`BasisRHF` 当前限于中性闭壳层 RHF；默认的甲烷几何用于最小演示。
共享训练工作流保留原有 CHNOF、`szp663_direct`/core6 配置及参数顺序。
这次文件重组没有扩大已验证的科学适用范围。

## 积分与微分约定

模型工作流通过 `gradscf.integrals` 隐式使用唯一的生产积分接口 libcint，
不选择另一套积分引擎。CPU 积分与 GPU 上的 MACE／SCF 之间保留显式数组传输；
这里的 CPU/GPU 是执行设备，不是积分引擎选项。

固定几何／指数时，缓存 primitive 的 S、h 和 packed DF 因子；JAX 通过
含归一化的收缩矩阵投影到 contracted AO。DF 投影按辅助指标分块展开，
但当前实现仍需要完整 packed 输入与完整 contracted 三指标输出，不是磁盘流式算法。

阶段 C 的 `FreeContractionRHF` 将固定积分投影、既有 implicit SCF 与系数反向
合并为一次 JIT 调用；`evaluate_device()` 保留设备上的系数梯度供 MACE 反向，
`evaluate()` 则为自由收缩参考优化返回主存数组。mmap 缓存每次完整能量／梯度
调用只作为一组输入传入，避免前后向分别传输同一组 primitive 因子。
稳定性检查仍逐次执行，其内部 HVP 使用与轨道旋转能量 Hessian 等价的
occupied/virtual DF 分块表达式，不改变稳定性阈值或驻点逃逸策略。

native 联合路径直接计算 contracted S、h、三中心积分并求系数／指数 VJP。
辅助基度量在固定几何和辅助基时复用。不缓存完整四指标 ERI。
当前 DF 参数 AD 不包含几何／辅助基导数，不能据此宣称完整力训练已受支持。
该 contracted 联合路径会显式拒绝带 ECP 的 layout，避免漏掉 ECP Hamiltonian。
`qvszps` 的已有固定 primitive／direct 系数路径继续加入标量 ECP；其指数和
ECP 几何导数仍未实现。
指数微调仍使用这条 native 路径；冻结网络输出时，指数相关的收缩归一化导数
仍由 native 参数规则保留。阶段 C 的融合不会把动态指数作为固定缓存复用。

现有固定指数和联合训练的目标都是 RHF 总能量，`encoder.py` 调用上游 MACE
时关闭 `compute_force` 和 `compute_stress`。`assemble()` 保留图坐标到 Bohr
基组中心的映射，但这不代表现有训练入口已经实现力损失；变动几何时仍须重新
准备积分和辅助度量，不能将固定几何缓存当成完整几何响应。

`tests/nnao/test_libcint_consumers.py` 的两几何验收使用当前小型随机初始化
MACE 模型，覆盖物理基组投影、系数／log-exponent 差分、能量梯度、一次 Adam
更新和检查点往返。它不依赖归档训练权重，也不构成真实训练检查点的重新验证。
2026-10-26 的迁移工作树未提供原始 best975 检查点；相应历史检查仍显式报告缺失或跳过。

本次迁移保留现有数值缓存签名与严格来源检查：旧的已验证缓存不被静默
解释为新版本训练结果。完整训练源码身份因迁移变化，旧运行目录不得直接
作为同一来源续训；需显式恢复参数并重新验证，不移除来源保护。

当前支持小样本候选池标定、自由收缩评估和指定候选池的固定指数 MACE 训练。
候选通过显式 JSON 输入，不修改默认基组模板。冻结 MACE 微调指数的阶段 D
尚未接入新候选池工作流；不默认周期性交替更新。
完整 C/D 训练编排和更大规模缓存管理仍需另行实现和验证。

更多已记录实验和方法说明见 [NNAO.md](NNAO.md)。

## 可训练参数与常量

MACE 的扁平状态同时包含权重和 `is_mutable=False` 的激活归一化常量。
当前共享模型保留 7171 个状态值，其中 7167 个可训练；4 个常量仍保存在检查点中。
`training.trainer.trainable_parameter_mask()` 是统一的掩码来源，固定指数和联合路径
共同使用。Adam 更新及方向有限差分必须遵守同一个掩码。上游对常量使用
`stop_gradient`；有限差分若改变这些常量，检查的就不是实际训练参数空间。
