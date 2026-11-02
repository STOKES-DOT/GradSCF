# Primitive 池标定与紧凑收缩评估

以下 CHNOF/core6 数值描述保留原小样本实验的定义。扩展到前四周期后的当前内层
规则见 [ELEMENTS.md](ELEMENTS.md)：`core_primitives='6-31g'`，内层 s/p6、3d3，
每个占据径向轨道只有一个固定收缩函数。新的候选必须携带 `core_reference`
策略与原始数据哈希；旧内层候选和缓存不能直接复用。

采用顺序流程：候选 primitive 池标定 → 小样本自由收缩评估 → 选择初始指数 →
固定指数训练 MACE → 冻结 MACE 微调指数。暂不引入周期性交替更新、多 GPU
分布式或跨分子共享 JIT。本模块当前实现前两项参考问题，未启动正式 MACE 训练。

## 基组与参数

共享训练当前显式使用 `szp663_direct`、`core_primitives=6`、spherical AO。
公开 `prepare_direct_basis()` 的默认仍是 `szp442_direct`；本工作流显式指定前者，
不修改其他调用者的默认值。

- H 的目标结构为 6s2p primitives → 1s1p，共 4 个 contracted AO，12 个池函数。
- 首排 p 区元素的价层／极化为 6s6p3d → 1s1p1d，共 9 个 contracted AO，39 个池函数。
- C/N/O/F 另保留一个由六个 primitive 构成的固定内层 s 收缩。因此每个原子共
  10 个 contracted AO、40 个混合池函数。
- **固定内层基组参数不等于冻结芯电子。** 所有电子仍进入同一 RHF SCF。
- 收缩系数在每个原子、每个活动壳层独立优化，同一 l 的 m 分量共享径向系数。
  自由收缩阶段不要求化学等价原子的系数相同；不训练 MACE。

现有 `primitive_basis()` 会把内层也解收缩；这里使用专门的
`integrals.pool.active_pool_basis()`，只解收缩 `slot >= 0` 的壳层。
固定内层块在 pool → compact 变换中使用单位矩阵。归一化和活动块复用公共
`contraction_matrix()`，不重复实现 Gaussian 归一化，也不对原始系数做普通 PCA。

## 阶段 A：候选池

指数按 `(element, l, shell_role)` 共享。对参考指数的对数均值 μ 和偏移 d，采用

```text
log(alpha_p) = μ + scale + exp(log_spacing) * d_p
```

零参数恢复原池；整体尺度和指数间距可以分别变化；指数始终为正。固定内层
指数／系数保持不变。初始化拒绝重复指数和不一致的共享池。

每次指数更新重新计算 native 积分并收敛全电子 RHF，通过既有 implicit SCF 和
native 指数导数求梯度。辅助基和几何固定。池重叠矩阵必须正定，最小本征值
大于 `1e-10`，条件数不超过 `1e10`；不会静默丢弃不良方向或换用其他求解器。

候选标定使用显式样本权重的平均 SCF 能量。默认试验限制两个对数参数在
`[-0.15, 0.15]` 内；这是局部候选探索范围，不代表指数已经全局最优。

## 阶段 B：自由收缩

对每个候选及每个分子缓存混合池的 S、h 和 packed DF 因子，然后保持固定
核心和目标 AO 数量独立优化活动收缩系数。变换为

```text
S_c = T.T @ S_pool @ T
h_c = T.T @ h_pool @ T
B_c[Q] = T.T @ B_pool[Q] @ T
```

缓存不含完整四指标 ERI。DF 辅助基、度量和 Hamiltonian 在两种表示间一致。
初值为当前模板收缩及确定种子的扰动，记录每个起点结果，保留最低有效能量。
优化采用 Adam 和显式能量下降回溯；候选必须通过 SCF/稳定性/有限梯度检查才
接受其参数与 Adam 状态。失败不覆盖已接受状态。

`max_steps`、`rejected` 与 `converged` 分开报告。有限步自由收缩结果是可达到的
局部结果，不是严格的最优收缩极限。未收缩能量也受 SCF 解分支限制；若出现
反序，工作流报错，不能用截断差值掩盖问题。

## 首批参考体系

`data/pool_pilot.json` 明确记录 H2、CH4、NH3、H2O、HF、CO 的理想化几何，单位
为 Å、权重均为 1，中性、`spin=0`。这些不是 GMTKN55 数据集条目，也不构成
性质或迁移性验证。使用 RHF、`def2-universal-jkfit`、float64，能量单位 Hartree。
开壳层原子、离子及其他电子态在本工作流中明确拒绝，不能静默改成 RHF 替代。

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
python -m gradscf.model.nnao.workflows.pool_calibration \
  --output-dir artifacts/pool-pilot \
  --calibration-steps 20 --contraction-steps 300 --starts 3
```

可用 `--ids H2 H2O` 选择更小子集；使用 `--references file.json` 替换参考问题。
输出目录必须为空，避免覆盖旧结果。每个优化检查点保存参数、梯度、Adam 状态、
起点和来源信息；此命令只处理 A/B，阶段 C 使用下述独立入口。

较大的参考集合使用 `JAX_PLATFORMS=cuda,cpu` 并增加 `--scf-platform cuda`。
native 积分、辅助基度量和指数 VJP 保持在 CPU；积分传到 GPU 求 SCF 与隐式
响应，积分余切再传回 CPU。自由收缩阶段的固定积分留在选定设备上复用。
请求 CUDA 时若没有 GPU 会明确失败，不会静默回退。
这避免大量 CPU SCF 编译产物耗尽 Linux 的 `vm.max_map_count`；提高主存
配额本身不能提高该上限。固定指数 MACE 的计算路径及收敛阈值保持不变。
`phase.json`、`ensemble-progress.json` 和 `timings.json` 记录当前阶段、
正在处理的体系以及已完成的计时，便于定位中途失败。

## 记录与选池

输出包含基组逐壳层清单、共享参数、native 库／源码哈希、指数方向有限差分、
每个起点检查点，以及未收缩／模板／自由收缩能量。网络能量和网络差值标为
未计算，不能把模板能量当作已经运行 MACE 的结果。

分别测量准备、含编译的首次调用、热调用和完整优化；内存使用可选 psutil
每 20 ms 采样的进程 RSS 峰值，缺失时为 null，不冒充精确峰值或 GPU 显存。

最终选择依据应是充分优化后的 compact 能量及其稳定性，而非仅看未收缩
能量下降。工作流始终输出 `selected_initial_pool: null`，不自动修改生产指数。
偶极矩、极化率和 BSSE 需要另行验证。

## 阶段 C：固定选定指数训练 MACE

将已选候选保存为 `candidate-exponents.json`：包含各 `(element,l,role)`
通道的原始／候选绝对指数（Bohr⁻²），以及 `parameters.keys/values` 中的
尺度和间距参数。加载时交叉校验原始 primitive 顺序与候选绝对值，保持内层不变。
该文件是此次训练的明确输入，不自动修改公共基组模板。

```bash
export PYTHONPATH="$PWD/src:$PWD/src/gradscf/model/nnao/vendor/mace-jax"
python -m gradscf.model.nnao.workflows.fixed_pool \
  --candidate path/to/candidate-exponents.json \
  --cache-dir artifacts/fixed-pool-cache \
  --output-dir artifacts/fixed-pool-train --epochs 5 --platform cpu
```

GPU 训练改用 `--platform cuda`，native 缓存准备仍在 CPU，投影、MACE 和 SCF
在请求的 JAX 设备执行。缓存支持 `--storage device/host/mmap`，保持 packed DF，
不保存四指标 ERI。相同缓存路径若指数、核心系数、辅助基或 native 库不一致则拒绝。

默认使用同一组六个中性单重态参考体系，等权 RHF 总能量为 loss；每个 epoch
遍历所有结构、取平均梯度，更新一次共享 MACE。网络输出完整原始收缩系数，
初值为当前模板的归一化参考收缩；所有后续更新均不依赖固定基线残差。
指数从不进入优化器，反向只经过 implicit SCF、JAX 积分收缩与 MACE。

入口先逐分子检查网络参数方向有限差分（步长 `1e-4`，绝对容差 `2e-6`、
相对容差 `2e-5`）、热调用能量和梯度一致性。SCF 使用能量 `1e-12 Eh`、
密度 `1e-10`、轨道残差 `1e-9`，implicit GMRES 容差 `1e-8`。
任何无效样本使整个候选更新失败，不跳过样本或覆盖已接受 Adam 状态。

有限差分的正负扰动复用已预热的能量／梯度 executable，舍弃额外梯度结果，
避免为每个分子再编译独立的仅能量 SCF。`phase.json` 区分冷调用、热调用、
方向检查与训练；`gate-progress.json` 在每个分子完成方向检查后写入。
`timings.json` 在各主要阶段结束时保存，不必等整个训练完成才查看。
检查方向只覆盖 NNX 元数据标记为可训练的参数；不可训练常量不扰动，Adam 也不更新。
掩码及可训练参数数量记录在 checkpoint identity 中，扁平状态和 Adam 序列化结构不变。

续训使用相同输入、源码和目录并加 `--resume --epochs <目标总轮数>`；先重算
已接受检查点能量和梯度。`last.npz` 保存最新有效状态，`best.npz` 保存最低平均
能量，均含参数、梯度与 Adam 状态；指数随完整 identity 保存。`gate.json`、
`metadata.json`、`timings.json` 分别记录数值检查、来源和分阶段时间／采样 RSS。
阶段 D 尚未接入此入口；短训练通过不代表 MACE 已充分优化或性质已得到验证。
