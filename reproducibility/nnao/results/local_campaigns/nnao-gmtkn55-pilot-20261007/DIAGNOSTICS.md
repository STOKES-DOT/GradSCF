# NNAO pilot 执行记录与未解决问题

日期：2026-10-07。正式结果来自 `job-93084/validation/summary.json`，共24条记录，23通过、1失败。`README.md`、`metrics.csv`、`verification.json` 不把诊断性运行或试验补丁的结果计为正式通过。

## 执行与来源

| 作业 | 范围 | 结果及使用规则 |
|---|---|---|
| 93079 | native构建准备 | 节点缺少手册列出的GCC12路径；启动失败，没有科学结果 |
| 93081 | 环境与最小验证 | 导入旧setup目录中的GradSCF，已停止；排除科学结论 |
| 93084 | CPU native构建、DF缓存、24结构GPU验证 | V100-SXM2-32GB，n002，16 CPU、64 GiB；运行13分40秒，退出1；23通过、S22/08失败 |
| 93092 | 依赖主作业的补查 | 主作业失败导致依赖无法满足，已取消 |
| 93093 | 三结构最终参数FD与exact shell-direct对照 | CPU，4核、16 GiB，退出0；见`job-93084/final_and_direct.json` |
| 93094 | 带调试回调的GPU诊断 | CUDA-only环境未注册CPU回调设备，诊断失败；不是原始梯度失败的根因证据 |
| 93096 | 独立GPU响应算子诊断 | 前向、JVP/VJP有限；wrapper真实残差未满足要求；见`response-gpu-93096.json` |
| 93098 | 更紧内部停止条件试验 | S22/08初始梯度仍失败，未进入其他结构回归 |
| 93099 | 带调试输出的残差修正试验 | 梯度有限，但存在instrumentation混杂；不能证明正式修复 |
| 93100 | 无调试输出的一次残差修正试验 | S22/08初始梯度仍失败，未进入其他结构回归 |
| 93103 | 无包装的cuda与cuda,cpu配置对照 | 两次能量、积分、梯度均在cuda:0，且都出现94个非有限梯度分量 |

激活Conda后重新设置源码路径，并对实际模块来源做断言，解决了93081的旧源码导入问题。正式运行源码及native库哈希在各结构结果中保存。native使用节点实际提供的GCC/G++8.5及环境MKL构建成功，没有安装新的依赖。

## S22/08：甲烷二聚体

正式GPU前向能量约为 **−80.3194689323405 Ha**；12个SCF周期，轨道残差约 **3.37×10⁻¹¹**，最小overlap特征值约 **0.0873954**，HOMO–LUMO间隙约 **1.14939 Ha**。这不是前向SCF未收敛或占据边界关闭。

独立密度响应JVP/VJP均有限。原GPU GMRES wrapper候选真实残差为 **2.4768×10⁻⁸**，RHS范数为 **20.8948**，相对残差 **1.18536×10⁻⁹**，超过原验收要求 **10⁻⁹**。有效性检查按设计拒绝该候选并传播NaN。扩大restart/maxiter未改善这一候选。

独立调用上游incremental/batched得到约 **3×10⁻¹⁰** 的相对残差；同结构本地CPU/JAX0.8.1完整梯度也有限。这些是定位线性求解与编译组合问题的证据，尚不能单独证明更底层原因，也不能将占据/虚轨道内部简并视为已确认根因。

## 尝试与最终代码状态

试验包含更紧的batched内部停止条件及一次`Aδ=b−Ax`残差修正，始终保持原始真实残差验收阈值与失败NaN策略。试验代码通过28项CPU定向回归，但没有在无诊断包装的原GPU案例中恢复梯度。

因此试验仅保存为 `experimental_gmres_response.patch`。本地和远程`gmres.py`已恢复到正式测量的基线版本，SHA-256：

```text
024432754b5753e3cd2b67eb1f1be3ac5c833f89ec9823071cd194d1ed2f6689
```

最终工作区仅增加pilot运行入口和9项编排/方向检查回归；既有用户改动保留。没有保留未解决原GPU问题的求解器算法修改。

最终源文件清单对比确认本地与远程所有共有Python/基组JSON/DAT文件哈希相同，远程整体哈希仍匹配93084的测量记录。本地仅多一个生成的CMake `InstallScripts.json`，不属于科学源码；清单见`source_inventory.json`。

## 最终本地验证

基线恢复后 **32 passed in 26.86 s**（CPU、JAX0.8.1、float64），`git diff --check`通过。确切命令：

```sh
PYTHONPATH=src /opt/anaconda3/bin/python -m pytest -q \
  tests/nnao/test_pilot_validation.py \
  tests/solvers/test_gmres_accuracy.py \
  tests/solvers/test_block_linear.py \
  tests/solvers/test_shared.py::test_linear_nonsymmetric_jit_transpose_and_second_derivative \
  tests/solvers/test_shared.py::test_linear_zero_tiny_rhs_and_failed_solve \
  tests/solvers/test_nonlinear.py \
  tests/test_scf_higher_order.py \
  tests/test_scf_autodiff.py::test_failed_adjoint_rejects_approximate_gradient \
  tests/test_scf_autodiff.py::test_shared_implicit_nonlinear_root_and_density_loss_pytree_jit \
  tests/test_scf_autodiff.py::test_fixed_point_argument_gradients_are_preserved
```

没有运行完整测试套件。当前结论是首轮验证完成、整体准入未通过；S22/08 GPU implicit backward仍待修复。本轮每结构独立优化5步，尚未启动共享模型训练。

## 数据与后续复现

TF原始选择与数据保持不变；本次结果存入独立的`results/hf_pilot_20261007/`。结果副本包含原始日志、逐结构JSON、5步轨迹、检查点、诊断及校验清单；大型DF积分缓存与native构建目录保留远程，不在本地/TF结果副本中。

下一次诊断应优先隔离实际custom-root backward的线性算子、数值求解与验真残差之间的编译差异，记录无instrumentation的HLO/数组来源，并以原GPU案例的重收敛FD和5步Adam作为修复验收。稳定性Hessian、共享训练、离子/UHF、DFT和力验证仍未进行。
