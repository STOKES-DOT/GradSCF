# Implicit SCF 反向数值余量修复

2026-10-09，GradSCF/NNAO stage-two，原训练作业 93538，第 41 轮有效检查点。

## 正式改动

- `src/gradscf/solvers/linear/gmres.py`：incremental 与 batched 数值路径的内部相对/绝对目标均为调用方目标的 0.1 倍。调用方最终真实残差认证、舍入余量、失败 NaN 保护、最大迭代和重启设置保持不变。当前 implicit 认证 1e-8，内部目标 1e-9。
- `tools/optimize_methane_nnao.py`：非有限系数梯度异常附带实际输入系数、原始梯度和已通过检查的前向诊断，保留原错误字符串及现有 Adam 接口。
- `tools/train_nnao_pilot.py`：失败时独立原子保存候选参数与异常数组；不接受失败更新，不覆盖 last/best/best_validation 或已接受 Adam 状态。初始失败也保存现场。
- 原超时续训脚本改用已验证环境 Python 的绝对路径，并在环境初始化前安装 EXIT trap。数值失败仍不自动续训；本次没有提交长训练。

## 本地验证

Python 3.12 / JAX 0.8.1，CPU，float64。

- 新 GMRES 精度回归先出现 8 个预期数值失败，再通过全部 13 项；覆盖 RHS 尺度 1e-12、1、1e12，纯 rtol/纯 atol、真实 batched 重试及一阶/二阶隐式导数。
- `PYTHONPATH=src /opt/anaconda3/bin/python -m pytest -q tests/solvers tests/test_scf_implicit_and_xc_energy.py`：151 passed，242.38 s；一条 ComplexWarning（complex-to-real cast）。
- `PYTHONPATH=src /opt/anaconda3/bin/python -m pytest -q tests/nnao/test_shared_pilot_training.py tests/nnao/test_methane_energy.py artifacts/nnao-stage2-20261009-adim6-fix/test_regression_provenance.py`：51 passed，25.84 s，含 H2/CH4 implicit FD、HVP、host/mmap、失败快照与恢复状态检查。
- 超时续训 guard：8 passed；Python 编译、作业脚本语法与 `git diff --check` 通过。

## 正式源码 GPU 回归

94841，jy_gpu_2/n002，Tesla V100-SXM2-32GB，32 CPU、128 GiB 主存，20 min 上限。使用原 jax_scf 环境（Python 3.11.15、JAX/JAXlib 0.10.0），无 solver monkeypatch 或调试 callback。

RHF/native DF，szp663_direct/core6，def2-universal-jkfit，float64，seed0，原 primitive packed DF mmap 缓存。SCF max150；能量/密度/轨道阈值 1e-12/1e-10/1e-9；implicit 最终认证 1e-8、max100 restart cycles、restart40、regularization0、rescue=.2 Ha。

从原第 41 轮参数、全批梯度与 Adam moments/count 重建第 42 轮候选，逐字节对照既有候选。顺序执行 MCONF/21、Amino20x4/LYS_xas、BSR36/r16，再对 ADIM6/AD6 调用原训练反向 100 次，全部通过。原检查点字节校验未变，作业退出码 0。回归程序耗时 692.60 s（不包括排队和启动）。

| 四阶中心差分方向 | analytic | FD | 绝对误差 | 结果 |
| --- | ---: | ---: | ---: | --- |
| 有效基组系数槽位中的归一化随机方向 | -0.0256578683332 | -0.0256578745924 | 6.2592e-9 | 通过 |
| MACE 参数梯度的归一化方向 | 3.4658387098541 | 3.4658387104021 | 5.4807e-10 | 通过 |

每个扰动点重新收敛 SCF；step=1e-4，atol=2e-6、rtol=2e-5。详细状态、103 条调用记录、100 组实际输入/梯度及 FD 数据在 `job-94841/regression/`，日志为 `regression-94841.out`。

## 源码身份与恢复边界

父科学源码 SHA256：`7f066083658982a552fda47e0144e762796f69788c6a1f1c7e70c77fbfec403a`。

修复后 SHA256：`87f14784cbc038349f65f311c4b664809925932abd2f0cccea834c5400102b30`。

`parent-sources.json` / `new-sources.json` 验证 764 个科学源码、原生源码/配置和训练工具文件，严格限定上述三个 Python 文件改变。远程其余源码、模型、数据、缓存及配置保留原训练版本。

这里是原失败候选点的重复与有限差分回归，不是 100 个训练 epoch，也未重做整批 200/50 新源码准入验证。训练仍保留第 41 轮有效状态。正式恢复须显式核验新旧源码及检查点兼容关系，不能静默修改旧 checkpoint identity 或假称旧 gate 已在新源码下执行。
