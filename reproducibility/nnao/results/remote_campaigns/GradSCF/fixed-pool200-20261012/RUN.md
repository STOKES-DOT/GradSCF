# 200 分子固定指数 MACE 训练

2026-10-12 提交作业 **96071**，已在 n004 运行。申请 A800 ×1、32 CPU、256 GiB
主存、72 小时。A100_40G 的 72 小时请求被 Slurm 拒绝（该卡型仅允许 8 小时），
随后改为 A800；调度器自动排除了 n005。

## 已确定配置

- 原 200 分子清单 SHA256：`9bf275038c21f3775111c749cf43287d1bc494503928a6079cd5d12fa0098f8f`。
- H/C/N/O/F，中性闭壳层，14–236 contracted AO，最大 40 原子。
- 固定本目录 `candidate-exponents.json`；szp663_direct/core6，全电子 RHF，
  def2-universal-jkfit DF，implicit SCF，float64 SCF。
- **从 seed=0 的初始 MACE 重新训练**，不继承 6 体系或旧联合训练参数。
- Adam 学习率 1e-4，目标 1000 epoch，无测试集／validation 集。
- 每个 epoch 在同一参数上遍历全部 200 个结构、平均梯度后更新一次。

## 运行阶段

1. 4 个隔离 CPU 进程准备混合池 S/h/packed DF 缓存；CPU 亲和性合计限制为 32。
2. 通过现有 `workflows.fixed_pool` 从 mmap 加载缓存，执行全样本 SCF／梯度与热调用检查。
3. 数值检查通过后自动进行 1000 epoch 的 GPU 训练；无效样本会停止，不能被跳过。

没有计算指数 VJP，没有保存四指标 ERI，没有改变共享 JIT 实现。
旧清单 full-primitive DF 估算为 238.49 GiB；新混合池实际大小以 preparation.json 为准。
保留现有 `vm.max_map_count=65530`，训练子进程映射数及 RSS 记录在 resource-watch.jsonl。

## 状态与复现

服务器目录：`/home/user_yuanjiao/private/homefile/GradSCF/artifacts/fixed-pool200-20261012/`。

- `train.slurm` / `launch.py`：资源、环境和实际启动命令。
- `references.json` / `basis-inventory.json`：完整输入及已通过的候选池加载核对。
- `source-sha256.txt`：提交前通过的 806 个源文件／数据文件哈希检查。
- `preparation.json` / `prepare-logs/`：固定缓存准备进度与逐分子错误。
- `training.log` / `training/batch_progress.json`：SCF 检查及训练进度。
- `training/gate.json`：数值检查结果；存在且 passed 才表示通过。
- `training/best.npz` / `last.npz`：最低平均能量状态／最新有效状态。
- `exit.json` / `launch-failure.json`：退出或启动失败记录。

提交时已经观察到缓存准备成功；这不代表 200 体系数值门槛通过或 epoch 1 完成。
