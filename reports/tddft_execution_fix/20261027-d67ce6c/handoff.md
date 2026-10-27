# 交接

实施树：`/Users/jiaoyuan/.codex/worktrees/9515/GradSCF`，分支
`perf/td-response-execution-20261027`。基线为 d67ce6c；五个生产提交分别处理
编译入口、网格 MO 场、公共 packed J/K、CPU RI→GPU 边界及可选求解器计数。
未合并或推送，原工作区 HEAD、index 和原有 4096 个未跟踪文件均保持不变。

最终源码的 CPU1 热态端到端：水/PBE TDA 9.308→2.369 s，full TDDFT
7.821→1.448 s，CH₂O/PBE0 12.743→6.177 s，OH/UKS 70.636→14.841 s。
苯 CPU8（4 物理核/8 逻辑 CPU）145.625→64.804 s。每项是一次首次调用加
三次新对象 warm；逐根原算子残差通过。不要混用中间 `final` 与最后的
`final-verified` 数据，也不要把 CPU1 与 GPU/CPU8 算直接速度比。

水新对象 warm 编译从 TDA 四个不同程序/full 三个程序降为零；不是四次
相同编译。苯的实际列数为 6/12，规划 batch32 没有把这些 action 补到32。
保留所有求解守卫，完整诊断另包含求解器外的五列谱观测。

55 项 CPU、13 项 GPU、8 项 solver/status，以及另外6项完整几何/NNAO
回归通过。MACE 是随机小模型 fixture，不代表真实训练权重或力监督验收。
显式 RI 几何 tracing 仍明确拒绝。opaque/global NN 回调没有被局部化或冻结。

CPU90 packed 临时量估计525 MB→6 MB，action0.475→0.160 s。GPU60默认
比旧 TD 私有路径慢约24%，但临时量108 MB→16 MB；相对旧公共逐行路径更快。
这是明确的资源取舍，不能宣传所有GPU路径都加速。真实 RSS 与 XLA估计分列。

`report.md` 为完整结论，`before_after.csv` 为匹配性能，`acceptance.json`
记录数值门槛，原始重复/编译/action 记录分别在三个 JSONL 中。完整日志、
HLO、NPZ、XML 在忽略 Git 的 `raw/evidence/`；哈希在 `raw_artifacts.json`。
`commands.txt` 与 harness README 可复跑。所有计算在 jy_gpu_2 的 n002/V100
Slurm110155内完成；任务结束后已取消该分配，不要复用旧作业号。
