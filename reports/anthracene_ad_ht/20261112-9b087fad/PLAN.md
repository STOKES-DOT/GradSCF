# Anthracene AD-HT — fixed execution and acceptance plan

记录日期：2026-11-12（Asia/Shanghai）。基线提交 `9b087fad`；分支
`examples/anthracene-ht-20261110`，沿用已有隔离工作树
`/Users/jiaoyuan/.codex/worktrees/bd0a/GradSCF`。

目标固定为蒽 S1–S8 的真正 AD-VG/linear-HT：由同一次原生几何／隐式 SCF／
完整 TDDFT X/Y 线性化产生 8 个激发能和 24 个有符号跃迁偶极分量的 Cartesian
导数，再投影到全部 66 个正常模。已完成的小体系与数组回归不能替代实际蒽验收。
本计划保留原 B3LYP/Cartesian def2-SVP/grid level 3 设置与已验证的基态几何、
72 列 Hessian、66 个正频率模；旧 39 个 FD 点仅用于独立验证。

- [x] **TODO 1 — 接通并验证 full-TDDFT 向量 AD。**
  - [x] `implicit_eigenvector` 显式启用共享 structured RPA；保留默认 `eigenvalue_only` 行为。
  - [x] 保留完整占据／虚轨道 Fock、积分、XC 参数的动态依赖；fused H action 沿用原物理 action 与 microbatch，不构造完整物理矩阵。
  - [x] 验证独立 X/Y、正 metric 归一化、dense oracle、nested JIT/JVP/VJP、MLP 参数梯度及无效响应拒绝。
  - [x] 归档先 RED 后 GREEN 的 XML；范围与数值容差见 [implementation_report.md](implementation_report.md)。

- [x] **TODO 2 — 在同一远端 Slurm allocation 内完成实现验收。**
  - [x] CPU：数组／共享 solver／RSS 组合 45 passed；额外 custom-XC MLP 1 passed。
  - [x] CPU 原生几何链：HF/HeH+、PBE/H2、B3LYP/H2 的 gap 与有符号 dipole JVP/VJP 对独立重收敛 FD，3 passed。
  - [x] 单独 GPU 数组 AD 2 passed；谱 fixture/provenance 5 passed。GPU 数组结果不扩展为 GPU 原生几何 AD。
  - [x] `/proc` 进程退出竞争先复现后修复；保持 process-group RSS 的页数到字节语义，归档命令、环境与退出码。

- [ ] **TODO 3 — 验收实际蒽 32 个 pullback 与 66 模导数。**
  - [ ] job `122119` 的正式 AD 阶段已于 `2026-11-12 10:58:55+08:00` 启动；取得同一次 shared VJP 的全部 32 个有限、同签名 Cartesian 分量 checkpoint 与成功进程 receipt。
  - [ ] 核对 8 个目标态的有符号锚定、完整占据／虚轨道空间、第 9 个 guard root、原生库及 ground/harmonic/input/source hashes；逐项检查响应有限性。
  - [ ] 通过 Cartesian 平移检查，保存 `(8,66)` gap gradients 与 `(8,3,66)` dipole derivatives，保留全部 66 个正频率模。
  - [ ] 以旧 FD 数据中四个完整正负方向独立核验 gap／有符号 dipole 导数；FD 不回填模型导数。取得 `mode_response_status.json: complete=true` 后才勾选本项。

- [ ] **TODO 4 — 生成、检查并交付物理谱。**
  - [ ] 仅在 TODO 3 成功后组装 S1–S8 各态 FC/HT/有符号干涉与非相干总强度；共用绝对 photon-energy 轴，不做逐态峰值归一化或经验能量平移。
  - [ ] 验证每态及总光学面积误差不超过 `1e-5`，检查 guard roots 不进入 8 态谱，并保留上游 AD `derivative_method` 与签名。
  - [ ] 生成 CSV/NPZ/PNG/PDF，实际查看总谱、2×4 态谱、8×66 Huang–Rhys 图及 8 个目标垂直跃迁棒图；记录实际耗时与 RSS。
  - [ ] 取得 final spectrum/status/process receipts 并更新最终报告；确认 Slurm 退出／资源释放后才记录交付完成。

资源固定为远端 `122119`：`n006`、64 logical CPU、256 GiB、1 张 NV5090，
24 h allocation 上限；同 allocation 顺序执行。实际几何 AD 由示例显式选择 CPU，
不会因 allocation 包含 GPU 或 `JAX_PLATFORMS=cuda,cpu` 就成为 GPU 原生几何计算。
正式 AD guard 为 72000 s／220 GiB RSS；后续谱步骤受前一步成功退出约束。
本地只整理代码／文档／receipts，不运行科学计算或重复测试。

模型边界固定为 0 K、独立同曲率 VG surfaces 与每态 linear HT；未包含
Duschinsky 旋转、激发态完整 Hessian、热布居、态间振电耦合。未取得实际蒽
32 pullbacks、66 模导数与最终谱成功证据前，不将本任务标为完成。
