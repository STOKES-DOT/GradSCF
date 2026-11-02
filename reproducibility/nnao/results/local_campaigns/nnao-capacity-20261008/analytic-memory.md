# NNAO 训练容量的数组尺寸分析

日期：2026-10-08。范围：当前 200 train＋50 validation，RHF、float64、球谐
`szp663_direct/core6`、`def2-universal-jkfit`。本分析只读取结构元数据和源码，
没有计算积分、SCF 或分配 GPU 大数组。精确字节数和来源哈希见 `counts.json`。

## 当前最大体系

本轮筛选显式限制 `natoms ≤ 40`；40 原子不是已测得的硬件容量上限。

| 指标 | 结构 | 组成 | 原子数 | AO N | Primitive AO P | Auxiliary AO Q | Packed DF / GiB |
|---|---|---|---:|---:|---:|---:|---:|
| 最大原子数 | ADIM6/AD6 | C12H28 | 40 | 232 | 876 | 1404 | 4.018 |
| 最大 AO 数 | BSR36/c4 | C14H24 | 38 | 236 | 918 | 1482 | 4.658 |
| 最大 primitive／缓存 | S22/07 | C10H11N7O2 | 30 | 234 | 987 | 1641 | 5.961 |

这 250 个结构的 AO、primitive、auxiliary 和 packed DF 估算全部经逐条复核。
全部 packed DF 缓存裸数据约 304.810 GiB，不能整体常驻单张 GPU。

## 原子贡献和尺度

| 元素 | Contracted AO | Primitive AO | Auxiliary AO |
|---|---:|---:|---:|
| H | 4 | 12 | 18 |
| C | 10 | 45 | 75 |
| N、O、F | 10 | 45 | 77 |

H 使用一个 6-primitive s 和一个 2-primitive p 收缩：AO 为 1＋3，primitive AO 为 6＋2×3。
重原子使用 core 6s、valence 6s6p、polarization 3d：AO 为 1＋1＋3＋5，
primitive AO 为 6＋6＋6×3＋3×5。辅助 AO 来自当前 snapshot 的实际壳层计数。

令 H 原子数为 h，重原子数为 a，则 N=4h＋10a、P=12h＋45a。
Q=18h＋75nC＋77(nN＋nO＋nF)。固定组成比例下，N、P、Q 均随原子数线性增长。
DF metric 的数值 rank 裁剪可使实际 Q 小于该上界。

## 源码明确出现的数组

所有公式单位均为 byte，使用 8-byte float64，1 GiB=2^30 byte。

- Primitive packed DF：`8 Q P(P+1)/2`。当前 mmap 只限制主机缓存常驻；
  `project_factors()` 仍将当前结构的整个 packed 数组转换到设备。
- Contracted DF：`8 Q N²`，当前输出为完整 Q×N×N 因子。
- Projection 每次最多展开 b=min(64,Q) 行，primitive 方阵为 `8 b P²`；
  可能的混合收缩中间数组为 `8 b P N`。每个 block 使用 rematerialization，
  避免反向保存全部 Q×P×P 展开张量，但不是对 packed 输入本身做设备流式读取。
- 前向 occupied-orbital exchange 中间数组为 `8 Q N nocc`。
  能量重建／raw density implicit action 的通用 JK 还存在 Q×N×N 中间数组，
  单份尺寸为 `8 Q N²`。
- Implicit GMRES 求解的状态是完整 N×N density，restart=40 时 Krylov 基通常含
  41 个长度 N² 的向量，即约 `8×41 N²`；Hessenberg 仅约 `8×40×41`。
  它没有显式构造 N²×N² Jacobian，也不按 max_iter=100 同时保存 100 份 Krylov 基。

GMRES 的 Krylov 存储为 O(N²)，当前最大 AO 的基约 17 MiB。
固定组成下，primitive 和 contracted DF 为 O(natoms³)，64-row 展开和 Krylov 基为 O(natoms²)。
当前主要容量负担是固定 primitive DF 和相关收缩／JK／AD 工作数组。
SCF 对角化和投影、JK 的计算耗时也会随尺寸快速增长，显存够用不能直接推出训练效率。

## 45／60／80／100 原子情景

以下是组成比例的算术情景，不是已验证分子的几何或训练结果。
H-rich 接近 ADIM6/AD6 比例；heavy-rich 接近 S22/07 比例，按整数调整并保持偶电子。
均假定辅助 metric 保留完整 rank。

| 组成情景 | 原子数 | N | P | Q | Packed DF / GiB | Contracted DF / GiB | 64-row 展开 / GiB | GMRES 基 / GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H-rich | 45 | 258 | 969 | 1551 | 5.431 | 0.769 | 0.448 | 0.020 |
| H-rich | 60 | 348 | 1314 | 2106 | 13.556 | 1.900 | 0.823 | 0.037 |
| H-rich | 80 | 464 | 1752 | 2808 | 32.127 | 4.504 | 1.464 | 0.066 |
| H-rich | 100 | 580 | 2190 | 3510 | 62.741 | 8.797 | 2.287 | 0.103 |
| Heavy-rich | 45 | 354 | 1497 | 2491 | 20.810 | 2.326 | 1.069 | 0.038 |
| Heavy-rich | 60 | 468 | 1974 | 3282 | 47.666 | 5.356 | 1.858 | 0.067 |
| Heavy-rich | 80 | 626 | 2643 | 4395 | 114.414 | 12.832 | 3.331 | 0.120 |
| Heavy-rich | 100 | 778 | 3279 | 5451 | 218.399 | 24.582 | 5.127 | 0.185 |

同样 80 原子，heavy-rich 的 primitive packed DF 是 H-rich 的约 3.56 倍。
不能只用原子数决定是否可训练；应先计算元素组成给出的 P、Q 和缓存字节数。

## A100 80GB 级别的判断边界

若暂以 80 GiB×0.9≈72 GiB 作为分析预算：45 原子两个情景及 60 原子 H-rich
具有明显的数据余量，可作为逐步测峰值的候选；60 原子 heavy-rich 和 80 原子 H-rich
需要实测完整 forward＋implicit backward。100 原子 H-rich 的 packed＋contracted 数据
已约 71.539 GiB，尚未包括投影 block、JK 中间数组、AD、CUDA 工作区和编译缓存，
不能据此承诺单卡训练。80／100 原子 heavy-rich 的 full-rank packed 输入本身已经超过
80GB 级卡的容量，当前整数组上卡路径不适合这些情景。

72 GiB 仅是算术假设，实际设备总字节数、GiB/GB 单位、可用空闲内存和 allocator 限额
均需读取后确认。各个数组有自己的存活区间，XLA 可融合、别名复用或重算；表中尺寸及
数组之和均不等于实测 GPU 峰值。还需计入多结构编译累计内存和其他并发进程。
本分析没有测量任何 A100 的训练峰值，不能给出已经验证的最大可训练原子数。
