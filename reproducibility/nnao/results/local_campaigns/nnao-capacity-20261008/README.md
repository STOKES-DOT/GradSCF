# 当前 NNAO / RHF 的容量估计

日期：2026-10-08。本地 CPU 抽象编译与实际 A100 GPU 探针是两种证据，数值不能互相替代。

## 当前最大体系与规划范围

当前250条结构中，原子数最多为ADIM6/AD6（C12H28，40个原子），SCF Hamiltonian为232×232，primitive AO=876，auxiliary AO=1404。40是本轮筛选上限，不是硬件上限。最大contracted AO为236；最大的primitive DF缓存反而是S22/07（C10H11N7O2，30原子），P=987、AO=234、Q=1641，packed输入5.961GiB。

在当前CHNOF、RHF/DF、float64、仅收缩系数训练且implicit一阶能量梯度的条件下，单张A10080GB的保守规划量级为：富氢烷烃约60个总原子，普通C/H比例约1的有机体系约45–50个，S22/07这类重原子比例高的体系约40–43个。65个富氢原子或45个重原子比例高的总原子属于需要实际测峰值的边界。这里的原子数全部包含H。

这些范围不是已验证的最大可训练分子。已验证的结构仍为这批≤40原子体系；更大的尺寸fixture没有真实积分、SCF、Adam优化或数值稳定性结果。

## CPU 抽象编译

使用当前 `MethaneRHF._implicit_value`，RHF/DF、float64、球谐 `szp663_direct/core6`、辅助基 `def2-universal-jkfit`，implicit tolerance=1e-8、rescue level shift=.2。
本地 JAX 0.8.1 / cpu / Python 3.12.2 / macOS-26.6.2-arm64-arm-64bit。

`ShapeDtypeStruct` 仅描述 S/H/DF 数组形状；调用 `lower().compile().memory_analysis()`，从未执行编译后的 SCF，也未计算 native 积分或创建真实巨型积分数组。每点及时清除 JAX 编译缓存并 GC。8 点全部编译成功，最大实进程累计 RSS 约 2.31 GiB。

总计为 `(argument + temporary + output - alias) / 2^30`。它描述该 CPU XLA 实现的分配估计，不是 GPU 的 live、pool 或 nvidia-smi 占用。代码、CUDA workspace、MACE/Adam、其他体系编译缓存、allocator 碎片和 OS I/O 均不包括在此总计中。

| 体系 | 类型 | 原子数 | 组成 | P | AO | Q | Args/GiB | Temp/GiB | 总计/GiB |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| S22/07 | 原始结构 | 30 | C10H11N7O2 | 987 | 234 | 1641 | 5.98 | 12.25 | 18.23 |
| ADIM6/AD6 | 原始结构 | 40 | C12H28 | 876 | 232 | 1404 | 4.03 | 10.55 | 14.58 |
| MCONF/21 | 原始结构 | 33 | C13H16N2O2 | 957 | 234 | 1571 | 5.38 | 11.62 | 17.00 |
| fixture_C16H34_50atoms | 尺寸fixture | 50 | C16H34 | 1128 | 296 | 1812 | 8.62 | 20.77 | 29.38 |
| fixture_C19H40_59atoms | 尺寸fixture | 59 | C19H40 | 1335 | 350 | 2145 | 14.28 | 32.68 | 46.96 |
| fixture_C15H16N10O4_45atoms | 尺寸fixture | 45 | C15H16N10O4 | 1497 | 354 | 2491 | 20.84 | 39.96 | 60.80 |
| fixture_C17H18N12O3_50atoms | 尺寸fixture | 50 | C17H18N12O3 | 1656 | 392 | 2754 | 28.19 | 50.93 | 79.12 |
| fixture_C26H54_80atoms | 尺寸fixture | 80 | C26H54 | 1818 | 476 | 2922 | 36.05 | 82.00 | 118.05 |

P 为 primitive AO，AO 为 contracted AO，Q 假定辅助 metric 保留完整 rank。原始结构的三个维度已与 selection 元数据核对。Synthetic fixture 仅规定组成和布局尺寸；坐标是分离格点，不能视为有效化学结构、能量、SCF 收敛或训练结果。

## 实际 A100 证据

独立 GPU 探针 [gpu-93809.json](gpu-93809.json) 使用 NVIDIA A100 80GB PCIe / JAX0.10，执行正式 `trainer.make_evaluators` 的 S22/07 MACE+implicit gradient 三次，全部有限且 SCF/原始 fixed point 检查通过。它只验证了这个30原子实例，未验证更大的fixture。

- JAX `peak_bytes_in_use`：12.082 GiB。
- JAX `peak_pool_bytes`：20.127 GiB。
- 自有进程 nvidia-smi 0.2 s 采样峰值：20.660 GiB；采样可遗漏瞬时峰值。
- 实际 allocator `bytes_limit`：71.327 GiB，memory fraction=.90，preallocate=false。
- 首次含编译评估 193.41 s；随后两次 2.78/2.75 s。这不是跨体系或跨epoch的加速benchmark。

CPU 对同一实例的18.23 GiB总计不等于GPU的12.08 GiB live或20.66 GiB采样占用。因此不得凭CPU heavy50的79.12 GiB断定其必然GPU OOM，也不得据CPU低于80 GiB宣布能安全训练。

## 如何使用这些结果

元素比例与基组配置决定 P/Q，固定比例下 packed DF `8 Q P(P+1)/2` 近似按原子数三次增长。现有40原子ADIM6/AD6比30原子S22/07反而更小，说明单看原子数不够。

CPU尺寸点把更大体系的风险量级显示出来；A100的规划应以已测GPU pool/采样峰值为锚、保留额外编译和碎片余量，再用更大真实结构进行forward+implicit backward实测。当前只能给待验证的规划区间，不能给已经确认的最大原子数。80 H-rich的CPU估计118.05 GiB是风险提示，并非GPU实测。

全部精确字节、输入/源码哈希、编译耗时、版本、失败状态和假设见 [results.json](results.json)。原默认80点编译为优先45/50边界而中止，未产生容量结果；后按新优先级重新编译80点成功。未执行2×S22/07或100原子点。

### 按GPU实测锚外推

以S22/07的BFC pool高水位20.127GiB为保守锚，按packed primitive DF字节数的比例估计：

`estimated_pool = 20.127 GiB × [Q P(P+1)] / [1641 × 987 × 988]`。

这是单点校准的尺寸外推，不是多点拟合的GPU内存模型。它忽略其他数组的不同尺度以及BFC分配粒度、碎片、CUDA工作区和跨体系编译缓存。以下预测不等于硬上限；规划尽量保留约10GiB以上的分配器余量。

| 组成／情景 | 总原子数 | AO | Primitive AO | 预测pool/GiB | 证据性质 |
| --- | ---: | ---: | ---: | ---: | --- |
| S22/07 | 30 | 234 | 987 | 20.13 | 实测pool高水位 |
| C16H34 | 50 | 296 | 1128 | 29.02 | 尺寸外推，未执行更大分子SCF |
| C19H40 | 59 | 350 | 1335 | 48.12 | 同上 |
| C20H42 | 62 | 368 | 1404 | 55.97 | 同上 |
| C21H44 | 65 | 386 | 1473 | 64.64 | 接近规划边界，需实测 |
| C22H46 | 68 | 404 | 1542 | 74.16 | 预测超出当前71.33GiB allocator限额 |
| S22相同组成比例连续外推 | 40 | 约312 | 约1316 | 47.70 | 不代表具体化学结构 |
| S22相同组成比例连续外推 | 43 | 约335 | 约1415 | 59.25 | 同上 |
| S22相同组成比例连续外推 | 45 | 约351 | 约1481 | 67.91 | 余量很小，需实测 |

45原子的整数fixture C15H16N10O4比S22连续比例多了少量重原子，实际P=1497/Q=2491，按同一模型预测约70.26GiB，因此不应把“45原子”当成通用安全范围。

当前64-row分块和checkpoint限制的是展开矩阵及反向中间量；当前primitive packed DF输入仍整块进入GPU。若目标扩展到100个以上常见有机分子原子，应先研究primitive DF输入的设备流式处理或更直接的收缩基组积分求导，保留float64与implicit一致性。该优化本轮未实施。

本报告不估计DFT网格、force/Hessian高阶导数或更多zeta基组的容量。80GB为[NVIDIA产品规格](https://www.nvidia.com/en-us/data-center/a100/)；实际GiB预算以本次device memory_stats为准。关闭预分配后仍有allocator池缓存与碎片，参见[JAX显存说明](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)。

GPU探针93809从4CPU申请调为2CPU后运行，32GiB主存、10分钟限时；原OMP/MKL/OPENBLAS线程设置仍为4。计时只用于说明同一体系的冷／热评估，包含每次mmap输入传输；不计探针前的缓存哈希核验和模型构建，也没有执行Adam更新。3次同一初始权重的评估能量一致，参数梯度均有限。

## 复现

从仓库根目录执行以下命令。它只抽象编译列出的8个case；不写ERI/DF，也不执行SCF。不同JAX/编译后端可产生不同内存估计。

```bash
PYTHONPATH=src /opt/anaconda3/bin/python \
  artifacts/nnao-capacity-20261008/estimate_capacity.py \
  --output artifacts/nnao-capacity-20261008/results-reproduced.json \
  --cases S22/07 \
  ADIM6/AD6 \
  MCONF/21 \
  fixture_C16H34_50atoms \
  fixture_C19H40_59atoms \
  fixture_C15H16N10O4_45atoms \
  fixture_C17H18N12O3_50atoms \
  fixture_C26H54_80atoms
```

`--append --cases ...` 可以追加相同scientific source/config下的少量尺寸点；科学配置或已记录源码哈希变化会拒绝追加。GPU探针使用独立Slurm allocation，不能在登录节点执行。
