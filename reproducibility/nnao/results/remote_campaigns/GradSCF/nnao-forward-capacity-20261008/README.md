# 只做 Forward 的 NNAO／RHF 容量估计

2026-10-08。本报告包含两类不同证据：训练缓存路线的CPU抽象编译，以及冻结MACE输出后直接计算native收缩积分的真实CPU前向验证。两者都不计算梯度。基组为球谐 `szp663_direct/core6`，辅助基 `def2-universal-jkfit`，float64，rescue=.2，保留实际SCF与raw固定点诊断。

7个点全部编译成功，每点约3–7秒，最大进程累计RSS约 1.39 GiB。巨大 S/H/DF 输入仅以 `ShapeDtypeStruct` 表示。Synthetic fixture 是组成与布局尺寸，不是计算出的真实分子结果。

## 已有缓存时的 Forward

下表单位为GiB。总计为 `argument + temporary + output - alias` 的CPU XLA分配估计；它不等于实际CPU RSS或GPU峰值。

| 体系／fixture | 原子数 | P | AO N | Q | Args | Temp | Forward总计 | 以前E+grad总计 | 首次DF至少两份payload |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S22/07 | 30 | 987 | 234 | 1641 | 5.98 | 2.68 | 8.66 | 18.23 | 11.92 |
| ADIM6/AD6 | 40 | 876 | 232 | 1404 | 4.03 | 2.25 | 6.28 | 14.58 | 8.04 |
| fixture_C26H54_80atoms | 80 | 1818 | 476 | 2922 | 36.05 | 19.74 | 55.79 | 118.05 | 71.99 |
| fixture_C17H18N12O3_50atoms | 50 | 1656 | 392 | 2754 | 28.19 | 12.62 | 40.81 | 79.12 | 56.30 |
| fixture_C19H40_59atoms | 59 | 1335 | 350 | 2145 | 14.28 | 7.84 | 22.11 | 46.96 | 28.50 |
| fixture_2xS22_07_60atoms | 60 | 1974 | 468 | 3282 | 47.72 | 21.43 | 69.16 | 未测 | 95.33 |
| fixture_C33H68_101atoms | 101 | 2301 | 602 | 3699 | 73.07 | 39.96 | 113.03 | 未测 | 145.98 |

相比同形状的CPU E+implicit-gradient，forward总计约降至43–52%，但这不是速度倍数。省去反向响应和AD工作数组后，primitive packed输入 `8 Q P(P+1)/2` 仍在。80H fixture的CPU forward估计55.79 GiB，而101H的packed输入本身约72.99 GiB；后者已超过实际A100探针的71.327 GiB allocator限额，在当前整数组上卡路线下不能仅靠省backward解决。

## CPU RAM 与首次生成缓存

这里假定固定积分已经存在。`AuxiliaryPlan.factors()` 首次生成时，native三中心 `three` 与 `solve_triangular` 生成的whitened factors至少同时存在两份同尺寸payload；辅助metric还需 `8 Q²`。从SciPy结果转换为JAX数组、C/F-order布局变化或native输出缓冲还可能带来额外副本。因此本表最后一列只给最低两份payload量级，不能当完整RSS峰值。

例如80H fixture：已有缓存forward逻辑分配55.79 GiB，但首次生成至少两份DF约71.99 GiB；101H对应113.03 GiB与145.98 GiB，尚未计第三份转换和BLAS工作区。CPU高内存节点在空间上可能覆盖更大规模，但是否能在具体128/256 GiB机器运行，需要实际RSS、I/O与耗时验证。

Mmap是文件映射，OS可回收页，不代表JAX一定零拷贝或所有输入页始终不驻留。CPU device-put可能受数组布局／对齐影响而复制输入，运算也可能触及大量映射页；即使某个buffer能够alias，RAM、页缓存、BLAS/native workspace和输出仍需预算。不要把CPU编译总计直接当实机RSS。

## 纯Forward还有更小的现有路线

本估计特意复用训练用primitive缓存，方便与backward匹配。若只需forward，NN输出系数已经确定、不需要对系数反传，已有 native `AuxiliaryPlan` 可以直接用 **contracted topology＋bound coefficients** 构建 `Q×N(N+1)/2` 三中心／DF，而不必先保存 `Q×P(P+1)/2`。

101H fixture的P2301/N602意味着单纯packed输入理论可从约72.99 GiB降至约5.00 GiB（约14.6倍）；仍有 `Q×N×N` 因子、JK、SCF、native与布局工作区。该路线现已在下面的CH4和S22/07上数值验证；百原子尺寸仍只有数组分析，不能据此宣称已成功计算。

CPU是否够用，还要看首次DF生成和之后SCF的实际耗时；forward省内存不等于算法耗时可忽略。尤其当前 primitive projection、三中心积分和白化工作量随P/Q增大很快。

## 实际CPU前向：MACE＋native contracted DF＋RHF

作业93873在jy_gpu_2的n001节点运行，8CPU、32GiB主存、10分钟限时，实际线程／affinity均为8，JAX/JAXlib0.10.0，CPU float64。使用与A100探针相同的7171参数seed0/zero-init模型；没有加载训练后的检查点，不构建或读取primitive积分缓存，也没有任何gradient、PySCF调用或完整ERI。

S22/07：30原子、234AO、1641auxiliary，3次真实模型预测及3次SCF全部通过；能量均为−914.277895146754Ha，与此前A100 primitive-route参考相差4.55e-13Ha。原始固定点残差1.24e-13，轨道残差7.40e-13，44cycles、未触发rescue。

| 步骤／指标 | 实测 |
| --- | ---: |
| MACE首调用（含编译） | 0.969s |
| MACE两次热态预测 | 1.87／2.10ms |
| Native S/H/DF首次生成 | 14.84s |
| SCF首调用（含编译） | 29.79s |
| SCF两次热态调用 | 28.14／27.92s |
| 进程整体峰值RSS | 5.464GiB |
| Contracted packed DF | 0.336GiB |
| Contracted dense DF | 0.669GiB |

大DF是显式JIT输入，计时同步等待结果完成；原始记录包含源文件／native library哈希、依赖版本和所有收敛诊断：[cpu-93873.json](cpu-93873.json)。每次新几何／新预测系数要重新生成对应积分；这里的SCF热态时间重复同一个固定输入，不代表不同分子的统一耗时。JAX计时需区分编译、数据传输与异步执行，见[官方benchmark说明](https://docs.jax.dev/en/latest/benchmarking.html)。

CH4另用seed0、0.02活跃系数扰动验证native contracted与primitive projected DF的等价性：S最大误差4.44e-16、H2.13e-14、DF factors2.99e-13、SCF能量1.42e-14Ha，两条11cycles收敛。证据为[contracted_forward_check.json](contracted_forward_check.json)。它验证了冻结系数的组装等价性，并没有运行MACE本体。

当前已有底层API支持 `layout.bind(outputs) → make_plan(contracted topology, native) → contracted auxiliary DF → HF-configured RKS`。生产训练工具继续保留primitive路线；没有新增自动选择的NNAO部署接口。

## 更大CPU推理体系的数组量级

以下为CHNOF组成情景、当前同一基组／辅助基的full-rank数组算术，未计算这些大体系的能量：

| 情景 | 总原子数（含H） | AO | Auxiliary | Native packed DF/GiB | Dense DF/GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| C33H68，富氢 | 101 | 602 | 3699 | 5.00 | 9.99 |
| C33H37N23O7，重原子比例较高 | 100 | 778 | 5451 | 12.31 | 24.58 |
| C66H134，富氢 | 200 | 1196 | 7362 | 39.26 | 78.46 |
| C50H55N35O10，重原子比例较高 | 150 | 1170 | 8205 | 41.88 | 83.68 |

Packed与dense之和不是峰值RSS。首次白化、JAX转换、unpack、J/K等均可能同时保留额外副本。CPU128GiB可以把约100个总原子作为下一轮验证目标；CPU256GiB可考虑更大的150–200原子情景，富氢与重原子比例高的预算不同。它们是测试目标而非已证实上限，耗时、SCF稳定性和实际RSS仍须测量。

本报告只讨论固定几何的能量forward。若部署时需要force／几何优化，冻结网络权重并不允许忽略 `coefficients(R)` 的基组响应，必须保留相应导数，不能直接套用无反传容量。

## 证据与复现

精确字节、源文件哈希、版本、编译耗时及失败状态在 [results.json](results.json)。对应训练梯度CPU估计见 [先前报告](../nnao-capacity-20261008/README.md)；真实A100探针是E+gradient，不能当作本轮forward峰值测量。

从仓库根目录执行（不执行科学迭代）：

```bash
PYTHONPATH=src /opt/anaconda3/bin/python \
  artifacts/nnao-forward-capacity-20261008/estimate_forward.py \
  --output artifacts/nnao-forward-capacity-20261008/results-reproduced.json \
  --cases S22/07 \
  ADIM6/AD6 \
  fixture_C26H54_80atoms \
  fixture_C17H18N12O3_50atoms \
  fixture_C19H40_59atoms \
  fixture_2xS22_07_60atoms \
  fixture_C33H68_101atoms
```

本轮未改生产代码或正式训练作业；另提交了上述短CPU验证。范围不包括DFT网格、梯度/力、TDDFT或其他后HF方法。
