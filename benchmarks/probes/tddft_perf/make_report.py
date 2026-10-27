"""Build the narrative from completed records; does not run scientific code."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics


parser=argparse.ArgumentParser()
parser.add_argument('directory',type=Path)
root=parser.parse_args().directory
runs=root/'runs'


def rows(name):
    return [json.loads(s) for s in (runs/name/'measurements.jsonl').read_text().splitlines()]


def values(name,stage,batch=None):
    selected=[x for x in rows(name) if x['stage']==stage and x['wall_seconds'] is not None
              and (batch is None or x.get('batch_size')==batch)]
    warm=('warm' in stage or stage in {'shared_gradscf_action','shared_pyscf_action'})
    if not warm and any(x['repeat']>0 for x in selected):selected=[x for x in selected if x['repeat']>0]
    return [x['wall_seconds'] for x in selected]


def med(name,stage,batch=None):return statistics.median(values(name,stage,batch))
def rss(name):return json.loads((runs/name/'process.json').read_text())['cpu_peak_rss_bytes']/1e6
def evidence(name):return f'`runs/{name}/measurements.jsonl`'
def line(name,stage):return next(x for x in reversed(rows(name)) if x['stage']==stage)


environment=json.loads((root/'environment.json').read_text())
cpu=next(s.split(':',1)[1].strip() for s in environment['cpuinfo'].splitlines() if s.startswith('model name'))
comparisons=json.loads((root/'comparisons.json').read_text())
baseline='| 体系 / 方法 | GradSCF warm / s | PySCF warm / s | 最大能量差 / eV | time_ratio | 验收 |\n|---|---:|---:|---:|---:|---|\n'
for item in sorted(comparisons,key=lambda x:x['case_id']):
    ratio='—' if item['time_ratio'] is None else f"{item['time_ratio']:.3f}"
    setting=json.loads((runs/item['gradscf_run']/'input.json').read_text())
    baseline+=f"| {item['case_id']} / CPU{setting['threads']} | {item['gradscf_warm_seconds']:.6f} | {item['pyscf_warm_seconds']:.6f} | {item['max_excitation_error_eV']:.2e} | {ratio} | {'通过' if item['excitation_gate_passed'] else '参考残差未通过'} |\n"
compiled=json.loads((root/'compile_counts.json').read_text())
warm_compiles=[x for x in compiled if not x['evidence'].endswith('profile-0.pstats')]
compile_med=statistics.median(x['backend_compile_cumulative_seconds'] for x in warm_compiles)
profile_wall=med('water-pbe-tda-profile-cpu1','end_to_end')
fraction=compile_med/profile_wall
shared='| 同一轨道 / 网格；batch=5 | GradSCF / s | PySCF / s | 最大 action 绝对差 |\n|---|---:|---:|---:|\n'
for key in ['pbe-tda','pbe0-tda','pbe0-full']:
    name='shared-water-'+key+'-cpu1-repaired'
    shared+=f"| {key} | {med(name,'shared_gradscf_action'):.6f} | {med(name,'shared_pyscf_action'):.6f} | {line(name,'shared_action_validation')['action_max_abs_error']:.2e} |\n"
original='water-pbe0-packed-original-cpu1-end_to_end'
candidate='water-pbe0-packed-packed-public-cpu1-end_to_end'
n90='packed-n90-original-cpu1-clean-memory'
n90b='packed-n90-packed-public-cpu1-clean-memory'
gpu='packed-n60-original-gpu8-clean-memory'
gpub='packed-n60-packed-public-gpu8-clean-memory'
op='water-pbe0-packed-original-cpu1-action'
opb='water-pbe0-packed-packed-public-cpu1-action'
nto_name='ch2o-pbe0-tda-nto-cpu1'
nto=line(nto_name,'nto_metadata')
nto_table='| 向量 | ‖X−Xr‖/‖X‖≤1e−4 所需 rank | rank=1 的 K-action 相对误差 | SVD / μs |\n|---|---:|---:|---:|\n'
for row in rows(nto_name):
    if row['stage']=='nto_offline' and row['rank']==1:
        nto_table+=f"| {row['sample']} | {row['ranks_by_x_tolerance']['0.0001']} | {row['k_relative_error']:.4f} | {row['svd_seconds']*1e6:.1f} |\n"
stages='ch2o-stage-costs-cpu1'
stage_table='| 独立微基准 | warm 中位数 / s |\n|---|---:|\n'
for stage in ['one_electron','eri_s4','ri_2c3c','ri_factors_complete','xc_hessian_fresh_reference','xc_hessian_same_reference','ao_df_jk_warm']:
    stage_table+=f'| {stage} | {med(stages,stage):.6f} |\n'
e0=line(original,'end_to_end')['excitations_ha'];e1=line(candidate,'end_to_end')['excitations_ha']
energy_diff=max(abs(a-b) for a,b in zip(e0,e1))
window=json.loads((root/'validation-window.json').read_text())
workspace=json.loads((root/'original_workspace_verification.json').read_text())
untracked_changes='\n'.join('- `'+name+'`' for name in workspace['untracked_changed'])
text=f'''# GradSCF TDA / TDDFT 性能探针：33a7cab

## 1. 基线、版本与边界

本轮实际运行的是 `33a7cab3473a123f027877e9e7e51a3d86f75fd8`，原工作区
`release/v1.0.0` 的 tracked 源码无 dirty patch。新 worktree 为
`/Users/jiaoyuan/.codex/worktrees/6583/GradSCF`，分支
`probe/tddft-perf-20261027`。未修改任何 `src/` 生产文件，三个实验均只在隔离
子进程安装明确标记的 monkeypatch；可审阅补丁位于 `experiments/`。

远端使用 `jy_gpu_2` 的 n005、{cpu}、一张 A800、64 个 Slurm CPU 和 512 GiB。
5090 当时全部占用，V100 节点总内存不足 512 GiB；采用先前获授权的 A800。
n005 强制最长 8 小时，未绕过该限制。作业 `110038` 已释放并 COMPLETED。
本项目重型计算串行；CPU1 affinity 为一个逻辑 CPU，CPU8 affinity 为
`[0,1,2,3,64,65,66,67]`，对应 **4 个物理核的 8 个逻辑 CPU**，不能称为 8 个物理核。
节点并非独占。参考 Slurm `allocation-final.txt`、环境与逐进程 affinity。
最终参考 worker 去除非数值导入后，另用 CPU-only 作业 110135（8 CPU / 16 GiB）
复测水 TDA/full 两个引擎；该作业 126 秒后 COMPLETED，未追加 GPU。

Python {environment['python'].split()[0]}，JAX/jaxlib {environment['versions']['jax']}，
NumPy {environment['versions']['numpy']}，SciPy {environment['versions']['scipy']}，
PySCF {environment['versions']['pyscf']}，jax-xc {environment['versions']['jax-xc']}。
libcint 为 6.1.3、固定源码 c72d30662ba08048940c9000cc352a220c867712。
新构建扩展 SHA256：`c4756801f7cad8aa826760fb5d761ba7ea6f345606c17402401cc8f4ac6b384b`。
实际导入位于新同步的 `gradscf-tddft-probe-20261027/source/src/`，1095 个生产源码
文件逐项核验均一致。使用现有 Python 依赖环境，未改变 editable 安装指向。
独立构建 exit=0，Release、diagnostics=OFF、GCC 8.5、LP64 `/lib64/libblas.so.3`、8 build jobs。
完整 BLAS/libxc/动态库哈希和设备信息见 `environment.json`、`build_configuration.json`。

正式验证时间窗约 {window['elapsed_seconds']:.1f} 秒，即 {window['elapsed_seconds']/60:.1f} 分钟，
小于预登记 7200 秒；构建与首次 smoke 单列。每个过程默认 600 秒、采样 64 GiB RSS guard，
不是 RLIMIT_AS。Slurm accounting 已禁用，无法提供 sacct 的历史 step MaxRSS。
保留控制器的实际分配/结束状态；其 DerivedExitCode 包含主动停止的大体系 step，
不能把作业正常结束解释为每个科学案例通过。

## 2. 原样基线与可比性

闭壳层采用实数 Cartesian def2-SVP，全电子，不冻结轨道，最低 5 个 singlet，
float64，grid level 2；PBE0 是 25% HF exchange + 75% PBE exchange + PBE correlation。
DF 为真正显式 `weigend` 辅助基的 2c/3c RI，未用 4c 谱分解冒充 RI。
SCF 为 hcore 初猜、最多 100 次、能量 1e−10 Ha、梯度 1e−8、GradSCF 密度 1e−9。
响应残差阈值 1e−8、最多 100 次，显式选 Davidson。软件 max_memory=4096 MB（十进制）。
网格等级/裁剪规则、AO 数、实际点数、能量均核对；未声称所有独立网格逐点相同。
严格共享轨道和网格的 action 另列，不能替代端到端。

每个端到端过程为 1 次首次执行 + 3 次新对象 warm，包含独立 SCF、响应和振子强度。
不传入 PySCF 轨道给 GradSCF。warm 不复用已收敛激发态初猜。
JAX 输出同步后计时；进程启动/导入在 process.json 另计，cold 不等于完整 Python 启动。
action 的 lowering、compile、first execute 与至少 10 次 warm 分列。
详细 cProfile、debug callback、正确性 oracle 与 clean timing 分开。

{baseline}

time_ratio 定义为 GradSCF/PySCF。只有独立端到端任务、硬件/线程、AO/网格维度、
总能 1e−7 Ha、激发能 1e−4 eV、收敛标记都通过时才计算。MAD、全部重复样本、
cold 与峰值见 `summary.csv`，逐次结果见 `measurements.jsonl`。

实际规模：水 25 AO / 5 occupied / 20 virtual / 21952 grid；CH2O 为 40 / 8 / 32、
146 auxiliaries / 33160 grid；苯 120 / 21 / 99、438 auxiliaries / 99480 grid；
OH 为 20 AO、alpha/beta 5/4 occupied、139 个总跃迁。

苯 GradSCF CPU1：SCF 首次 33.154 s，首次 TDA 未在 600 s 内完成，峰值 RSS
{rss('benzene-pbe0-tda-gradscf-cpu1-clean-e2e'):.1f} MB，未触及 RSS guard。
停止该规模的求解升级，保留 PySCF 当时的部分记录与主动停止原因，**无速度比**。
其独立准备后 action 已运行，batch=5 约
{med('benzene-pbe0-tda-gradscf-cpu1-action','action_warm',5):.3f} s，PySCF 约
{med('benzene-pbe0-tda-pyscf-cpu1-action','action_warm',5):.3f} s；两者的独立轨道输入
不是严格共享输入，不能据此把全部 timeout 归因于编译。

OH GradSCF TDA 已收敛，warm 总时间
{med('oh-pbe-uks-tda-gradscf-cpu1-clean-e2e-corrected-harness','end_to_end'):.3f} s。
PySCF 在相同 hcore/100-cycle/阈值配置下 SCF 未收敛，未偷偷换初猜、Newton 或放宽阈值；
该外部 U-TDA 对照未完成。其 GradSCF action 与自旋/XC 路径已实际执行。

CH2O 的 PySCF TDA 部分残差为 3.1e−8–6.8e−8，full TDDFT 某根约 2.38e−7；
虽能量很近，仍排除速度比。诊断中赋值 `td.max_space=40` **在 PySCF 2.13 不生效**：
该版本 kernel 只把 max_memory 传给 `_lr_eig`，后者自行规划子空间。
两次相同结果不能证明有效改变子空间无影响。此限制已核对当前安装源码。
未继续改变 lindep 或容差制造一个“通过”的比较。

## 3. 已确认瓶颈

### 3.1 新对象仍重复编译，不能归因于 AD 本身

水/PBE-TDA 的独立 profiling warm 每次 **4 次 backend_compile_and_load**，
中位累计 {compile_med:.3f} s，诊断总时间中位 {profile_wall:.3f} s，约 {fraction:.1%}。
父栈落在 response → solve_hermitian → solve_ritz → fori_loop/scan。
四次调用与新建 action/solver 闭包一致，但尚未逐个隔离其 cache-key 来源。
证据：`compile_counts.json`、`runs/water-pbe-tda-profile-cpu1/profile-1.pstats` 等。
同进程输入形状一致仍编译是实测；FXC 数值缓存和编译缓存需区别处理。
若仅在该 profiling 条件下理想移除这些编译，Amdahl 上限约
{1/(1-fraction):.2f}×；这不是 clean 端到端的承诺或跨体系预测。

严格共享输入 action 结果说明，准备完成后的核心作用不普遍慢于参考程序：

{shared}

这三例使用同一个 GradSCF SCF 轨道/能量/密度和逐点网格，让 PySCF 单独计算其 J/K/XC。
输入生成和转换另计；随机 X/Y 相互独立、nocc≠nvir、非对称 transition density，
batch=5。GradSCF 有预编译准备后 action，PySCF 为其 gen_vind 原生路径，
这是相同数学作用、不同内部准备策略的内核比较，不是几何起点速度比。
最终水端到端表采用 `final-clean` 复测：PySCF 进程明确未导入 JAX/GradSCF。
早期 PySCF worker 的 post-timing provenance 导入污染其整体 RSS，原始值保留，
不用于纯 PySCF 内存比较；它没有提供 GradSCF 轨道/积分，也不改变数值任务。

### 3.2 packed exchange 的 N^4 临时区有编译后证据

TD 私有 `_jk_from_eri_pair_matrix` 的 vmap gather，与公共
`integrals.molecular.jk.build_jk_from_packed` 重复。dynamic s4 数据和相同非对称密度：

| CPU1，90 AO，batch=5 | 原实现 | 公共分块实现 |
|---|---:|---:|
| clean warm action / s | {med(n90,'packed_jk_warm'):.6f} | {med(n90b,'packed_jk_warm'):.6f} |
| XLA temp estimate / bytes | 525043800 | 6160624 |
| sampled whole-process peak RSS / MB | {rss(n90):.1f} | {rss(n90b):.1f} |

这不是只从 vmap 源码推断：编译器 temp 与 N^4×8 大小相符，独立 clean-memory
RSS 也支持实际工作区差异。这里是合法对称 s4 的合成积分，未冒充分子 scaling。
25/60/90 AO 使用同一构造系列；只证明这些尺寸，未拟合渐近复杂度。

A800 上 60 AO 的 warm 原实现 {med(gpu,'packed_jk_warm'):.6f} s、分块
{med(gpub,'packed_jk_warm'):.6f} s，后者约
{med(gpub,'packed_jk_warm')/med(gpu,'packed_jk_warm'):.2f}× 慢。
allocator lifetime peak bytes-in-use 为 205809408 → 138558720；pool peak 为
335544320 → 268435456。它们不是 NVML 物理 VRAM 峰值；reserved=0 的插件字段
也不表示整个 GPU 没有保留内存。未测 GPU AD。

### 3.3 准备和设备边界

以下为 CH2O 的独立、各自同步的微基准，包含 Python binding/JAX dispatch，
**不是 C 内核独占时间，也不能将父子阶段简单相加**：

{stage_table}

相同 reference 的 XC Hessian 缓存已有效；新 reference 的约 0.25 s 准备有成本，
远小于水的重复编译。此数据不支持立刻大改 native driver。

原 GPU RI 入口实际报 `No FFI handler registered for gradscf_compact_cpu_v1 ... CUDA`。
未修改生产源码修复它。另测显式 CPU 积分/SCF → GPU response，包含独立参考态与传输：
CH2O CPU8 warm 总时间 {med('ch2o-pbe0-tda-gradscf-cpu8-end_to_end','end_to_end'):.6f} s；
混合路径 {med('ch2o-pbe0-tda-cpu-reference-gpu8-end_to_end','end_to_end'):.6f} s。
混合路径 batch=5 warm action 为
{med('ch2o-pbe0-tda-cpu-reference-gpu8-action','action_warm',5):.6f} s。
这些小体系数据表明，热态 action 很快不等于完整工作流明显受益。
`cpu_reference_to_gpu` 单列同步时间与逻辑传输字节估计，未取得 PCIe 总线计数；
CPU libcint 不被称作 GPU native 积分。

## 4. 三个局部实验与数值/AD 验收

1. **packed-public**：只替换 TD 私有 packed J/K。CPU 内核明显改善，但水/PBE0
   端到端 {med(original,'end_to_end'):.6f} → {med(candidate,'end_to_end'):.6f} s；
   batch=5 分子 action {med(op,'action_warm',5):.6f} → {med(opb,'action_warm',5):.6f} s。
   小体系总时间差处于重复离散量级，不能声称有可靠的整体提速。
   激发能最大差 {energy_diff:.2e} Ha；共享输入 X/Y VJP 最大差 2.28e−13。
   H2/HF packed full-TD 几何导数在替换前后为 −0.512850418869904 Ha/bohr，
   相差约 1e−16，两个 FD 步长误差约 5.0e−9 / 1.3e−9 Ha/bohr。
2. **df-demand**：TDA 只准备 J-VO、K-OO、K-VV。nocc=11、nvir=49、naux=150，
   raw factors 故意非对称；不同 J/K factors、batch 和数组 VJP 均验收。
   outer JIT 中原实现的未用分支已 DCE，warm action 改善很小；eager 准备可少做工作。
   CH2O 分子前后与完整重复见 summary.csv，不把毫秒级准备节省夸成秒级求解改善。
3. **df-share**：仅 full TDDFT 同源 J/K 对象共享 OV/VO 变换，不假定 raw factor 对称。
   合成 outer JIT warm 约 0.01075 vs 0.01076 s；eager 准备约 0.01194 → 0.00883 s。
   编译器 CSE 已消除所测外层图的重复。最终 Qvv 存储仍存在，不能称作降低 Qv²。

未组合改变近似、虚轨道窗口、网格或停止阈值。原始/实验按独立进程保存。
原 microbenchmark 的数值 oracle 发生在计时之后，但污染 whole-process RSS；
因此另做四个 clean-memory 过程。早期 RSS 保留且不用于宣称内存改善。
所有 harness 修正和早期失败见 `harness_corrections.md` 与独立 diff。

## 5. AD / neural XC / NNAO 的实际范围

- 完整 H2/STO-3G、level-0 几何链：重建移动积分、原子中心网格、SCF 轨道、
  PBE TDA、PBE full TDDFT、四层全局 custom NN full TDDFT。两个步长
  2e−4 / 1e−4 bohr，FD 误差约 1e−8 / 2.5e−9；JVP/VJP 一致。
  目标是第二个原子 z 坐标方向的 dω/dR，包含完整物理依赖链；未构建 3N Jacobian，
  也未将激发能差的导数称作激发态绝对核力。
- PBE value 与 value_and_grad 的 warm 为约 0.0085 / 0.275 s；全局 NN fixture
  为约 0.0021 / 0.0042 s。它们是不同目标，不能用两者比值推断一般 AD 开销。
  这些小案例没有“AD 必然低效”的统一结论，也不能外推大分子梯度成本。
- 16 项现有窄回归通过：非对称独立 X/Y、参数 JVP/VJP、opaque clipped XC 的
  separate actions、跨网格 global HVP、legacy neural_xc 的 streamed projection /
  parameter VJP、两种 H2 几何的 NNAO coefficient/log-exponent 产品。
- small random MACE 的完整 DF-RHF 能量/optimizer step 已执行，含 trainable mask、
  参数更新、指数更新和 checkpoint 检查；`nnao-step.json` 记录四步及耗时。
  没有真实训练权重，不能称为真实 NNAO 模型精度验收。基组/SCF 为 float64；
  模型参数 dtype 未单独记录，不能进一步声称所有模型权重也是 float64。
- 未测试 NNAO 激发能训练、GPU AD、振子强度完整反传、高阶 Gaussian 指数导数或
  NNAO force-loss mixed derivative。当前显式辅助基 assembly 对几何 tracing
  仍明确拒绝；本轮几何验收走 exact direct/packed，而 NNAO RI 参数产品在固定几何。
  没有 stop_gradient 或旧积分器 fallback 补洞。

## 6. NTO 离线探针与 ISDF/THC

CH2O/PBE0/RI 实际捕获 {nto['actual_operator_calls']} 次运行时 action（含守卫/检查，
不能直接称作 Davidson 迭代数），保存早/中/晚期 trial、真实残差、预条件方向和收敛态。
原定义算子最终五根最大残差 {max(nto['original_operator_residual']):.2e}。
捕获用了 debug.callback，相关 solve 时间不进入正式速度表。

{nto_table}

rank-1 的 K 错误从约 1.5% 到 86%，不能仅凭轨道的视觉局域性认定足够。
早/中期和多个残差方向需要 full rank=8；收敛态也不统一低秩。
SVD 约 80–230 μs，精确 K 约 245–272 μs；低秩 K-only 时间不包括 SVD、转换与编译。
未做近似 eigensolve，因而不声称获得求解加速或 exact refinement。
依赖输入的截断会使 action 非线性，未把它放进普通 Hermitian LinearOperator。

实测 CH2O Qoo/Qov/Qvv 分别 {nto['qoo_bytes']} / {nto['qov_bytes']} / {nto['qvv_bytes']} bytes。
苯对应基本唯一块的理论容量约 43.2 MB，AO DF factors 约 50.5 MB，四通道 AO grid
约 382 MB；这些逻辑大小与 RSS/编译峰值不同。
示意 THC rank R=2*nao 时，MO 因子 + R² metric 可写为
8[R(nocc+nvir)+R²] bytes：CH2O 76800 bytes、苯 691200 bytes。
**R 的精度、拟合成本、导数与能否摊销均未测**，这些不是测得的存储方案或提速。
当前小体系的主要证据指向编译、明确的 packed 工作区和设备边界；不建议据此
直接开始完整 ISDF/THC 工程。GW/BSE/波函数方法仅可参考这些共享层结论，本轮未测试它们。

## 7. H1–H7 审计

| 假设 | 状态 | 当前证据与边界 |
|---|---|---|
| H1 私有 packed J/K / gather | confirmed | 25/60/90 AO 编译 temp、CPU/GPU clean memory、非对称密度与 VJP；不能归因苯的 RI timeout |
| H2 full-ERI 绕过公共 exchange | source_only | response.py:749 仍直接 einsum；未分配超过 2³¹ 元素的大 full ERI 验证索引/吞吐 |
| H3 eager 过量 DF 准备 | confirmed | TDA 的 need_b_terms 未传到底层，OV/VO 和同源 J/K 重复；eager 微基准与闭包容量 |
| H3 outer-JIT 重复计算 | disproved in tested graphs | dynamic raw-factor outer JIT 的 CSE/DCE 后时间/内存接近；不外推其他静态闭包或 batch |
| H4 X/Y 复用 | already_fixed (AO); source_only (DF 剩余) | restricted AO 已合并密度和线性 XC；U AO/XC 也有融合；DF full 保留四个 orientation action。opaque clipped callback 不强行合并 |
| H5 内存来源 | confirmed for measured points | packed N⁴ temp、Q 块、grid、compiler temp、实际 RSS/allocator 已分列；未发现所查 batch=1/5 action 中的 D×D 响应矩阵，不能当作全程序保证 |
| H6 重复准备/编译 | confirmed; cache partially effective | 同签名新对象 4 compiles；相同 reference Hessian 缓存有效。尚未隔离每个 compile 的完整 cache key |
| H7 统一 forward/backward 预算 | source_only gap | response_options 规划 forward batch/subspace，但 tda.py/unrestricted.py 调 solve_hermitian 未传 max_memory/reserved，RPA 亦需独立核对。未测大规模性质反传峰值 |

主要位置：`tddft/response.py:749,777,953,1322`，`lowrank_response.py:8`，
`_semilocal_response.py:35`，`response_options.py:172`，`tda.py:48`，
`unrestricted.py:1055`，`solvers/eigen/api.py:19`，`solvers/eigen/primal.py:22`。
默认 eigenvalue-only 与 eigenvector/振子强度响应不是相同输出契约。
primal Ritz 上的 stop_gradient 配合显式谱响应，并不表示冻结物理输入。

## 8. MOLGW：实际运行、有限对齐与失败

运行了现有参考 executable，SHA256
`bedee25b4d99844d9a59cb834a9a509dc0d276b48103aaa5a71da0ab105075bf`。
未使用 MOLGW_COMPARE_*、QP 文件、restart、比较专用可执行文件或源代码 patch。
只把共同 def2-SVP 基组**数据**转成其输入格式；不是 PySCF 积分后端。
MOLGW 目录没有 .git，程序 banner 的 git SHA 也不可用，因此源码 commit 与二进制
来源未完全独立核实；`stock` 是继承的标签，不是此次完成的源代码洁净证明。
参考构建的版本声明为 3.4，Intel Fortran 2021.10、libxc 7.0.0。

full TDDFT very-high grid 五根有限；insane 相比 very-high 前几根移动约 3.3e−4 eV，
说明 very-high 未通过本轮 1e−4 eV 的网格门槛。少量根 TDA 在两种网格均 NaN，
但同一小体系 TDA 的 nexcitation=0 dense 全谱给出有限根。
这确认对求解分支敏感，未定位具体 Fortran 根因；全谱任务不是五根任务。
既有 raw 行曾只看进程 exit=0 标记 ok，汇总新增 scientific_status 标记数值失败。
未对这些 MOLGW 数据给出速度比。全部输入、基组数据、输出 YAML 与 stdout 均保留。

## 9. 下一轮按证据排序的三项工作（本轮不实施）

| 优先项 | 证据 / 占比 | 最小实施与验收 | 成本、精度 / AD 风险与收益界 |
|---|---|---|---|
| 1. 稳定 TD 数值函数和编译边界 | water profile warm 4 compiles / {compile_med:.2f} s，约 {fraction:.0%} 诊断总时间 | 构建稳定 action/solver callable，轨道、积分、grid、XC/NN 参数作为 dynamic PyTree；同 shape 新对象编译计数，几何/参数改变后能量与 JVP/VJP 不变 | 中；不能静态化可变 self 或复用旧物理数组。无新近似；仅该诊断的理想上界约 {1/(1-fraction):.2f}×，实际收益未知 |
| 2. 公共 packed J/K 与设备/预算布局策略 | CPU90 clean RSS 约 {rss(n90):.0f}→{rss(n90b):.0f} MB，temp 525→6.16 MB；GPU60 分块反而慢约 {med(gpub,'packed_jk_warm')/med(gpu,'packed_jk_warm'):.2f}× | 复用公共非对称/batched 语义；CPU 与 GPU 分开选工作区策略，增加多个尺寸/预算点再定切换边界；保留 packed 几何/数组 AD 验收 | 低至中；pair 重数、任意密度、complex-density、导数权重有风险。无新近似；小水端到端目前无可靠改善，不能承诺 3× overall |
| 3. 明确公共 CPU 积分 / GPU 数组边界 | 原 RI/GPU 入口 FFI 报错；显式 staged 路径已成功，CPU8 与 staged GPU 小体系总时间接近 | 把 CPU-only 2c/3c 与一次性设备搬运显式组织，记录传输、失败与参数反传；复用现有单一 libcint 接口，不造 GPU 积分器 | 中；涉及 placement、tracer/参数依赖与 RI 几何尚未支持的边界。无新近似；目前是可用性缺口，不能把失败基线算成加速 |

DF preparation sharing 可作为局部简化，但现有数据仅毫秒级 eager 节省，未成为
优先性能工程。ISDF/THC、全新求解器、多 GPU、SCF/CC/GW 大改不由本轮证据支持。
后续还应把公开的 forward/response workspace 契约接入完整内存预算；当前 H7 只有
源码证据，不能宣称已测出大体系 backward 的峰值来源。

## 10. 交付、复跑与未覆盖项

- `manifest.json` / `environment.json` / `source-verification.json`：基线、实际路径、构建、库哈希。
- `cases/` 与 `runs/*/input.json`：有效配置和原始注册输入；metadata 修正单列。
- `measurements.jsonl` / `summary.csv` / `comparisons.json` / `compile_counts.json`：逐次数据与汇总。
- `runs/`：原始 stdout、RSS 时间线、NPZ、优化 HLO、profile、MOLGW 输入/输出；保留于 worktree。
- `experiments/` 与 `harness-*.patch`：候选和计时辅助代码的独立修改。
- `commands.txt`：smoke 与主要结果的复跑命令。

尚未验证：RSH、MGGA、大规模分子完整求解、GPU AD、NNAO 激发能训练/力损失、
全 U-TDDFT geometry、性质反传峰值、真实训练 checkpoint，以及 ISDF/THC 的精度/成本。
没有把 timeout、NaN、未收敛或缺少 lowering 的案例加入可信速度比。

原工作区 tracked HEAD/diff 与 index **逐字节一致**。3772 个既有 untracked 文件中，
以下 {len(workspace['untracked_changed'])} 项在最终指纹核对时与起始状态不同：

{untracked_changes}

本任务没有直接写入这些文件，也未将它们迁入基线；不能据此宣称所有 untracked
字节保持不变，亦未推断变化来源。现场保留，见 `original_workspace_verification.json`。
未合并、推送或修改生产数值核。
'''
(root/'report.md').write_text(text)
print(root/'report.md')
