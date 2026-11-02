# 94688 续训启动审计

诊断日期：2026-10-09；SSH 登录节点 `slurm-client`，仅执行轻量 shell/环境检查，没有启动科学计算、安装软件、修改环境或重新提交训练。

## 已确认的事实

- `resume-94688.out` 只记录 Anaconda3 模块重载，没有 guard、allocator 或训练输出。
- `run_resume.sbatch` 在 `module load anaconda3`、`conda activate jax_scf` 之后才调用 guard；EXIT trap 更晚才安装，故这次启动失败未留下脚本级退出码。
- 新鲜 bash 中 `module load anaconda3` 返回 0，随后激活 `jax_scf` 返回 0。
- 继承已加载模块但尚未激活 Conda 的 bash 子进程，也可正常激活。
- 继承已激活 `jax_scf` 的 bash 子进程在模块重载后执行 `conda activate jax_scf`，可复现 bash segmentation fault，退出码 **139**；严格模式与非严格模式都复现。因此不是 `set -e` 或模块返回非零导致。

## 复现的根因

集群模块定义的 `conda` wrapper 如下：

```bash
conda () {
    if [ -z "${_CONDA_LOADED_FROM_MODULE:-}" ]; then
        eval "$(/opt/data/hpc/common/softwares/anaconda3/bin/conda shell.bash hook)" > /dev/null 2>&1 || true
        export _CONDA_LOADED_FROM_MODULE=1
    fi
    conda "$@"
}
```

父进程激活后导出 `_CONDA_LOADED_FROM_MODULE=1`，新 bash 继承这个标记但没有继承真实 Conda 函数。模块重载重新定义上述 wrapper。wrapper 因标记仍为 1 跳过 hook，随后递归调用自身，直到 bash 栈溢出。

清除该标记后不再发生 139，但尚不足以可靠激活环境：新 login shell 的 PATH 与继承的 CONDA_PREFIX 不一致时，本次测试 `conda activate jax_scf` 返回 0，而 `command -v python` 仍指向 Anaconda base。环境身份必须由 `sys.executable` 核对。

## 对 94688 的证据边界

上述启动缺陷已经隔离复现，日志形状与 94688 一致。但 94688 已被 Slurm 控制器清除，`sacct` 被禁用，而且旧脚本没有早期 EXIT trap，因此不能宣称拿到了原作业的退出码 139。若提交进程当时已激活 Conda，复现能够完整解释其失败；该提交状态需结合原提交记录确认。

## 最小稳妥启动方式

无需重新激活 Conda，使用已验证环境 Python 的绝对路径：

```text
/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin/python
```

- 该解释器在无 `LD_LIBRARY_PATH` 的新鲜登录 shell 中可正常启动；在加载 anaconda3 模块后也可正常启动，均返回 0。
- 模块设置 `LD_LIBRARY_PATH=/opt/data/hpc/common/softwares/anaconda3/lib`；激活 `jax_scf` 不再增加库路径。这里只验证了 Python 启动和安装元数据，没有在登录节点初始化 GPU。
- 已核对 Python 3.11.15，JAX/JAXlib 0.10.0，Flax 0.12.7，Optax 0.2.8，NumPy 2.4.4。
- 唯一 activate.d 脚本 `gradscf-cuda-cache.sh` 为 Slurm 任务设置节点本地 CUDA_CACHE_PATH；若绕过激活，计算脚本可显式保留相同缓存设置。
- 启动时应先安装 EXIT trap，并记录模块返回码和 `sys.executable`，以免再次丢失启动诊断。

本次没有修改共享 Anaconda 模块、Conda 环境或任何生产源码。
