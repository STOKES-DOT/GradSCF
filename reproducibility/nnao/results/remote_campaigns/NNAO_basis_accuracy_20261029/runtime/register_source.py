"""Register pinned model-local sources in this independent environment only."""
from pathlib import Path
import site
import sys
import json
import shlex
import importlib.metadata as metadata

prefix=Path('/home/user_yuanjiao/private/homefile/conda-envs/gpu4pyscf_bench')
assert Path(sys.prefix).resolve()==prefix.resolve()
source=Path('/home/user_yuanjiao/private/homefile/GradSCF_family500_20261021')
paths=[source/'src',source/'src/gradscf/model/nnao/vendor/mace-jax']
for path in paths:assert path.is_dir()
(Path(site.getsitepackages()[0])/'gradscf_nnao_pinned.pth').write_text(''.join(str(p)+'\n' for p in paths))
link=Path('/home/user_yuanjiao/.conda_from_module/envs/gpu4pyscf_bench')
link.parent.mkdir(parents=True,exist_ok=True)
if not link.exists():link.symlink_to(prefix,target_is_directory=True)
else:assert link.resolve()==prefix.resolve()
site_packages=Path(site.getsitepackages()[0])
sdk=Path('/opt/data/hpc/common/softwares/nvidia/hpc_sdk/Linux_x86_64/25.3')
toolkit=sdk/'cuda/12.8'
assert (toolkit/'bin/nvcc').is_file() and (toolkit/'include/cuda_runtime.h').is_file()
cuda_libraries=[site_packages/'cutensor/lib',toolkit/'lib64',sdk/'math_libs/12.8/lib64']
cuda_libraries += [site_packages/'nvidia'/name/'lib'
                  for name in ['cudnn','nccl','nvshmem','cuda_cupti','nvtx']]
assert all(p.is_dir() for p in cuda_libraries)
assert any((p/'libcusolver.so.11').is_file() for p in cuda_libraries)
activation='''# Complete cluster CUDA Toolkit; pinned pip cuTENSOR takes priority.
if [[ ${_GPU4PYSCF_BENCH_LD_ACTIVE:-0} != 1 ]]; then
    export _GPU4PYSCF_BENCH_LD_OLD_SET="${LD_LIBRARY_PATH+x}"
    export _GPU4PYSCF_BENCH_LD_OLD="${LD_LIBRARY_PATH-}"
    export LD_LIBRARY_PATH=__CUDA_LIBRARIES__"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export _GPU4PYSCF_BENCH_PATH_OLD_SET="${CUDA_PATH+x}"
    export _GPU4PYSCF_BENCH_PATH_OLD="${CUDA_PATH-}"
    export _GPU4PYSCF_BENCH_HOME_OLD_SET="${CUDA_HOME+x}"
    export _GPU4PYSCF_BENCH_HOME_OLD="${CUDA_HOME-}"
    export CUDA_PATH=__CUDA_TOOLKIT__ CUDA_HOME=__CUDA_TOOLKIT__
    export _GPU4PYSCF_BENCH_LD_ACTIVE=1
fi
'''.replace('__CUDA_LIBRARIES__',shlex.quote(':'.join(map(str,cuda_libraries)))).replace(
    '__CUDA_TOOLKIT__',shlex.quote(str(toolkit)))
deactivation='''if [[ ${_GPU4PYSCF_BENCH_LD_ACTIVE:-0} == 1 ]]; then
    if [[ ${_GPU4PYSCF_BENCH_LD_OLD_SET:-} == x ]]; then
        export LD_LIBRARY_PATH="$_GPU4PYSCF_BENCH_LD_OLD"
    else
        unset LD_LIBRARY_PATH
    fi
    if [[ ${_GPU4PYSCF_BENCH_PATH_OLD_SET:-} == x ]]; then
        export CUDA_PATH="$_GPU4PYSCF_BENCH_PATH_OLD"
    else
        unset CUDA_PATH
    fi
    if [[ ${_GPU4PYSCF_BENCH_HOME_OLD_SET:-} == x ]]; then
        export CUDA_HOME="$_GPU4PYSCF_BENCH_HOME_OLD"
    else
        unset CUDA_HOME
    fi
    unset _GPU4PYSCF_BENCH_LD_ACTIVE _GPU4PYSCF_BENCH_LD_OLD_SET _GPU4PYSCF_BENCH_LD_OLD
    unset _GPU4PYSCF_BENCH_PATH_OLD_SET _GPU4PYSCF_BENCH_PATH_OLD
    unset _GPU4PYSCF_BENCH_HOME_OLD_SET _GPU4PYSCF_BENCH_HOME_OLD
fi
'''
for directory,content in [('activate.d',activation),('deactivate.d',deactivation)]:
    hook=prefix/'etc/conda'/directory/'gpu4pyscf_bench_cuda.sh'
    hook.parent.mkdir(parents=True,exist_ok=True);hook.write_text(content)
out=Path('/home/user_yuanjiao/private/homefile/NNAO_basis_accuracy_20261029/runtime')
report=dict(prefix=str(prefix),python=sys.version,source_paths=list(map(str,paths)),
    cuda_toolkit=str(toolkit),cuda_toolkit_provider='cluster nvhpc/25.3',
    cuda_library_paths=list(map(str,cuda_libraries)),
    packages={name:metadata.version(name) for name in ['pyscf','gpu4pyscf-cuda12x','cupy-cuda12x',
      'cutensor-cu12','jax','jaxlib','flax','e3nn-jax','e3nn','torch','numpy','scipy']})
(out/'installed-environment.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
