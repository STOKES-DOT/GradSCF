# Independent GPU4PySCF and MACE-JAX runtime

On jy_gpu_2:

```bash
module load anaconda3 nvhpc/25.3
conda activate gpu4pyscf_bench
python -m pip check
```

Environment prefix: `/home/user_yuanjiao/private/homefile/conda-envs/gpu4pyscf_bench`.
The original `jax_scf` environment is unchanged.
Activation selects the complete NVHPC 25.3 CUDA 12.8 Toolkit and math
libraries, with pinned pip cuTENSOR 2.2.0 first in `LD_LIBRARY_PATH`.
It sets CUDA_PATH/CUDA_HOME; deactivation restores previous values. When using
the absolute Python path without Conda activation, source
`$prefix/etc/conda/activate.d/gpu4pyscf_bench_cuda.sh` first, where `$prefix`
is the environment prefix above. `minimal.slurm` already does this.

GPU4PySCF 1.7.0 uses its official CuPy 13.4.1, cuTENSOR 2.2.0 and PySCF
2.8.0 combination. NumPy 2.2.6 and SciPy 1.15.3 support the pinned JAX
0.10.0 / Flax 0.12.7 / e3nn-jax 0.21.0 model stack. Torch 2.7.1+cpu is
for upstream e3nn utilities. `environment-freeze.txt` contains exact versions.

Checkpoint-matched GradSCF and vendored mace-jax are registered by `.pth`
from `GradSCF_family500_20261021/src`. No unrelated mace-jax release is
installed, and the pinned source is not modified by installation.

`minimal.slurm` requests one V100, eight CPUs, 64 GiB and 30 minutes.
It configures node-local CUDA/CuPy/JAX caches, float64, no JAX preallocation,
and the complete cluster CUDA 12.8 compiler. Submit from the logical campaign path:

```bash
cd /home/user_yuanjiao/private/homefile/NNAO_basis_accuracy_20261029
sbatch --chdir="$PWD" runtime/minimal.slurm
```

The test checks bare CuPy, cuTENSOR, GPU DF-MP2 (FP64, frozen O 1s, no T2),
and omegaB97M-V with both quadrature levels set to 3 and NLC enabled.
cc-pVDZ and frozen NNAO-SZP-S water bases are compared to CPU PySCF using
identical explicit orbital/auxiliary shells. Error tolerances are 1e-7 Eh
for MP2 and 1e-6 Eh for DFT; SCF tolerances are 1e-11 Eh / 1e-8 orbital
gradient. CPU and GPU MACE inference must reproduce the frozen basis within
1e-10. JSON reports distinguish installation, CPU inference, and GPU tests.

The user has requested direct formal execution. No additional CPU/GPU tests
are running. Formal jobs are tracked in ../scheduler.json; minimal.slurm is
an archived diagnostic entry point, not a release dependency.

Formal execution is physical-only: 1680 energy cases, no CP/BSSE, ghost
atoms or dimer-conditioned monomers. Completed physical energies are reused
via the unchanged run_gpu_engine.py fingerprint. CP outputs are not required
or read by the final aggregate.py scoring/plots.
