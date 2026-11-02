# GPU4PySCF manual and runtime review — 2026-10-29

Primary sources:

- https://github.com/pyscf/gpu4pyscf/tree/v1.7.0
- https://raw.githubusercontent.com/pyscf/gpu4pyscf/v1.7.0/README.md
- https://raw.githubusercontent.com/pyscf/gpu4pyscf/v1.7.0/requirements.txt
- https://pyscf.org/user/gpu.html

## Confirmed

The package's own v1.7 requirements pin CuPy 13.4.1, cuTENSOR 2.2.0 and
PySCF 2.8.0. Existing jax_scf contains CuPy 14.0.1, cuTENSOR 2.6.0 and
PySCF 2.13.0. The log's `using cupy as the tensor contraction engine` warning
matches the official installation warning; the installed code caught an
ImportError/AttributeError while importing CuPy's cuTENSOR binding.

Direct gpu4pyscf classes plus `.density_fit()` are an officially supported
calling pattern, so the constructor style itself is supported. GPU DF-MP2
remains explicitly checked as a gpu4pyscf class, FP64, no saved T2.

The A800 node reports driver API 12.4 while the existing runtime/NVRTC are
12.9. Independent bare CuPy also spends time in NVRTC compilation; it eventually
evaluated `arange(2)`. No electronic energy case has passed yet. This does not
prove that every observed delay is caused solely by the version pairing.
Removing the CUDA memory-sampling thread did not immediately resolve compilation.

## Authorized independent environment setup

Independent environment: gpu4pyscf_bench, under
`~/private/homefile/conda-envs/gpu4pyscf_bench`. Follow this server's environment
management rules; do not replace packages in jax_scf, which also supports JAX
and separate workflows.

The user explicitly approved installation and minimal testing on 2026-10-29.
Install the official v1.7 dependency set plus a matched CUDA 12.4 runtime;
requirement files are beside this note. NumPy 2.2.6 and SciPy 1.15.3 allow
the pinned JAX 0.10 / Flax / e3nn-jax model stack in the same environment.
CPU Torch 2.7.1 supports the upstream e3nn utilities; MACE inference uses JAX.
The vendored, checkpoint-matched mace-jax source is registered through a
`.pth` file, without editing the source or installing an unrelated release.

Conda index locking first failed with ENOLCK on this filesystem. The retry
uses an isolated package cache and process-local `CONDA_NO_LOCK=true`;
global Conda configuration and the old environment are unchanged.
Installation completed at 2026-10-29 13:43:05 CST, and `pip check` passed.
The classic `module load anaconda3; conda activate gpu4pyscf_bench` sequence
was also verified. `installed-environment.json` and `environment-freeze.txt`
record the actual packages. The reused JAX PJRT wheel matches the official
PyPI SHA256; `reused-wheel.json` records this check.

CPU MACE job 111743 completed with ExitCode 0:0. The SZP-S update-1150
checkpoint reproduces the frozen water basis within 1.11e-16, below the
1e-10 tolerance, in 18.17 s. See `mace-validation-cpu.json`.
V100 minimal GPU job 111740 ran but failed on GPU4PySCF import because
`libcusolver.so.11` was not on the loader search path. The wheel and all
its dependencies are installed; `LD_LIBRARY_PATH` was unset. A separate
process with only this path changed loads the same installed library.

The independent environment now has Conda activation/deactivation hooks
for the NVIDIA and cuTENSOR directories. Actual loads of cuSOLVER, cuBLAS,
cuSPARSE, NVRTC and cuTENSOR resolve to this environment, and deactivation
restores the original path, including its unset state. Package versions did
not change. The batch script explicitly sources the same hook.

Corrected V100 job 111786 is pending. At 13:58 CST Slurm estimates
2026-10-29 14:20:23 on n002; this can change. Water energies and GPU MACE
remain unvalidated. Installation and numerical validation are separate gates.

For the minimal test: validate a bare CuPy kernel, confirm
cuTENSOR loads, and perform the original water HF/MP2/omegaB97M-V checks with
CPU comparisons. Also verify MACE CPU/GPU outputs against the frozen basis.
Freeze the resulting exact environment and loaded library
paths. All 30 geometries and 28 frozen orbital-basis files remain unchanged.
Auxiliary basis definitions are also frozen from the original PySCF 2.13
export so the environment change cannot quietly change the benchmark basis.

## Current execution state

Owned GPU preflight 111631 is held/requeued; its queue controller was stopped.
Other user tasks are unchanged. The three photophysics jobs previously called
GPU jobs are verified CPU-only, with `JAX_PLATFORMS=cpu` and no GPU TRES.
The controller now uses authoritative Slurm TRES rather than node labels.

Nine-model export passed using checkpoint-matched sources. Old L-SZP source
was restored from its exact d86bd78... archive. 28 basis files cover 840
basis/physical-geometry records. No benchmark accuracy plot has been produced.
