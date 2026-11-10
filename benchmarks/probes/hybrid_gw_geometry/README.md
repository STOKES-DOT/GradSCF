# Hybrid KS references and molecular GW response probes

These scripts call GradSCF production kernels. Run them inside an allocated
compute job with JAX float64; set OpenMP/OpenBLAS/MKL to one thread and bind
the requested logical CPUs. Native Gaussian/RI integrals run on CPU. The GPU
group explicitly transfers the prepared numerical inputs into diagonal GW.

The frozen phenanthrene inputs are in
`reports/hybrid_gw_geometry/20261031-f0c9b2b/cases/`. Cartesian def2-SVP has
260 AOs; Cartesian def2-TZVP has 564 AOs, 47 occupied and 517 virtual orbitals.
The six reported QP targets leave the complete G/W spaces intact. The original
omegaB97X-D recipe includes its CHG dispersion. The grid level is explicit.

Small CPU smoke:

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python benchmarks/probes/hybrid_gw_geometry/run.py geometry /tmp/h2-pbe0.jsonl \
  --xc pbe0 --backend direct --grid 0 --threads 1 --repeats 0 --finite-difference
```

Large fixed-geometry workflow (one GPU must be allocated and visible):

```sh
PYTHONPATH=src JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=32 \
python benchmarks/probes/hybrid_gw_geometry/run.py forward /tmp/phi-pbe0.jsonl \
  --case-input reports/hybrid_gw_geometry/20261031-f0c9b2b/cases/phenanthrene-def2-tzvp.json \
  --xc pbe0 --method g0w0 --backend direct --grid 1 --threads 32 \
  --screening 1e-13 --device gpu --repeats 0 --checkpoint /tmp/phi-pbe0.npz
```

Repeat with `--xc wb97xd` for the RSH reference. Add `--ri-seed` to test
RI-PBE0 initialization followed by the **same final direct** calculation.
Its seed time is included in reference/total time; `ri_seed` is a nested stage,
so adding it again would double-count. `--backend df` instead changes the final
SCF integral approximation and must be labeled separately. RSH RI-JK and RI
geometry are currently rejected explicitly. No alternate integral engine is used.

For derivative tests, `--restart` accepts a density written by this probe;
the reference and perturbed states still reconverge. Geometry AD uses zero
screening regardless of the static starting-density preparation threshold.
`--finite-difference` evaluates two steps, 3e-4 and 1e-4 bohr, along a fixed
normalized direction. A completed compile is not a completed derivative run.

`jk.py` measures a fixed synthetic density on the real AO basis. It reports
one first and three warm calls, native library hashes and whole-process
ru_maxrss. Its timings exclude SCF and cannot be presented as end-to-end speed.

`scgw_response.py` is an independent small dense-Jacobian diagnostic of the
production implicit response. It uses the same converged state and physical
parameters, not a second GW implementation. Array geometry tests use a
traceable exact AO-pair reference, which is separate from native auxiliary RI.

Raw samples, compiler estimates, actual RSS, GPU allocator counters and failed
runs have distinct fields in the report. The default per-calculation timeout
for the remote controllers is 6000 seconds. Preserve each output variant and
its manifest; do not overwrite a failed baseline with a successful rerun.

For a large geometry calculation, use `--forward scf --grid-block-size 4096`
to select the existing DIIS numerical solve and the same implicit response.
`--screening 0` retains the exact direct geometry product. This block size
does not change the level-1 quadrature. The default `forward=minimize` is
unchanged.

An explicitly separate local-response stage can use `--forward reference`
with `--reference-state` from this campaign's completed strict zero-screening
SCF. Its JSON metadata verifies the case, basis, grid, native library and
convergence. It is shared-input response, not an independent end-to-end run.
Use `--gradient-record` and `--fd-step` with `--forward scf` for FD-only
verification: every displaced geometry relaxes a new SCF, rather than
evaluating the old orbitals there. Both modes call production response code.

`finish_geometry.sbatch` replays the frozen remote acceptance sequence, with
one process per stage and no overwrite of an existing result. Submit it only
after its reference/gradient prerequisites exist, from the campaign root.
`validate_final.sbatch` runs the last bounded regression separately, using one
CPU and no GPU, after all timing runs. These scripts record process exit and
scientific gates separately; a returned process does not alone pass a force
or FD test. The remote campaign paths and environment are intentionally
explicit, rather than silently selecting another checkout or binary.
