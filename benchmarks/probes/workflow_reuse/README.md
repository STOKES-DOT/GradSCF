# Bounded workflow reuse validation

`run.py` is a thin caller of `src/gradscf` APIs. Run the same probe with the
parent and candidate source trees in **separate Python processes**, serially in
the same Slurm allocation and with identical CPU affinity/native binary. Copy
this directory into the parent tree when it is absent there. Do not import
candidate numerical modules into the parent process.

Default scientific inputs are H2, 0.74 Angstrom, Cartesian STO-3G, PBE, grid0,
CPU float64, full native packed four-center SCF, 16 GW frequency points, and
one response root. GW uses the SCF exact-pair spectral factors with the public
facade's 1e-12 factorization threshold. `--backend direct` selects native direct
SCF; the GW chain then explicitly uses independent `weigend` auxiliary RI.
Those representations must be compared separately. No QP or screening energies
are supplied or frozen by the runner.

SCF energy/density/orbital-gradient tolerances are 1e-11 Ha/1e-10/1e-9; TD and
BSE residual tolerances are 1e-9; GW outer tolerance is 1e-7 Ha with 80 cycles,
damping .3, and scGW beta=20 Ha^-1/mixing=.5. Defaults are assembly/convergence
gates, not basis, frequency-grid, or physical accuracy convergence studies.

Run inside an existing allocation (the parent orchestrator owns submission):

```sh
srun --exclusive -n1 -c1 env PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=1 python benchmarks/probes/workflow_reuse/run.py /absolute/new/scf_td.jsonl --suite scf-td --repeats 1
srun --exclusive -n1 -c1 env PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GRADSCF_NATIVE_THREADS=1 python benchmarks/probes/workflow_reuse/run.py /absolute/new/gw_bse.jsonl --suite gw-bse --repeats 1
python benchmarks/probes/workflow_reuse/compare.py /absolute/parent/scf_td.jsonl /absolute/candidate/scf_td.jsonl /absolute/new/scf_td_comparison.json
python benchmarks/probes/workflow_reuse/compare.py /absolute/parent/gw_bse.jsonl /absolute/candidate/gw_bse.jsonl /absolute/new/gw_bse_comparison.json
```

Run small molecule coverage first. Optional water or benzene G0W0/BSE coverage
uses the unchanged public path with `--molecule water` or `--molecule benzene
--methods g0w0`; increase `--nw` identically in both trees if needed. Use the
allocation's existing time limit; do not interpret a timeout as convergence or
as a memory failure. The runner does not submit jobs.

The first repeat of the first case includes process-cold JAX work; later first
calls can share signatures compiled by preceding cases. Warm repeats construct
fresh facades and recompute the entire SCF/response chain. Stage timings include
their public preparation, synchronizing only at stage boundaries. The workflow
timing includes all phases and record overhead. RSS is the cumulative process
peak, not the incremental memory of that phase. The adjacent manifest records
physical inputs, all source Python hashes, probe/native hashes, versions,
Slurm identity, affinity, and thread variables. Use the same case order and
probe hash for a before/after comparison.

Capabilities are explicit:

- RHF/UHF/RKS/UKS continue to both TDA and full TDDFT/TDHF.
- GHF/GKS ground states are calculated; generalized spinor TD response is
  currently unimplemented, and both public response calls must reject it.
- Real RKS sources feed G0W0, evGW, qsGW, and Matsubara scGW. G0W0/evGW/qsGW
  continue to TDA and full BSE with the actual recorded screening spectrum.
- scGW provides imaginary-axis G/Sigma and density/energy, without validated
  QP continuation. Both BSE calls must raise `NotImplementedError`; this
  guard is recorded separately and does not count as a successful BSE result.

Focused numerical gates, serially in the same allocation:

```sh
PYTHONPATH=src python -m pytest -q tests/test_workflow_reuse_integration.py
PYTHONPATH=src python -m pytest -q tests/test_scf_reference_reuse.py tests/test_gw_bse_compile_reuse.py tests/test_td_execution_reuse.py
PYTHONPATH=src python -m pytest -q tests/test_response_external_memory.py -k 'three_training_modes or preserves_three_modes'
PYTHONPATH=src python -m pytest -q tests/test_neural_scf_force_training.py
PYTHONPATH=src python -m pytest -q tests/test_scf_orbital_native_geometry.py
PYTHONPATH=src python -m pytest -q tests/test_molecular_gw_hybrid.py
```

The new integration module shares preparation with the runner. Existing tests
retain neural fixed-density/explicit/implicit gradients, native geometry
response, all four GW geometry JVP/VJP and finite differences at 3e-4 and
1e-4 Bohr. They remain necessary scientific gates; successful forward timing
alone does not establish these derivative properties. The comparison helper
checks matching stage keys, fields, masks, and numerical values; inspect both
manifests for matched hardware, parameters, and native binary before using its
descriptive time ratios as performance evidence.
