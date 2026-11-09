# Reproduce the accepted direct-J/K checks

Run numerical commands only inside an existing approved Slurm allocation on
jy_gpu_2. No new job is submitted by this document. The final paired benchmark
used CPU8, one NV5090 and 256 GiB; integrals execute on CPU. Use an isolated
checkout/build output, preserving existing environments and baseline files.

```sh
PY=/home/user_yuanjiao/private/homefile/gradscf-libcint-nnao-20261026/env-compare/bin/python
RUN=/home/user_yuanjiao/private/homefile/gradscf-direct-jk-ad-20261109
cd "$RUN"
export PYTHONPATH="$RUN/src:$RUN:$RUN/tests"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 JAX_ENABLE_COMPILATION_CACHE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GRADSCF_NATIVE_THREADS=8 XLA_PYTHON_CLIENT_PREALLOCATE=false
```

Smoke, using the production native library:

```sh
srun -n1 -c8 --cpu-bind=cores "$PY" -m pytest -q tests/integrals/test_direct_parameter_ad.py
```

Consumer and training regression after the UHF interface correction:

```sh
srun -n1 -c8 --cpu-bind=cores "$PY" -m pytest -q \
  tests/test_direct_molecular_policy.py tests/test_hfks_direct_dense.py \
  tests/test_external_neural_xc.py::test_three_training_modes_have_fd_gradients_and_accept_external_params \
  tests/test_neural_scf_force_training.py \
  tests/nnao/test_direct_basis_energy.py \
  tests/nnao/test_szp3_direct.py::test_methane_all_electron_energy_and_gradient_without_ecp
```

CUDA-stage checks, within the allocated GPU visibility:

```sh
JAX_PLATFORMS=cuda,cpu srun -n1 -c8 --cpu-bind=cores "$PY" -m pytest -q tests/integrals/test_direct_parameter_gpu.py
```

Exact executed commands, process guards and peak-RSS sampling are embedded in
`acceptance.json` and the raw queue scripts. Final native build command and
hashes are in `manifest.json`; use diagnostics only in a separate source/build
tree and set `GRADSCF_EXPECT_DIAGNOSTICS=1` for the counter gate.

Fixed-density AD probe (no SCF):

```sh
srun -n1 -c8 --cpu-bind=cores "$PY" "$RUN/products.py" --workers 8
```

The large forward benchmark requires the archived input and converged density:

```sh
STATE=/home/user_yuanjiao/private/homefile/gradscf-hfks-finish-20261106/representatives-v2/beta-pbe0-direct/gradscf
srun -n1 -c8 --cpu-bind=cores "$PY" "$RUN/forward.py" \
  "$STATE/input.json" "$STATE/density.npz" forward-replay --workers 8 --repeats 3 \
  --reference "$RUN/forward-final-before"
```

`forward-replay` must be a new directory. Compare complete J/K arrays, not just
energies. The immutable baseline source and native library are retained under
`$RUN/baseline-src`; the baseline commit alone does not provide a matching built
extension. Raw input/library/harness hashes are in the measurements manifest.
This benchmark has no SCF and must not be labelled end-to-end HF/DFT performance.

Local, standard-library-only evidence regeneration after syncing small logs:

```sh
python3 benchmarks/probes/direct_jk_ad/summarize.py \
  reports/direct_jk_ad/20261109/raw reports/direct_jk_ad/20261109/results
```

The remote archive keeps the copied runners as `$RUN/products.py` and
`$RUN/forward.py`; their repository sources are under
`benchmarks/probes/direct_jk_ad/`. The runner imports only production `src` for physics. Raw logs/NPZ data are not
Git package assets. Recorded RED/failing intermediate runs are intentional
evidence; the acceptance ledger identifies their GREEN replacements.
