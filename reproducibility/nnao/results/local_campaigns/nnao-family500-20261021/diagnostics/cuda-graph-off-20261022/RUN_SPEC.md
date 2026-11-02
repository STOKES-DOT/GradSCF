# CUDA Graph disabled: bounded S-TZP recovery comparison

- Source: frozen `GradSCF_family500_20261021` campaign; no scientific-source changes.
- Seed: immutable copy of `runs/nnao_tzp_s/train/last.npz`, accepted update 16,
  completed epoch 1, next batch cursor 6, with Adam moments and online accumulators.
- Original failure: CUDA graph instantiation OOM during update 17 at NBPRC/bz;
  the process reported 75,476 alive graphs.
- Runtime change: `XLA_FLAGS=--xla_gpu_enable_command_buffer=` before JAX startup.
- Hardware: n005, one A800 80G, 32 CPUs, 512 GiB host RAM, at most 8 hours.
  Original failure used A100 80G on n001; report this hardware difference when
  interpreting timings. A same-device A/B comparison is not claimed.
- Preserve all 500 structures, fixed S primitive pool, full MACE, seed 0,
  batch size 50, float64, DF-RHF, implicit tolerance 1e-8, Adam state and schedule.
- Stop at accepted update 40: replay the uncommitted failed batch once, finish
  epoch 2, then evaluate complete epochs 3 and 4. Epoch 4 can expose warmed-up
  behavior after all 500 molecule programs have been compiled.
- No finite differences, gradient checks, proposal scoring or extra SCF passes.
- Output stays under this diagnostic directory; original training output and
  checkpoint remain unchanged. Passing update 40 does not prove indefinite memory
  stability or fully identify a memory leak.
- Record checkpoint hashes, runtime configuration, every-10-second GPU memory/
  utilization and process peak RSS samples, epoch/update progress and wall time.
- This test occupies the TZ lane. Existing M-TZP must depend on this test before
  it can start, retaining the at-most-three-GPU campaign limit. M-DZP continues.
