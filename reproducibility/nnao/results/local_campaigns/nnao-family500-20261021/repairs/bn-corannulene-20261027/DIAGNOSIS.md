# L-TZP BN-Corannulene OOM: initial evidence

Observed 2026-10-27; investigation only, no numerical source patch or training
restart performed for this failure yet.

- L-TZP job 108612 passed the A100 exact C6Cl6 forward/implicit contraction/
  MACE-backward gate (source `c28a091033269e2a50f935050ab481e2480fa5566ef2a45099a6e5a47703179c`).
- It accepted three minibatch updates, then failed in epoch 1 / batch 4 on
  `GMTKN55-v2/INV24/BN_Corannulene`.
- Error: `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 29.95GiB`
  on A100 80G. This is device memory exhaustion, not another diagnosed SCF
  convergence failure or a host-memory-limit error.
- `last.npz` remains present (397904029 bytes); checkpoint metadata is readable:
  update_step=3, epoch=0, batch_cursor=3. No optimizer reset or checkpoint
  migration has been performed.
- Failed system: 420 contracted AOs, mixed primitive-pool AO count 920,
  packed DF shape `(1700, 423660)`, fixed-cache arrays 5775318400 bytes.
  Storage remains mmap; the full training dataset is not resident on the GPU.

The exception is surfaced when `_checked_info` converts an asynchronous device
scalar to a host value. This stack location does not identify the allocating
operation, nor does it show that Python float conversion needs 29.95 GiB.

Frozen `density_fitting.project_factors` already unpacks at most 64 auxiliary
rows and uses checkpointed blocks in reverse AD. Do not claim that simply
adding auxiliary blocking/rematerialization is a new fix without tracing the
compiled workspace and live buffers. The NNAO objective additionally transforms
contracted factors into the orthonormal SCF/implicit coordinate system.

Next diagnostic: use the exact accepted update-3 parameters, geometry, exponent
pool and cache, compare a clean-process complete single-system evaluation with
the accumulated training process, and inspect compiled argument/output/temp
memory and live device buffers. Preserve float64, all AOs, implicit tolerance,
basis definition and optimizer state. No finite-difference validation is needed.
