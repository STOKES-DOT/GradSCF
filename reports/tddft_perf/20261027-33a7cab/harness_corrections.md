# Harness-only corrections

No production source was changed. The first AD fixtures inherited the water
case metadata even though the existing regression fixture explicitly computes
H2/STO-3G, level 0, one root, 1e-10 response tolerance. Those raw records remain
under `relaxed-geometry-cpu1` and are superseded by the corrected-metadata runs.
The numerical functions and settings were unchanged. Consolidated summaries
exclude the superseded AD metadata, rather than silently relabeling raw files.

UKS stores its prepared reference directly rather than retaining RKS-style
`_scf_inputs`. The harness now reads grid/factor metadata from that reference
when needed; this changes no numerical call. UKS formal timings use the
`corrected-harness` variant. Initial restricted baselines remain unchanged.

The DF TDA factory intentionally leaves unused B-branch closure cells unbound. The synchronization/size inspector now skips those empty cells. Original failure logs remain; both original and demand-driven kernels are rerun under `-repaired`. No numerical function or dependency is replaced.

The low-level response builders take a Functional object, unlike the public facade that accepts a name. Shared-input/NTO probes now pass SemilocalResponseFunctional; failed raw-string calls are preserved and shared cases rerun with a repaired label. Public TDA/TDDFT baselines were unaffected.

Kernel correctness oracles ran after timed actions in the original microbenchmark process. This polluted whole-process RSS (and GPU allocator lifetime peaks) with the oracle implementation. Four additional `clean-memory` processes omit that already-validated oracle, isolating original/candidate CPU90 and GPU60 memory. Earlier process peaks are retained but not interpreted as pure kernel memory savings.

The initial independent PySCF worker also imported JAX at entry and GradSCF after timing for provenance. Numerical inputs remained independent, but pure-engine RSS was contaminated. The final worker imports neither for PySCF end-to-end mode. Water TDA/full, both engines, are rerun in a small CPU allocation under final-clean labels; no original peak is overwritten.
