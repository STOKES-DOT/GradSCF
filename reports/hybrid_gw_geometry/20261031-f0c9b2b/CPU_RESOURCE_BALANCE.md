# libcint CPU resources and derivative scheduling

This is a measured extension of the hybrid/GW geometry plan, not a new
integral backend or a change to exchange screening. All numerical work below
ran in the existing Slurm allocation 114012 on n006. The allocation owns one
NV5090, 64 logical CPUs and 256 GiB; the formal molecular Fock probe pins 32
physical cores. OpenMP and BLAS thread settings are one, and native work uses
the supplied JAX FFI pool. No nested OpenMP team was added.

## Reference and observed problem

The eight selected CP2K files are pinned at
`d077e66fe06a862d49bdcc2b91c179463e9487eb`; their hashes and Git blob IDs are
in `cp2k_source_manifest.json`. Its HFX code accounts for shared Fock/density
and private work buffers, and supports explicit task-bin load balancing. The
documented allowance is per MPI process and shared by its threads:
[CP2K HFX memory](https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/DFT/XC/HF/MEMORY.html).
This is architectural guidance, not a CP2K/GradSCF timing comparison.

A disposable native diagnostic recorded a requested pool of 32, an available
pool of 32, but only 13 selected workers for one-density J/K at 564 AOs.
The worker-private matrices consumed 66,163,968 bytes, just below the old
64 MiB cap. Thus an artificial workspace cap restricted this call despite
the much larger declared method allowance. These are selected worker counts,
not a claim that each worker was busy throughout the calculation.

An independent uniform eight-shell geometry fixture found highly uneven
canonical quartet counts under contiguous first-shell partitions:
6, 49, 176 and 435. A four-worker cyclic outer-pair partition instead measured
the multiset 153, 162, 171 and 180 (666 total). The serial and parallel
coordinate pullbacks agree. Raw trace: `raw/native-cyclic-worker-loads.xml`.
Counts on this uniform fixture do not prove a molecular throughput gain.

The imbalance follows from the triangular orbit enumeration. With zero-based
canonical pair index `p`, that outer pair owns `p+1` inner pairs. Equal numbers
of first shells therefore do not imply equal work: summing these pair lengths
within a first shell grows approximately cubically with its index. Cyclic
outer-pair assignment interleaves short and long lists without retaining an
O(nshell^4) task table. Angular momentum, primitive counts and screening can
still make equal quartet counts unequal in time; the molecular timing below
is required in addition to this count check. No geometry-dependent task-time
cache or nondeterministic work-stealing queue was introduced.

## Production changes

* `4943d0b`: distribute coordinate VJPs across canonical shell-pair tasks.
  Both first and directional second coordinate products retain the existing
  shell-derivative permutations, contraction and fixed worker reduction.
* `22387db`: bind private accumulation to the existing `max_memory` budget.
  Resident arrays, AO work and fixed native work are charged first; private
  accumulation uses at most a quarter of remaining workspace across the
  existing three-call derivative-overlap allowance. An explicit smaller
  provider quota is preserved. Standalone providers retain 64 MiB.
* Physical basis parameters remain dynamic PyTree leaves. The quota is
  static execution metadata. Projected NNAO providers propagate it through
  their primitive action without freezing the contraction transform.
* The changed native ABI has an explicit new symbol and requires rebuilding.
  Missing symbols produce an actionable error; there is no old-integrator
  fallback. Linux and macOS linker allowlists and their build dependencies
  are updated. Only the Linux native build was numerically validated here.

## Matched Fock measurements

Same phenanthrene def2-TZVP Cartesian input, 564 AOs, PBE0, grid level 1,
zero integral screening, float64, CPU32, full-array numerical gates. The
runner is the frozen `stationarity-v9.py` from commit `2e1e330`, SHA256
`5b5abca5a79d9362515820b9530d64b5790504cb9cbf3c98ae520bee47534853`.
Each process has one first execution and three warm executions.

| Prepared-input Fock | Previous cap | Adaptive quota |
|---|---:|---:|
| Warm samples, seconds | 40.162343, 39.965757, 39.950043 | 26.006910, 26.058645, 25.903943 |
| Median, seconds | 39.965757 | 26.006910 |
| Process peak RSS, bytes | 9,566,560,256 | 9,571,676,160 |
| XLA temporary estimate, bytes | 2,196,302,024 | 2,196,302,024 |

The time ratio is 0.650730, a 34.9% reduction within this timing boundary.
Maximum Fock difference is 3.1264e-13 Ha, energy difference 9.0949e-13 Ha,
and density difference is zero. The actual native libraries are respectively
`eaad347f97164de3ad719790f303053ff128a9c7cb2a5b9a814ed2173b357cd6`
and `ce51d885d4b65e753b5f56863470f1f1bb9c41d4b0762e47df5128e0d0b8feb8`.
See `quota-fock-comparison.json` for gates, raw samples and source hashes.

This excludes SCF iterations, integral/grid preparation and all derivatives.
RSS includes preparation/compilation and is not the incremental private-buffer
peak. The XLA estimate excludes native heap allocations. Neither metric may
be substituted for the other. No end-to-end or force speedup follows from
this Fock measurement.

## Validation and remaining gates

The integrated native group passed 88 tests; six diagnostic-only tests were
skipped in its production build. A separate diagnostic load test passed.
The integrated spin-response group passed 53 tests, including native quota
binding, complex spin mixing and UKS/GKS geometrical gradients and directional
second derivatives. The native GKS geometry fixture is HF with an explicitly
moving complex spin field, not a full semilocal molecular-grid campaign.
Separate array tests cover collinear PBE0 and noncollinear LDA, including
clipping policies. Existing GKS support is dense/packed ERIs and collinear
LDA/GGA/global hybrids or noncollinear LDA/HF; this patch does not add GKS
direct J/K, RSH, MGGA or noncollinear GGA.

The six-test UKS group also passed (241.34 s), including OH LDA/GGA and H2+
PBE0/wB97X-D relaxed geometry, field response, HVP and two FD steps. These
groups overlap; their counts are not a unique-test sum. CPU/GPU matched
array-kernel measurements passed full-output, dynamic-input and array-AD
checks; the measurements below retain their distinct timing boundaries.

The final parameter-protection group passed 24 tests in169.48s with the
current production library. It includes fixed-primitive contraction response
(full/direct/projected), native mixed parameter/geometry products, a tiny
neural-XC force-supervision optimizer step in explicit and implicit modes,
and traditional/custom-network potential/kernel checks. This is a bounded
interface/AD fixture, not a pretrained production NNAO campaign or a new
direct primitive-exponent derivative. The original564-AO reconverged
finite-difference gates remain open. The molecular-basis integral-VJP
throughput and CPU-time measurements below are now complete. The earlier complete
72-component gradient is finite but remains unaccepted against those FD
gates. Large qsGW/scGW capacity and native RI geometry remain separate open
limitations; this scheduling work does not claim to solve them.

## Coordinate-product throughput and CPU use

Same synthetic density on the actual phenanthrene/def2-SVP Cartesian basis,
260 AOs, zero screening, 25% exchange, 32 distinct physical cores, no tracing
callbacks. Both processes retain one first execution plus three warm calls.
The full 24-by-3 coordinate product differs by at most 9.3259e-12; it excludes
orbitals/Pulay, XC/grid and nuclear energy and is not a complete force.

| Coordinate integral VJP | Contiguous first-shell partition | Cyclic canonical pair partition |
|---|---:|---:|
| Warm median, seconds | 70.614273 | 22.130468 |
| CPU-time / wall-time, median | 8.9227 | 30.4326 |
| Process peak RSS, bytes | 759,635,968 | 758,890,496 |

Time ratio 0.313399. CPU-time is Linux process user+system time, including
all its threads, divided by synchronized action wall time. It measures
average busy logical CPUs, not an instantaneous thread count. CPU work was
not eliminated: redistribution removed much of the idle tail. The production
coordinate pushforward retains its previous execution path; this scheduling
change specifically targets coordinate pullbacks, including second products.
See `cpu-comparison-114012.json` and raw per-repeat records.

A separate same-process worker sweep pins all 64 allocated logical CPUs,
which map to 32 physical cores. Three warm J/K medians for 8/32/64 workers
were 4.120243/1.692318/1.291669 s. Full J/K arrays differ by at most
2.35e-13 across this sweep. Peak process RSS is 772.7/836.0/888.2 MB; because
this is one process, later peaks include previous compilation and allocations.
Its maximum known native-work bounds are 49.3/190.5/378.7 MB. This supports
using the allocated SMT threads for this kernel, not an automatic universal
64-worker default or an end-to-end speed claim.

## Matched spin-array kernels

These use symmetry-valid synthetic packed s8 inputs, not molecular SCF or
Gaussian-integral preparation. Ten warm samples each, CPU affinity eight
logical CPUs; GPU is the allocation's NV5090. Full output differences are
zero, and the requested ERI/density JVP/VJP and two-step FD checks pass.

| Kernel | Spatial AO | Device | Before median, s | After median, s |
|---|---:|---|---:|---:|
| GKS pure PBE, energy+Fock | 30 | CPU | 0.00941728 | 0.00125921 |
| GKS PBE0, energy+Fock | 90 | CPU | 0.12450311 | 0.08278270 |
| UKS J/K | 30 | CPU | 0.00277731 | 0.00241160 |
| GKS PBE0, energy only | 30 | GPU | 0.00739069 | 0.00738860 |

The GPU energy-only case has no demonstrated warm speed benefit. The small
energy-only flag is retained to match the existing energy/potential request
contract and avoid explicitly requesting unused XC potentials; it does not
justify a GPU speedup claim. No device-specific integral engine was added.
See `spin-comparison-114012.json` for all samples, compilation, memory and
parameter-gradient errors. The 90-AO case tests forward only; array AD was
intentionally limited to 30 AOs.

## Downstream budget compatibility

The final consumer check reproduced a high-budget SCF quota incorrectly
rejecting a valid320 MB response plan (metadata-only384 AO fixture). Commit
`a3631c2` rebinds the response-local private quota and copies only execution
metadata into its preparation view; reference arrays and parameters are shared
and the original reference is not mutated. Both restricted and unrestricted
operator preparation use the planned quota. This adds no numerical kernel.

The integration run passed64 tests and failed two pre-existing hard-coded
minimum-memory assumptions. Running only those two tests on the frozen parent
reproduced both failures:334432 bytes required versus180000/200000 available.
The tests now check correct rejection below fixed native work and GPU cache
policy with adequate space, without changing a production budget or tolerance.
The affected quota/cache group then passed9 tests in6.02 s. Preserve all
three JUnit runs; they are not a single fresh66-pass invocation.

The new564-AO reference independently reconverged from an old saved density
used only as an initial guess:13 cycles421.577 s, actual orbital residual
9.5166e-11, original1e-12/1e-11/1e-10 thresholds. Its full first gradient
returned72 finite entries in937.921 s after18.690 s lowering and12.424 s
compilation, with process peak RSS6390472704 bytes and energy error2.39e-12 Ha.
It differs from the previous finite gradient by at most2.54e-11 Ha/bohr.
The parent1676.422 s execution included a stack sample and a different
tightly converged seed, so no clean speed ratio is reported. Both original
reconverged directional-FD gates remain required; their final status is
tracked separately from these finite/energy checks.

## Final FD status checked on 2026-11-01

The original3e-4 bohr directional FD passed: AD0.00666258005943 versus
FD0.00666263512509 Ha/bohr, absolute error5.5066e-8 under the unchanged
3e-6+4e-5*abs(FD) gate. Its plus/minus reconvergence took4061.395/1606.061s.
The1e-4 bohr pair was stopped by the remaining-lease bound after2035.165s
without a completed endpoint. Finisher exit124 and the release marker are
verified; allocation114012 is no longer running. The second-step gate remains
unaccepted. This is a resource/performance limitation, not a reported finite
derivative disagreement. See `large_geometry_gate.json`.
