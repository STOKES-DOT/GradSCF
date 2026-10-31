# DFT execution: CP2K, PySCF and GradSCF

Scope: molecular LDA, GGA, global hybrids and range-separated hybrids, including
geometry response. The performance measurements remain those in `report.md`;
this source comparison does not establish a CP2K/PySCF speed ratio.

## Source provenance

CP2K official repository: <https://github.com/cp2k/cp2k.git>, commit
`d077e66fe06a862d49bdcc2b91c179463e9487eb`. A bulk partial checkout failed
with RPC/early EOF. Eight selected DFT source files were then downloaded from
the same pinned official revision and individually checked against their Git
blob IDs. The local reference is `artifacts/reference/cp2k-selected/`, with
SHA256 and byte counts in its manifest. The code is reference material and is
not a GradSCF production dependency or integral backend.

PySCF source links below refer to the official published source. The installed
comparison environment is 2.13.0; no claim is made that a changing online page
is byte-identical to that installation.

## Per-family execution

| Family | Work that is required | Reference strategy | Current GradSCF finding |
|---|---|---|---|
| LDA | Coulomb J, AO values, density and local XC | Request J only; evaluate AO order 0 | direct/DF and normal RKS pair-J already do this; packed fallback flags required a small correction; dense assembly still retains DAO |
| GGA | LDA work plus density gradients | AO order 1, reuse blocked contractions and buffers | budgeted streaming exists; dense AO-D contraction reuse needs an actual HLO/timing probe |
| Hybrid | J, weighted full-range K, local XC | Incremental J/K, robust initial density, bounded integral storage or explicit RI | unique-quartet native patch measured; RI-initialized final-direct reference measured on 564 AO |
| RSH | Full Coulomb J and the declared SR/LR exchange combination | Separate operator contexts and request only nonzero exchange pieces | direct recipe already skips zero pieces correctly; full/LR dense representation and absent RSH RI-JK remain important boundaries |

PySCF's `get_veff` selects J-only for pure functionals, uses the density
increment when valid, and combines the correct full/LR exchange terms. Its
integrator uses different AO derivative orders for LDA and GGA and plans grid
blocks from available memory. These are execution contracts, not interchangeable
functional definitions. [RKS source](https://pyscf.org/_modules/pyscf/dft/rks.html),
[numerical integration source](https://pyscf.org/_modules/pyscf/dft/numint.html).

In CP2K, density/grid construction and the Poisson solve are separate parts of
the KS update. Its GPW/GAPW representation is a different discretization from
the present molecular four-center Gaussian path. It is useful guidance for
the periodic/grid infrastructure, rather than a local replacement for J.
[Pinned KS source](https://github.com/cp2k/cp2k/blob/d077e66fe06a862d49bdcc2b91c179463e9487eb/src/qs_ks_methods.F#L447),
[density source](https://github.com/cp2k/cp2k/blob/d077e66fe06a862d49bdcc2b91c179463e9487eb/src/qs_rho_methods.F#L370).

## What is already implemented and measured

- Native direct J/K now distributes small batches by unique shell quartets
  with a 64 MiB private-output bound. On the same 260-AO input, CPU32 warm JK
  medians changed from 4.721 to 1.755 s (ordinary) and 5.092 to 1.724 s (LR).
  These are kernel results, not complete DFT speedups.
- On phenanthrene/def2-TZVP, PBE0's final direct reference needed 114 cycles
  from the original guess. An explicit RI-PBE0 starting-density calculation,
  followed by the same final-direct equations and tolerances, reduced this to
  35 cycles. Including seed cost, cold reference time changed from 3432.6 to
  1140.0 s; final energy differs by 4.94e-11 Ha. This is an initial-density
  experiment with all preparation charged, not replacement of the final
  conventional target by RI.
- `forward="scf"` now validates converged DIIS orbitals and attaches the
  original implicit response without another orbital minimization. Eight
  tests cover nonzero-field PBE0/RSH gradients, HVP and two FD steps. Its
  large-system compile/runtime result is still being evaluated.
- Pure packed contractions now propagate J-only flags to opaque native calls;
  UKS also requests only K in its spin-exchange calls. Twenty-six tests cover
  values, density JVP/VJP, packed selection and UKS/DF-hint regression. This
  patch adds no net production lines. The normal pure-RKS direct/DF/pair-J
  paths were already correct.

## Most useful CP2K details to adapt

The pinned HFX source determines the OpenMP team size once, enters one
parallel region, and declares primitive/contraction work buffers private.
When dynamic load balancing is enabled, its workers take task bins from a
shared counter under a short critical section. Per-bin wall times are retained
and the load distribution can be updated in subsequent SCF iterations. This
is explicit scheduling of integral work, not a nested BLAS/OpenMP pool per
shell quartet. See hfx_energy_potential.F lines 336, 352, 907, 1146 and 1169
in the pinned reference; static assignment is also supported.

GradSCF already uses JAX's supplied FFI thread pool and asynchronous promises.
Adding another OpenMP pool inside libcint callbacks would introduce a second
resource owner. First measure the existing pool's requested versus available
workers. Its current private-output ceiling is 64 MiB per direct J/K call;
at 564 AOs the calculated upper limits were 13 workers for one-density JK and
6 for two-density JK. A later disposable diagnostic confirmed the actual
selected count 13 with both requested and available pool counts 32. This is
a selected worker count, not a measure of continuously active CPU threads.
The whole allocation remains capped by affinity; BLAS/OpenMP use one thread
in the present integral probes. Matrix/XC kernels and native work must share
the process budget rather than independently requesting all CPUs.

CP2K's documented HFX allowance is per MPI process, shared across its OpenMP
threads. It includes helper buffers before any integral-storage remainder.
GradSCF's method max_memory uses decimal MB, while this native private cap
uses binary MiB. A larger allowance must pass through the existing shared
workspace accounting; raising the native constant alone would invalidate
low-memory caller bounds. Also preserve deterministic summation: direct
dynamic work stealing is not automatically compatible with the current
bitwise worker-repeatability tests.
[HFX memory contract](https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/DFT/XC/HF/MEMORY.html).

CP2K's HFX implementation subtracts Fock/density matrices, private primitive
buffers and screening helpers from the module memory allowance before assigning
the remainder to integral storage. It reuses compressed integrals with an
explicit geometry-change boundary. This is stronger lifecycle guidance than a
cache limited only by entry count.
[HFX source](https://github.com/cp2k/cp2k/blob/d077e66fe06a862d49bdcc2b91c179463e9487eb/src/hfx_energy_potential.F#L655),
[memory contract](https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/DFT/XC/HF/MEMORY.html).

Its screening code computes block density bounds. Such screening is a numerical
approximation and is distinct from the present basis-only linear J/K action.
GradSCF's independent trial-density response must retain its existing linear
contract; a primal-dependent mask cannot be silently reused as an exact JVP/VJP.
[Screening source](https://github.com/cp2k/cp2k/blob/d077e66fe06a862d49bdcc2b91c179463e9487eb/src/hfx_screening_methods.F#L113).

CP2K prepares RI two-/three-center tensors before the SCF contractions and
uses distributed sparse tensor operations. Its batching controls reduce
intermediate workspace, not the size of every retained tensor. GradSCF should
apply the same distinction to RI factors and conversion lifetimes.
[RI source](https://github.com/cp2k/cp2k/blob/d077e66fe06a862d49bdcc2b91c179463e9487eb/src/hfx_ri.F#L195),
[RI documentation](https://manual.cp2k.org/trunk/methods/dft/hartree-fock/ri_gamma.html).

ADMM performs exchange in a smaller projected auxiliary space with a correction.
It is an additional method, not an exact scheduling change. It needs separate
energy/force validation and will not be used silently to improve the existing
hybrid benchmark. [ADMM documentation](https://manual.cp2k.org/trunk/methods/dft/hartree-fock/admm.html),
[pinned implementation](https://github.com/cp2k/cp2k/blob/d077e66fe06a862d49bdcc2b91c179463e9487eb/src/admm_methods.F).

## Next bounded work, in priority order

The accepted local implementation uses a real Brillouin residual for certified,
unregularized stationary states. With the present generator convention,
`g[p,q] = 2 sum_s (n[s,q]-n[s,p]) (C.T F[s] C)[p,q]`, where F is the
**actual scalar energy's density gradient**, not an independently clipped
reported potential. This targets repeated differentiation of the dense Cayley
chart. The same-zero-set statement is local to a nonsingular chart at a
stationary solution; unchecked and regularized policies retain the original
chart definition. The later spin extension applies the corresponding complex body residual
to certified GKS stationary states, with conjugate-transpose and real/imaginary
packing tested separately; regularized/unchecked policies retain the chart. The original envelope rule
and implicit solver remain in use. Toy linearizations and full response tests
are necessary; a shorter source expression alone proves no speedup.
[PySCF orbital-gradient implementation](https://pyscf.org/_modules/pyscf/scf/hf.html#get_grad).

1. Finish large geometry acceptance; distinguish lowering,
   compilation, first force and reconverged FD. A successful SCF or compile
   alone does not pass this gate.
2. Same-geometry LDA/GGA forward measurements are complete. Quantify dense DAO ownership and XC
   contraction costs before changing derivative demand or introducing an
   occupied-factor shortcut. Retain current grid accuracy.
3. Evaluate call-local, budgeted HFX preparation/cache reuse and RSH true RI
   as separate candidates. Record retained storage, numerical workspace,
   conversion cost, actual memory and target accuracy independently.

All candidates must preserve fixed-density, explicit and implicit training;
geometry JVP/VJP/HVP; and custom/neural input requirements. Classic LDA needing
only rho does not imply an external model needs only rho. NNAO cache signatures
must include primitive exponents, normalized coefficients, independent centers,
auxiliary parameters and operator controls when those quantities can change.
A geometry-only invalidation flag from a fixed-basis program is insufficient.
No new Gaussian integral engine or silent fallback is proposed.

## Implemented CPU follow-up

See `CPU_RESOURCE_BALANCE.md`: existing method memory now binds direct J/K
private storage, and coordinate VJPs distribute canonical outer-pair tasks
cyclically on the same FFI pool. The matched 564-AO prepared Fock median is
39.9658 -> 26.0069 s with a 3.13e-13 Ha maximum full-Fock error. This is
not a complete DFT or force speed ratio. Native/consumer regression groups
pass; molecular derivative scheduling throughput and final large FD gates
remain separately tracked in the plan.
