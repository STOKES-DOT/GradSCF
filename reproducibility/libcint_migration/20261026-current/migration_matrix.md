# libcint operator and consumer migration matrix

Starting columns describe snapshot7d2e859 (2026-10-26); the final column records completed bounded acceptance on2026-10-27. Sole production interface: libcint.
All values and products are float64. Cartesian and spherical native plans
already exist; molecular DFT facade currently requires Cartesian grids.

| Operator/product | Starting producer / consumers | Starting dynamic leaves / AD | Migration / acceptance |
|---|---|---|---|
| S/T/nuclear V | libcint plan plus duplicate JAX public matrices; SCF/OFDFT/NNAO | independent centers/nuclei; raw coefficients normalized in JAX; exponent normalization live. Coordinate 2nd (Cartesian ordinary S/T/V through4th); coefficient2nd, exponent1st | PASS: one libcint producer, live actual AO conventions; public/binding/geometry and complete SCF response gates. |
| dipole/origin | libcint plan plus JAX public matrices; TDDFT/BSE/properties | coordinates+origin2nd, coefficients2nd/exponents1st | PASS: libcint value, coefficient/exponent and coordinate/origin rules; transition/property response gates preserve origin convention. |
| point/Gaussian rinv | JAX public one_electron only; dft/hfx local-exchange grids | actual callers use dynamic grid origins with static zeta=omega², point when None | PASS: native int1e_rinv with static Gaussian zeta; coordinate/origin through2, coefficient through2 and exponent1 products. No zeta-AD claim. |
| higher multipoles | No current common public producer/consumer found in this audit | not a previously implemented matrix capability here | Do not advertise new arbitrary multipole/AD support |
| ordinary 4c full | libcint plus separate JAX engine; conventional SCF/MP/CI/CC | native coordinates through4th Cartesian, 2nd spherical; native coefficient/exponent full-ERI products absent | PASS: Cartesian coordinate through4 / spherical through2; native coefficient2 and exponent1. S9 symmetry, FD/transpose/thread and existing higher gates. |
| 4c s4/s8 | native compact value, JAX engine alternatives; SCF, AO2MO, OFDFT | native packed coordinate AD missing; traced SCF expands s1 then gathers | PASS: direct packed coordinate1/2, no full-s1 gather; cart/sph, transpose, HVP and fresh-process/installed gates. |
| long-range erf-Coulomb | native full or direct J/K; RSH | fixed omega, coord1st/2nd; omega derivative never promised | PASS: same fixed-omega value and coordinate1/2 in full/packed/direct products; basis/omega and higher unsupported cases remain explicit. |
| direct J/K | native shell contraction and obsolete JAX direct provider; SCF/TDDFT | density+geometry1st/2nd; native dynamic primitive parameters unsupported; projected contractions remain JAX | PASS: bounded value, density/geometry1/2 and projected products; obsolete JAX provider/dispatch removed. Primitive basis products remain unsupported. |
| 2c/3c explicit RI | native auxiliary plans; DF-SCF, Qia, GW/BSE, NNAO | actual NNAO uses fixed aux metric/geometry, orbital coeff2nd/exp1st. General auxiliary-parameter/geometry/mixed native AD absent | PASS: true auxiliary metric/3c values and fixed-aux orbital coefficient2/exponent1 energy training. Moving geometry/auxiliary AD rejects explicitly. |
| ERI spectral factorization | JAX array algebra after 4c; no-aux fallback | differentiable matrix factorization on native-produced ERI | Not a second Gaussian engine; must remain labeled spectral approximation, not true RI |
| selected MO/Qia blocks | shared ao2mo plus native shell slabs or native RI factors; MP2/CC/CI | existing orbital and coordinate1st/2nd products | PASS: selected AO2MO/slabs/RI consumers, R/U MP4 and CC/CI/EOM fixtures; no full-MO/spin expansion introduced by unification. |
| scalar semilocal ECP | pinned PySCF ECP C code, not libcint body; qvszps NNAO | fixed primitive ECP + JAX contraction gives coefficient response; direct ECP geometry/exp/spin-orbit unsupported | PASS for preserved fixed-primitive ECP coefficient route and effective charges/core metadata. Contracted variable-parameter RI/ECP now fails closed; primitive ECP geometry/exponent/spin-orbit remain unsupported. |
| periodic FFT/GTH/AO/grid/XC | existing JAX numerical/grid infrastructure | not a duplicate Gaussian Coulomb integral engine | Retain; do not erase numerical/plane-wave, AO sampling or XC when deleting JAX Gaussian quadratures |

## Actual NNAO workflow requirements

- `NativeJointEngine` learns radial coefficients and shared `(element,l)` beta
  with alpha=alpha_ref*exp(beta), retaining fixed core and channel masks.
- Fixed-pool projection and contracted native paths are both active.
  `PoolRHF` has distinct role-resolved `(scale,log_spacing)` parameters.
- Current training objective is the reconverged RHF energy mean.
  Encoder explicitly sets MACE compute_force=False and compute_stress=False;
  non-vendor NNAO workflows have no force-loss consumer. Existing NNAO.md
  already excludes basis/geometry mixed second derivatives in this joint path.
- Graph positions are Angstrom and shell coordinates bohr. There is no learned
  independent-center head; do not claim its training was tested.
- Real historic trained `best.npz` is absent. Existing small MACE architecture
  and random fixture can validate one complete energy optimizer update, but
  cannot be labeled trained-checkpoint reproduction.

## Registered gates

- Pure reorganization: matrix parity around 2e-12--2e-10 depending operator.
- External total/correlation energy: 1e-7 Ha; TDDFT1e-4 eV;
  matched GW/BSE1e-3 eV; full nuclear gradients1e-5 Ha/bohr.
- NNAO integral coefficient/log-exp directional FD: steps2e-4,1e-4,
  atol2e-8/rtol2e-6. Complete reconverged mean-energy parameter FD:
  atol2e-6 Ha/rtol2e-5. Two geometries0.74/0.91Angstrom.
- Preserve existing 2nd and actual mixed contracts. Unsupported original
  combinations must continue explicit errors, never zero/fallback.

Initial RED evidence: job108635 isolated baseline: public/default/compile/
packed/NNAO17fail1pass; canonical/public products/ECP10fail. Those RED selections are historical; final product/consumer gates are recorded in `implementation_report.md` and `s9_results.json`.

## S9 consumer and AD probe supplement

Baseline is pre-symmetry production commit `22251aa` plus the separately
recorded harness checkpoint. The phase1 starting-state table above is retained;
this supplement describes current source/fixtures, not another numerical PASS.

### 1. Actual ownership and derivative scope

`R` includes the live AO centers and nuclear positions relevant to an operator;
`c/a` are raw AO contraction coefficients / primitive exponents, with the JAX
normalization chain. `C` means supplied MO coefficients, not native Gaussian
coefficients. An integral rule's maximum order does not certify the same order
for an entire electronic-structure method.

Shared product contracts used below: **G4** = ordinary full s1 Cartesian ERI
`R<=4`, spherical `R<=2`, `c<=2/a<=1`; ordinary static Cartesian shell windows
retain `R<=4`, with selected `c<=2/a<=1`. **P4** = s4/s8 geometry through order2,
without native packed c/a response. **JK** = bounded native density action and
coordinate products through order2; projected contractions stay in JAX, while
direct primitive c/a response is not advertised. **RI** = true 2c metric and
packed 3c `(naux,npair)` values; fixed auxiliary metric/geometry allow orbital
`c<=2/a<=1` on the 3c branch. General auxiliary/metric/geometry and mixed RI
derivatives remain unsupported. S/T/V have ordinary Cartesian `R<=4`, other
representations `R<=2`, `c<=2/a<=1`; position dipole and point/Gaussian rinv have
their existing center/origin rules through2, with static zeta. Long-range and
special-operator orders retain their existing narrower contracts.

| Method family / actual consumer | Gaussian producer and layout actually used | Existing method-level AD and boundary | Existing fixture / source anchor |
|---|---|---|---|
| RHF/RKS, semilocal/global hybrid/RSH | `scf.inputs` S/T/V + eager packed ERI or full/direct; moving `MolecularEnergy` uses G4 or explicit HF JK; AO/grid/XC remains JAX | Relaxed R gradient/HVP; HF molecular cubic/quartic fixture; RKS grid/Pulay response and HF/PBE moving-gradient gates. Do not infer all XC fourth derivatives from G4 | `test_molecular_scf_forward.py`; `test_molecular_geometry_higher.py`; `test_molecular_wb97xd_response.py`; `tools/.../geometry_forward.py` |
| UHF/UKS and existing ROHF/ROKS | Same real Gaussian S/T/V and full/packed/JK data, with spin-resolved densities or common spatial frame | UHF relaxed R1/R2 and UKS NH R1/dipole; finite AO/grid input derivatives for ROKS. No new RO excited-response interface | `test_molecular_unrestricted_response.py`; `test_scf_orbital_native_geometry.py`; `test_scf_orbital_dft_gradients.py`; `test_scf_rohf_roks.py` |
| GHF/GKS | Real spatial S/T/V/ERI promoted to complex spinor array algebra; grid AO remains JAX; no independent spinor Gaussian engine | Existing complex/spin-rotation value tests and fixed-input orbital-minimizer energy/density derivatives. Generalized TDA/TDDFT remains explicitly unsupported; no new full molecular GKS R-response claim | `test_scf_ghf_gks.py`; `test_scf_orbital_dft_gradients.py[implicit-gks_col/implicit-gks_ncol]` |
| TDA/TDDFT/TDHF/CIS, R/U | S/T/V + full/packed or JK response action, position dipole for strengths; semilocal response uses moving grid/AO | Relaxed excitation R1, JVP/VJP and displaced FD through SCF/operator/root; CIS transition/density response includes amplitude response. Existing selected CIS(D) uses AO2MO slabs and R1/R2 | `test_direct_excitation_geometry.py` H2/H2+/HeH; `test_cis_transition_response.py`; `ci/test_cis_d_geometry.py` |
| CI/CISD/... and FCI | `ao2mo` selected normal-order blocks / active-space four-index output from G4/P4/JK or supplied arrays; frozen core adds native core J/K | CI selected R1/R2 fixture; FCI native preparation R/C/projection JVP/VJP/HVP is a supplied-frame partial; FCI energy/density spectral response is a separate fixed-integral gate. Only an explicit `CorrelatedEnergy` geometry map includes HF relaxation | `ci/test_selected_integrals.py`; `ci/test_geometry.py`; `fci/test_native_reference.py`; `fci/test_response.py` |
| Conventional R/U MP2 | `mp.integrals.from_orbitals` requests OVOV blocks, native shell slabs or P4/full; orbital resolvent remains array algebra | Complete implicit HF -> MP2 geometry gradient/HVP, plus given-frame C/F/ERI partials. Native slab R2 limits this selected route, independently of full G4 | `test_molecular_correlated_response.py`; `mp/test_native_gpu_geometry.py`; `mp/test_orbitals.py` |
| True RI-MP2 / DF-SCF | Native RI2c/RI3c -> explicit auxiliary factors -> selected OV factors; not spectral compression of 4c | Value/amplitudes and factor/orbital-array derivatives. Native RI geometry/auxiliary derivatives are not supplied by freezing the metric | `worker.py method=dfmp2` with `weigend`; `mp/test_mp2.py`; `test_density_fitting_rks.py` |
| MP3+ recursive MP | `mp.series.run_from_orbitals`, selected normal-order kernels from slab/P4/full or supplied arrays; finite wavefunction histories retained | Existing selected R/U MP4 R1/R2; N2 MP4/MP6 independent Rayleigh-Schrödinger gradient fixture. Perturbation order is distinct from derivative order | `mp/test_selected_series_geometry.py`; `mp/test_series_geometry.py`; `mp/test_series.py`; `examples/mp/implicit_gradient.py` |
| CC/CCSD, CCSD(T), R/U | Selected seven-block preparation + dense or bounded bare-virtual action; native slab/JK products; streamed triples panels | Complete HF/amplitude response R1/R2, Lambda/density/two-body contractions; streamed (T) JVP/VJP through2 and explicit rejection of third. Full ordinary MP2/CCSD mixed Raman fixture is separate from slab R2 | `cc/test_scf_unrestricted_geometry.py`; `cc/test_streamed_triples_geometry.py`; `cc/test_two_body_geometry.py`; `test_molecular_correlated_response.py` |
| EOM-EE/IP/EA and Dyson | Reuses prepared CC blocks / `VirtualERI` native action, no method-owned integral scheduler | Existing native right/left action parity, ground + C partial root derivative; fixed-MO EOM/Dyson response includes required ground/Lambda/eigenvector dependencies. These are not a newly validated relaxed nuclear EOM map | `cc/test_native_eom_action.py`; `cc/test_eom.py`; `cc/test_eom_dyson.py` |
| Molecular GW/BSE, R/U | GW facade reuses existing DF factors or spectrally factorizes P4/full; BSE reuses MO factors + dipoles. `MolecularGW` instead rebuilds G4 -> differentiable spectral factors | Existing RHF relaxed G0W0/evGW0/evGW geometry R1 and BSE optical R1; UGW/BSE and qsGW/scGW response fixtures vary explicit MO/factor inputs. Do not relabel these as RI metric geometry or full nuclear response | `test_molecular_gw_response.py`; `gw/test_gw_outer_response.py`; `gw/test_gw_scgw_response.py`; `bse/test_unrestricted.py`; `worker.py method=gwbse` |
| Gaussian molecular OFDFT | `ofdft.representations.gaussian`: native S/T/V and P4; AO/gradient/quadrature and kinetic/XC remain JAX | Existing fixed-Gaussian-input potential/density/number response and Taylor order2. A new molecular nuclear-force contract is not inferred from this potential-response gate | `dfpt/test_ofdft.py::test_molecular_gaussian_ofdft_response` |
| NNAO fixed-primitive / native joint / fixed-pool | Native S/T/V + RI2c/RI3c; fixed primitive projection is JAX; joint c/log-alpha uses the fixed auxiliary metric; qvszps retains separate pinned scalar ECP | Actual theta/beta energy-loss gradient + optimizer/checkpoint fixture; native 3c c2/a1 tests; native mixed basis/geometry and force-loss training are not upgraded. Cache identity is metadata, not an AD gate | `nnao/test_libcint_consumers.py`; `nnao/test_integral_paths.py`; `nnao/test_libcint_cache_identity.py`; `nnao_step.py` |
| Periodic FFT/GTH HF/DFT/TD/GW/BSE/OFDFT | **N/A for molecular libcint 4c scheduling**: reciprocal-space FFT/GTH Coulomb/pseudopotential and AO sampling stay in their existing periodic implementation | Existing periodic force/response contracts are controls, not libcint derivative-speed claims; periodic Gaussian OFDFT also must be classified by its actual reciprocal producer | `pbc/test_gradients.py`; `gw/pbc/test_full_response.py`; `dfpt/test_ofdft.py` periodic cases |

### 2. Minimal representative groups

The thin `tools/libcint_migration/probes.py` catalogue records the exact
selected existing entries. It does not introduce another SCF, response,
amplitude, QP or Gaussian formula. A family is covered by one distinct native
calling path plus existing method assertions, rather than method x functional
x spin x representation multiplication.

| Group | Required existing entry / smallest useful distinction | Physical outputs and response scope |
|---|---|---|
| `scf` | Worker RHF/PBE fresh object; molecular HF/PBE moving-geometry gates | E, D, F, SCF residual/cycles; relaxed R gradient; HF R3/R4 safety separately |
| `spin` | NH UHF/UKS geometry fixture; GKS collinear facade + complex orbital/input-response gates | Spin electron counts/densities, E, gradients; GKS input partial explicitly labeled |
| `excitation` | H2 RKS-PBE TDA and full TDDFT; HeH both-spin TDHF; CIS transition gate | Excitations/strengths, relaxed R1, transition-amplitude response; generalized rejection control |
| `ci_fci` | Selected R/U CI chart + native FCI core/active preparation + fixed-integral FCI eigenpair response | CI/FCI E/roots/densities and preparation JVP/VJP/HVP; supplied-frame versus relaxed labels retained |
| `mp2_ri` | Existing worker MP2 and true `dfmp2`; direct MP2 CPU/CUDA geometry gate | E_corr/E_total/t2 shape; conventional relaxed R1/R2; RI fixed-factor partial/value |
| `recursive_mp` | Selected H2-series/H3-tensor MP4; generic series fixed-input gradient/HVP | Full requested order/history, E and relaxed R1/R2, not a lower-order replacement |
| `cc_eom` | Worker CCSD/(T); selected UCCSD(T) geometry; native EOM EA response / EE-IP-EA action parity | CC amplitudes/residuals, nonzero (T), geometry HVP; EOM C partial and left/right actions |
| `gw_bse` | Existing full-window formal worker configuration unchanged; small MolecularGW geometry/optical and UGW/outer/scGW factor-response fixtures | Same QP targets/G/W windows, residuals/weights/masks, BSE roots/strengths; geometry versus MO-input partial labels |
| `ofdft` | Molecular Gaussian OFDFT existing DFPT potential gate | E, density, N/mu and potential response/Taylor coefficients; fixed Gaussian inputs |
| `nnao` | Unchanged real-MACE `nnao_step.py` full-step + components in separate processes | Loss/gradient/optimizer step, per-phase transfer/kernel/response time, final checkpoint check; random fixture status preserved |
| `periodic_na` | Existing FFT/GTH geometry/response fixture as a control | Native 4c count is N/A, not a fabricated zero-speedup measurement |

### 3. Counter and timing protocol

Profiling-only adapter contract is `reset()` / `snapshot()`, backed when
available by private `gradscf.integrals._native.diagnostics.reset_counts` and
`get_counts`. Preserve the raw `libcint-call-counts-v1` snapshot: enabled flag,
build kind, actual library SHA, per-family `one_electron/eri4c/ri3c/ri2c/ecp`
evaluations/cache_queries and optional scheduled_quartets/emissions. Do not
infer an integral evaluation count from FFI calls or assume an eightfold factor
for a differentiated leg pattern. Missing/disabled instrumentation is reported
as unavailable, not as zero calls.

Compile/prewarm first; then reset -> execute the unchanged task -> synchronize
all requested outputs with the existing `worker.ready` boundary -> snapshot.
Validation-fixture counters include their explicit FD/reference/transpose
work and are labeled as such; they are not per-single-gradient counts or clean
timings. The native product-level counter gates separately establish orbit
multiplicities. NNAO full-step/component counters include the explicitly
recorded model/native transfer boundaries; whole-fixture counts cannot be
presented as one molecular step.

Clean speed tables use independent production/default-OFF processes and the
existing first + three warm protocols; counter/profile runs are separate.
Preserve exact input JSON, probe/source/native hashes, source revision, device,
CPU affinity/thread counts and output scopes. Record isolated-process RSS as a
high-water mark including compilation, not as a live tensor-memory bound.
Use the same new n006/5090 node and input lineage for before/after; n005 phase1
timings are context only.

### 4. Acceptance and non-upgrades

Reuse the phase1 numerical tolerances above and the exact selected test's
registered FD steps/tolerances. The probe catalogue records completed,
nonconverged, failed, skipped and unsupported states separately. Existing
pytest oracles remain checks; a passing metadata/fixture assertion does not
mean saved full spectra, whole-process memory or all-family AD has passed.
Formal GW/BSE windows, all requested outputs and 6000 s/case controls remain
owned by the existing controller; a small AD fixture never substitutes them.
No automatic reduced case, new derivative order, frozen auxiliary/physical
leaf, new functional, or altered stationarity/solver threshold is introduced.
