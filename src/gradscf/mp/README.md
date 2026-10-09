# Molecular Moller--Plesset methods

`gradscf.mp` implements real canonical RHF MP2/MP3 and collinear UHF MP2.
The eager interfaces reuse a converged GradSCF HF calculation:

```python
from gradscf import gto, scf, mp

mol = gto.M(atom="O 0 0 0; H 0 .75 .58; H 0 -.75 .58", basis="6-31g*")
mf = scf.RHF(mol, conv_tol=1e-12, conv_tol_grad=1e-10).run()
pt = mp.MP2(mf, frozen=1).run()
print(pt.e_corr, pt.e_tot)
pt3 = mp.MP3(mf, frozen=1).run()
print(pt3.e2, pt3.e3, pt3.e_tot)
```

`MP2` dispatches to `RMP2` or `UMP2`; `MP3` currently dispatches only to `RMP3`.
Explicit `scf.reference.RestrictedReference` and `UnrestrictedReference` MO
Hamiltonians are also accepted. ROHF, noncanonical orbitals, complex orbitals,
fractional occupations, UMP3, MP4 and relaxed/unrelaxed densities are not part
of this version. The orbital frame must diagonalize the **physical** HF Fock
matrix; a level shift is never added to MP denominators.

## Array API and differentiation

```python
config = mp.MPConfig(order=3, with_t2=False)
result = mp.run_mp(h1_mo, eri_mo, nocc=nocc,
                   nuclear_repulsion=enuc, frozen=1, config=config)
```

Restricted `h1` and `eri` are real MO tensors in chemists' notation. For UMP2,
`h1=(ha,hb)`, `eri=(gaa,gab,gbb)`, and `nocc=(nalpha,nbeta)`. Occupied orbitals
precede virtual orbitals in each spin channel. `nocc`, `frozen` and configuration
are static controls; numerical Hamiltonian arrays remain differentiable.

MP2 algebra and the MP3 interaction action use JAX AD, including JIT, JVP/VJP
and higher derivatives on valid inputs. No iterative MP amplitude fixed point
is introduced for canonical orbitals. A complete parameter derivative still
needs the HF state and integral response; the eager facade is not a traced
SCF-to-MP function. `examples/mp/implicit_gradient.py` demonstrates the full
chain using the existing implicit orbital HF solver and reconverged finite
differences. It differentiates a dimensionless contraction coefficient in H2,
not nuclear coordinates.

## Results and validity

- `e2` and `e3` are individual perturbation orders; `correlation_energy` is
  their sum, and `total_energy = reference_energy + correlation_energy`.
- `t2` means **first-order** doubles amplitudes for MP2 and MP3. Restricted
  amplitudes have shape `(nocc,nocc,nvir,nvir)`; unrestricted amplitudes are
  `(taa,tab,tbb)`, with antisymmetric same-spin blocks. `with_t2=False` omits
  these output tensors.
- `same_spin_energy` and `opposite_spin_energy` (facade `e_corr_ss/e_corr_os`)
  decompose **E2 only**, including when E3 is requested.
- `canonical_error` is the maximum absolute off-diagonal physical Fock entry
  before freezing. Its default threshold is `1e-8 Ha`.
- `min_abs_denominator` diagnoses doubles denominators, excluding Pauli-forbidden
  same-spin entries. Its default threshold is `1e-10 Ha`. Empty active doubles
  spaces give zero energy and an infinite minimum denominator.
- `valid=False` returns NaN total/correlation energy; eager `kernel()` raises.
  Reference values, diagnostics and still-valid lower-order components may
  remain available. Differentiation is supported only on valid inputs.
  Small denominators are diagnosed, never silently clipped or shifted.

`frozen` follows the shared post-HF convention: a core count or an orbital
index list, optionally separate alpha/beta lists for UMP2. Frozen occupied
electrons still contribute to the Fock and HF reference energy; frozen virtual
orbitals are excluded only from the correlated active space.

## Integral storage

MP2 facade transforms only occupied--virtual blocks from dense/s4 AO data.
For DF it retains `B[Q,i,a]` per spin and contracts one occupied-index slice
at a time. With `with_t2=False`, neither full MO ERIs nor the full MP2 doubles
tensor are constructed by the DF evaluation. The loop is checkpointed for AD.
This is an algorithmic storage property, not a measured GPU peak-memory claim.
The dense/s4 path still stores the selected `ovov` block. Direct-only SCF
without saved two-electron data is explicitly rejected.

MP3 is currently an **in-core reference implementation**. It reuses the
existing full-MO post-HF adapter and restricted CC integral blocks; DF input
is expanded to full MO ERIs on this path. `with_t2=False` suppresses the output
but MP3 still needs first-order doubles internally. Streaming MP3 is a future
extension, not implied by MP2 DF support.

## Validation

`tests/mp` checks RHF water/6-31G* MP2 and UHF H3/STO-3G MP2 against independent
PySCF energies/amplitudes, with Cartesian orbitals and CPU float64. Tests also
cover frozen orbitals, selected dense/packed/DF transforms, pure JAX gradients
and HVPs, stale references, physical denominator validity and empty spaces.
MP3 on H2/3-21G and LiH/STO-3G is checked against independent determinant-space
Rayleigh--Schrodinger coefficients using PySCF FCI Hamiltonian actions, rather
than the CC expressions used by the implementation. Complete implicit-HF
parameter derivatives are checked against independently reconverged finite
differences.

Run from the repository root after building the native CPU integral library:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest -q tests/mp
PYTHONPATH=src JAX_PLATFORMS=cpu python examples/mp/molecular.py
PYTHONPATH=src JAX_PLATFORMS=cpu python examples/mp/implicit_gradient.py
```

See [REFERENCES.md](REFERENCES.md) for equations and attribution.
