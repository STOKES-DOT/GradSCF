# Molecular GW controls

```python
from gradscf import gto, scf, gw, bse

mf = scf.RHF(gto.M(atom='H 0 0 0; H 0 0 .74', basis='sto-3g')).run()
mygw = gw.GW(mf, method='evgw0', nw=100,
            max_cycle=80, conv_tol=1e-9).run()
print(mygw.mo_energy)             # Ha
print(mygw.result.qp_weight)      # local frequency derivative, dimensionless
response = bse.BSE(mygw, nroots=1, tda=False).run()
```

The restricted facade supports `g0w0`, `evgw` and `evgw0`. The latter two
reuse the existing eigenvalue fixed-point loop. `evgw0` fixes the original
mean-field spectrum in W while updating G; `evgw` updates both. Orbitals and
mean-field exchange subtraction stay fixed. W0 is currently recomputed from
its fixed inputs, rather than stored between iterations. `max_cycle`,
`conv_tol` and `damp` configure only the outer eigenvalue loop. Outer-loop
AD/JIT is not implemented; an inner implicit QP derivative is not its substitute.

## Three independent windows

- `run(orbs=...)`: which QP levels to correct; others keep their MF values.
- `g_orbitals=...`: intermediate orbitals in the correlation Green's function.
- `screening_occupied=...` and `screening_virtual=...`: transitions in W.

Indices are zero-based original MO indices, not renumbered active-space
indices. `None` means all allowed indices; an empty tuple removes the
corresponding correlation sum. These windows do not truncate mean-field J/K
or change electron count. All bare MO factors are still formed and stored.
Changing the optical window later in BSE does not alter W.

For example, excluding the first doubly occupied core orbital from G and W
uses `g_orbitals=range(1,nmo)` and `screening_occupied=range(1,nocc)`.
Explicit windows avoid element-dependent implicit frozen-core rules.

The restricted low-level `g0w0_cd_restricted` takes the same windows plus an
independent `screening_energy` spectrum. Its default is the existing G pole
spectrum, preserving older calls. `evgw_cd_restricted(update_w=False)` exposes
the fixed-W variant without another iteration implementation. These additions
currently target the real closed-shell molecular route; unrestricted/periodic
window controls and new self-consistency methods are not implied.

## Quasiparticle diagnostics and provenance

`GWResult` records QP energies, self-energy, residuals, computed/converged masks,
W energies, original-MO correlation windows and the producing method.
`qp_weight = 1/(1-d Re Sigma/d omega)` is evaluated at returned energies with
G/W fixed. Unrequested levels and singular slopes have NaN weights. A finite
weight does not certify root convergence or guarantee a physical main peak;
negative weights are not clipped. This is not a multi-root search or an
automatic largest-weight satellite-selection algorithm.

The native BSE adapter inherits the actual W spectrum and windows. G0W0/evGW0
therefore give W0; converged evGW supplies its updated W. Missing legacy
metadata retains the prior explicit-reference rules. Cached GW/BSE properties
are invalidated by source/settings changes; failed GW reruns clear their result.

The QP implicit VJP and fixed-input G0W0/BSE derivatives retain their existing
contracts. See [BSE properties](../bse/README.md), [scGW](SCGW.md), and
[executed MolGW comparisons](../bse/VALIDATION.md). COHSEX, open-shell BSE,
periodic BSE, analytic continuation of scGW, and distributed execution remain
outside this increment.

## Algorithm references

Window separation and fixed-W iteration were checked against MolGW 3.4 source
revision `b831818d7a845c36f295d036dc8ceef59daf9991`:
[m_selfenergy_evaluation.f90](https://github.com/molgw/molgw/blob/b831818d7a845c36f295d036dc8ceef59daf9991/src/m_selfenergy_evaluation.f90),
[m_selfenergy_tools.f90](https://github.com/molgw/molgw/blob/b831818d7a845c36f295d036dc8ceef59daf9991/src/m_selfenergy_tools.f90).
GradSCF retains its JAX contour-deformation implementation and shared solvers.
The executable comparison uses MolGW's independent analytic spectral route;
its graphical QP weights require a separate frequency-grid convergence check.


## Optional QP forward solvers

`GW(mf, qp_solver='newton')` and `GW(mf, qp_solver='hybrid')` use AD frequency
slopes for the inner G0W0 root calculation. `UGW` accepts the same choice.
The default remains secant; the measured small CPU cases do not show a speedup.
Forward method selection does not change the implicit backward. See
[algorithms, safeguards and measured costs](QP_SOLVERS.md).
