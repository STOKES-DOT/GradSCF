# Molecular G0W0 execution and derivative scope

For a large fixed-geometry calculation, keep the SCF representation independent
of the explicitly chosen GW RI approximation:

```python
from gradscf import gto, scf, gw

mol = gto.M(atom="O 0 0 .117790; H 0 .755453 -.471161; H 0 -.755453 -.471161",
            basis="def2-svp", cart=True)
mf = scf.RHF(mol, conv_tol=1e-12).direct_scf().run()
qp = gw.GW(mf, nw=100, max_memory=60000).density_fit("weigend").run(orbs=(3, 4, 5, 6))
print(qp.mo_energy)
print(qp.result.qp_residual)
```

`UGW` exposes the same budget and explicit auxiliary-basis entry. The supplied
SCF is neither replaced nor rerun by `density_fit`. Bare GW exchange uses the
declared GW factors; the subtracted mean-field potential is reconstructed from
the **original** SCF Coulomb representation. Selecting another auxiliary basis
is an explicit approximation change, not an implementation speed comparison.

`max_memory` is decimal MB, defaults to the MF value or the shared 60000 MB,
and controls the known molecular G0W0 forward working arrays. It is not a hard
RSS limit or a guarantee for arbitrary outer AD/model/compiler state. It does
not newly budget qsGW/scGW/periodic GW. Selected MO factors and projected W
outputs must fit; full target-vertex out-of-core streaming is not implemented.
The full packed native 3c input remains an explicit capacity boundary and is
checked on CPU and destination GPU before allocation/transfer.

The default uses bounded dense screening, including the existing small-CPU
RPA fast path when its complete known workspace fits. To request matrix-free
auxiliary products explicitly:

```python
from gradscf.solvers import LinearSolverConfig

config = LinearSolverConfig(method="gmres", rtol=1e-11, atol=1e-13, restart=40)
qp = gw.GW(mf, nw=100, screening_config=config).density_fit("weigend").run(orbs=(3, 4, 5, 6))
```

Explicit configurations currently select GMRES. Omit the configuration for
dense screening; `method="direct"` is rejected rather than silently ignoring
its controls. GMRES has true primal/transpose residual checks and no automatic
dense fallback. Lower memory does not guarantee a faster solve for many RHS.

The eager facades orchestrate concrete references. Differentiable composition
uses `g0w0_cd_restricted` / `g0w0_cd_unrestricted` with live physical arrays.
The optional restricted `v_mf_matrix` (and spin-resolved unrestricted value)
carries the actual starting potential when GW and SCF use different fits.
`reserved_bytes` is an internal caller reservation, not a trainable variable.

At fixed geometry/auxiliary metric, the existing native NNAO coefficient and
log-exponent chain remains differentiable through SCF and the selected QP.
The old small-molecule complete nuclear-response adapter remains available.
The named-RI facade is not a new full nuclear/auxiliary-metric AD interface;
moving auxiliary centers/metric and rank-changing points are not frozen to
pretend otherwise. QP derivatives apply to certified isolated root branches,
not discontinuous changes of root/pole selection.
