"""Small molecular GW/BSE geometry maps using GradSCF's existing methods."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from gradscf import bse, gw, scf
from gradscf._memory import array_device, resolve_memory_mb, tree_storage_bytes, workspace_bytes
from gradscf.integrals.molecular.factorization import eri_pair_matrix_to_df_factors_traceable
from gradscf.gw.conventional import _coulomb
from gradscf.gw.g0w0 import _exchange_mo, _mo_factors
from gradscf.solvers import EigenResponseConfig, EigenSolverConfig, solve_hermitian
from .molecular_response import MolecularEnergy


class MolecularGW:
    """Real Aufbau RHF/RKS -> molecular GW, with moving native inputs.

    Coordinates are in bohr and energies in Hartree. All AO integrals, the
    metric, KS orbitals, QP roots and screening are rebuilt inside the map.
    Canonical MO roots must be isolated; orbital crossings are not assigned
    individual derivatives. Full-shape symmetric factors of the exact AO-pair
    Coulomb matrix restrict this helper to small molecules, including when the
    KS solve uses direct J/K. This is not auxiliary-basis RI geometry AD.
    Factor eigenvalues must remain separated from the fixed truncation threshold.

    evGW0 holds W0 fixed only during its QP iteration. W0 still depends on the
    geometry. evGW updates both G and W. Both use the existing outer implicit
    response; neither differentiates the numerical fixed-point trajectory.
    ``max_cycle`` and ``damp`` control that trajectory; ``conv_tol`` controls
    the converged outer root (default 1e-10). It is separate from G0W0's inner
    ``qp_tol`` and the unchanged 1e-10 implicit linear-response tolerance.
    A QP residue is a local frequency derivative, not an EOM Dyson weight.

    qsGW and finite-T Matsubara scGW use the KS frame to initialize their full
    implicit self-consistency. scGW returns an SCGWResult with no QP poles;
    beta/mixing replace real-axis eta and outer damping. Its total energy uses
    bare nuclear repulsion, without the starting DFT dispersion correction.
    max_memory bounds known preparation and CD/qsGW arrays, not arbitrary AD,
    compiler residency or the scGW/BSE working arrays. Original full/LR ERIs need
    CPU placement; the moving reference otherwise follows MolecularEnergy.
    """

    def __init__(self, mf, *, method="g0w0", nw=100, eta=None, qp_solver="secant", qp_tol=1e-11,
                 factor_tolerance=1e-12, orbs=None, g_orbitals=None,
                 screening_occupied=None, screening_virtual=None, qp_initial=None,
                 max_cycle=100, conv_tol=1e-10, damp=.3, diis_space=0, beta=20., mixing=.5,
                 jk_backend="full", max_memory=None, forward="minimize"):
        if np.asarray(mf.mo_coeff).ndim != 2 or np.iscomplexobj(mf.mo_coeff):
            raise NotImplementedError("This geometry adapter requires a real restricted RHF/RKS reference.")
        method = str(method).lower()
        if method not in {"g0w0", "evgw0", "evgw", "qsgw", "scgw"}:
            raise ValueError("Choose g0w0, evgw0, evgw, qsgw or scgw.")
        if not np.isfinite(conv_tol) or conv_tol <= 0:
            raise ValueError("conv_tol must be finite and positive.")
        if method == "g0w0" and conv_tol != 1e-10:
            raise ValueError("G0W0 has no outer conv_tol; configure its inner qp_tol instead.")
        if qp_solver not in {"secant", "newton", "hybrid"} or (method != "g0w0" and qp_solver != "secant"):
            raise ValueError("QP solver selects only the inner G0W0 root method.")
        if not np.isfinite(factor_tolerance) or factor_tolerance <= 0:
            raise ValueError("The factor truncation tolerance must be positive and finite.")
        windows = dict(orbs=orbs, g_orbitals=g_orbitals,
                       screening_occupied=screening_occupied, screening_virtual=screening_virtual)
        if method in {"qsgw", "scgw"}:
            if any(value is not None for value in windows.values()):
                raise ValueError("qsGW/scGW require the complete orbital space.")
            if qp_initial is not None or diis_space != 0 or qp_tol != 1e-11:
                raise ValueError("qsGW/scGW do not use diagonal-QP seeds, DIIS or qp_tol.")
        if method == "scgw":
            if eta is not None or damp != .3:
                raise ValueError("Matsubara scGW uses beta/mixing, not eta/damp.")
        elif beta != 20. or mixing != .5:
            raise ValueError("beta and mixing configure only Matsubara scGW.")
        self.max_memory = resolve_memory_mb(max_memory, mf)
        self.reference = MolecularEnergy(mf, gradient_tolerance=1e-10, jk_backend=jk_backend,
                                         max_memory=self.max_memory, forward=forward)
        self.coordinates = self.reference.coordinates
        self.method, self.nw = method, nw
        self.eta = None if method == "scgw" else (1e-5 if eta is None else eta)
        self.beta, self.mixing = beta, mixing
        self.qp_solver = qp_solver
        self.qp_initial = qp_initial
        self.qp_tol = qp_tol
        self.max_cycle, self.damp = max_cycle, damp
        self.conv_tol = float(conv_tol)
        self.diis_space = diis_space
        self.nmo = mf.mo_coeff.shape[1]
        self.nocc = int(np.count_nonzero(np.asarray(mf.mo_occ)))
        self.factor_tolerance = factor_tolerance
        self.pairs = np.tril_indices(mf.mo_coeff.shape[0])
        self.windows = windows
        self.outer_response = scf.SCFDifferentiationConfig(tolerance=1e-10, max_iter=100, restart=40)

    def _prepare(self, coordinates):
        state, inputs = self.reference.solve_with_inputs(coordinates)
        coefficients = state.mo_coeff
        fock = state.fock_matrix.mean(axis=0)
        fock_mo = coefficients.T@fock@coefficients
        orbitals = solve_hermitian(.5*(fock_mo+fock_mo.T),
            config=EigenSolverConfig(method="dense", nroots=self.nmo, atol=1e-10),
            response=EigenResponseConfig(target="eigenpairs"))
        coefficients = coefficients@orbitals.vectors
        density = state.density_matrix.sum(axis=0)
        metric = inputs["overlap"]
        occupation = coefficients.T@metric@density@metric@coefficients
        expected = jnp.diag(jnp.concatenate([jnp.full(self.nocc, 2.), jnp.zeros(self.nmo-self.nocc)]))
        reference_valid = state.stationary & jnp.all(orbitals.response_valid)
        reference_valid &= jnp.max(jnp.abs(occupation-expected)) < 1e-8
        owners = (self.reference._memory_inputs(), coordinates, state, inputs, orbitals,
                  fock, fock_mo, coefficients, density)
        pairs = len(self.pairs[0])
        # Pair extraction, eigensolver/root matrices, and unpacked factors are
        # simultaneous known owners; do this check before native pair assembly.
        workspace_bytes(self.max_memory, resident_bytes=tree_storage_bytes(*owners)
            + 8*(5*pairs*pairs + 2*pairs*coefficients.shape[0]**2), device=array_device(coordinates))
        eri = inputs["eri"]
        direct = inputs.get("direct_basis")
        rows, columns = self.pairs
        if direct is not None:
            source = dict(direct_basis=direct)
            pair = direct.response_eri_pair_matrix()
        elif eri.ndim == 2:
            source = dict(eri_pair_matrix=eri)
            pair = eri
        else:
            source = dict(eri=eri)
            pair = eri[rows, columns][:, rows, columns]
        factors = eri_pair_matrix_to_df_factors_traceable(
            pair, nao=coefficients.shape[0], tol=self.factor_tolerance)
        common = dict(mo_energy=orbitals.values, mo_coeff=coefficients, nocc=self.nocc,
                      df_factors=factors, hcore_matrix=inputs["hcore"], nw=self.nw)
        owners += (pair, factors)
        if self.method in {"g0w0", "evgw0", "evgw"}:
            extra = tree_storage_bytes(*owners, exclude=(source, density))
            coulomb = _coulomb(source, density, self.max_memory, reserved_bytes=extra)
            common.update(fock_matrix=fock, density_matrix=density,
                          v_mf_matrix=fock-inputs["hcore"]-coulomb,
                          eta=self.eta, **self.windows)
        elif self.method == "scgw":
            common["nuclear_repulsion"] = inputs["nuclear_repulsion"]
        if self.method != "scgw":
            common.update(max_memory=self.max_memory,
                          reserved_bytes=tree_storage_bytes(*owners, exclude=(common,)))
        return common, reference_valid

    def _solve(self, coordinates):
        common, reference_valid = self._prepare(coordinates)
        if self.method == "g0w0":
            result = gw.g0w0_cd_restricted(**common, resolvent_expansion=True,
                                          qp_solver=self.qp_solver, qp_initial=self.qp_initial,
                                          qp_tol=self.qp_tol)
        elif self.method in {"evgw0", "evgw"}:
            result = gw.evgw_cd_restricted(**common, update_w=self.method == "evgw",
                max_iter=self.max_cycle, tol=self.conv_tol, damping=self.damp,
                diis_space=self.diis_space, initial_guess=self.qp_initial,
                differentiation=self.outer_response)
        elif self.method == "qsgw":
            result = gw.qsgw_cd_restricted(**common, eta=self.eta,
                max_iter=self.max_cycle, tol=self.conv_tol, tol_density=self.conv_tol, damping=self.damp,
                differentiation=self.outer_response)
            result = replace(result, qp_computed_mask=jnp.ones(self.nmo, dtype=bool),
                converged_mask=jnp.full(self.nmo, result.converged, dtype=bool),
                g_orbitals=tuple(range(self.nmo)), screening_occupied=tuple(range(self.nocc)),
                screening_virtual=tuple(range(self.nocc, self.nmo)))
        else:
            result = gw.scgw_matsubara_restricted(**common, beta=self.beta, mixing=self.mixing,
                max_iter=self.max_cycle, tol=self.conv_tol, particle_tol=1e-12,
                differentiation=self.outer_response)
            result = replace(result, converged=result.converged & reference_valid,
                total_energy=jnp.where(reference_valid, result.total_energy, jnp.nan),
                density_matrix=jnp.where(reference_valid, result.density_matrix, jnp.nan),
                chemical_potential=jnp.where(reference_valid, result.chemical_potential, jnp.nan))
            return result, common["df_factors"]
        # Preserve the GW driver's per-level diagnostics on numerical failure.
        # The result's masks certify QP convergence; its implicit rule rejects
        # derivatives of failed roots. An invalid KS frame invalidates all levels.
        result = replace(result, mo_energy=jnp.where(reference_valid, result.mo_energy, jnp.nan),
                          converged=result.converged & reference_valid,
                          converged_mask=result.converged_mask & reference_valid)
        return result, common["df_factors"]

    def pole_seeds(self, coordinates=None, *, max_dense=2048):
        """Host-only eta=0 largest-weight seeds for the full G0W0 orbital space.

        Native KS and the existing direct-RPA pole expansion provide the
        model; the shared Hermitian solver selects isolated spectral groups.
        Use the resulting seeds in the finite-eta CD solve and check its actual
        roots/residues. The seed selector itself is not differentiated and does
        not certify finite-eta global maximum weight or geometric continuity.
        Partial G/W windows are deliberately excluded from this small adapter.
        """
        from gradscf.gw.branches import pole_seeds
        from gradscf.gw.poles import rpa_pole_expansion

        if self.method == "scgw":
            raise NotImplementedError("Matsubara scGW supplies no QP poles for starting-branch seeds.")
        coordinates = self.coordinates if coordinates is None else coordinates
        if any(isinstance(x, jax.core.Tracer) for x in jax.tree.leaves(coordinates)):
            raise TypeError("Molecular pole seeds are host-only; build them outside JIT and AD")
        if self.method != "g0w0":
            raise NotImplementedError("This seed adapter currently constructs G0W0 starting branches")
        allowed = dict(orbs=tuple(range(self.nmo)), g_orbitals=tuple(range(self.nmo)),
                       screening_occupied=tuple(range(self.nocc)),
                       screening_virtual=tuple(range(self.nocc, self.nmo)))
        if any(value is not None and tuple(value) != allowed[name] for name, value in self.windows.items()):
            raise NotImplementedError("Molecular pole seeds currently require the full G/W/QP windows")
        dimension = 1+self.nocc*(self.nmo-self.nocc)*self.nmo
        if dimension > max_dense:
            raise ValueError(f"Pole seed dimension {dimension} exceeds max_dense={max_dense}")
        inputs, valid = self._prepare(jnp.asarray(coordinates))
        if not bool(valid):
            raise RuntimeError("Converge an isolated canonical Aufbau KS reference before making seeds")
        c, factors = inputs["mo_coeff"], inputs["df_factors"]
        mo_factors = _mo_factors(factors, c)
        # Reuse the same exchange subtraction as the production CD driver.
        v_mf = c.T@inputs["v_mf_matrix"]@c
        delta_v = jnp.diag(_exchange_mo(mo_factors, self.nocc)-v_mf)
        poles, couplings = rpa_pole_expansion(inputs["mo_energy"],
            mo_factors[:, :self.nocc, self.nocc:], mo_factors)
        return pole_seeds(inputs["mo_energy"]+delta_v, inputs["mo_energy"], self.nocc,
                          poles, couplings, max_dense=max_dense)

    def __call__(self, coordinates):
        """Return GWResult, or SCGWResult for finite-T matrix scGW."""
        return self._solve(coordinates)[0]

    def optical(self, coordinates, *, config=None, occupied=None, virtual=None):
        """Return (GW result, BSE result, moving MO position, excitation space).

        This is a complete dOmega/dR or optical-property geometry map, not a
        definition of a variational GW+BSE total excited-state energy.
        Full BSE and TDA have their separate shared response contracts.
        """
        if self.method == "scgw":
            raise NotImplementedError("Matsubara scGW has no QP poles for a static optical/BSE reference.")
        result, ao_factors = self._solve(coordinates)
        factors = _mo_factors(ao_factors, result.mo_coeff)
        space = bse.make_bse_space(self.nmo, self.nocc, occupied=occupied, virtual=virtual)
        screening = bse.make_bse_space(self.nmo, self.nocc,
            occupied=result.screening_occupied, virtual=result.screening_virtual)
        response = bse.run_bse(result.mo_energy, result.screening_energy, factors, space,
            screening_space=screening, qp_computed_mask=result.qp_computed_mask,
            qp_converged_mask=result.converged_mask, config=config)
        moving = replace(self.reference.parameters, centers=coordinates[self.reference.owners],
                         nuclear_coords=coordinates)
        position = self.reference.plan.evaluate("dipole", moving, origin=jnp.zeros(3))
        dipole = jnp.einsum("xpq,pi,qj->xij", position, result.mo_coeff, result.mo_coeff)
        return result, response, dipole, space
