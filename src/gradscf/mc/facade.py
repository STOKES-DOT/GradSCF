"""Molecular CASCI and CASSCF front ends with common spatial orbitals."""

from dataclasses import dataclass
from numbers import Integral
from threading import RLock

import jax
import jax.numpy as jnp
import numpy as np

from gradscf._cache import array_cache
from gradscf._memory import (
    resolve_memory_mb, tree_storage_bytes, validate_memory_mb, workspace_bytes,
)
from gradscf.ao2mo.active import orbital_selection, read_ao_representation
from gradscf.solvers import EigenSolverConfig

from .casci import solve_casci
from .csf import make_csf_space
from .integrals import transform_active


_CSF_SPACE_LOCK = RLock()


@array_cache(maxsize=16, max_bytes=64_000_000, payload_bytes=lambda space, *_: space.storage_bytes)
def _cached_csf_space(ncas, nelecas, spin2, max_memory):
    return make_csf_space(ncas, nelecas, spin2, max_memory=max_memory)


def _facade_csf_space(ncas, nelecas, spin2, *, max_memory):
    # Validate types before normalizing the key. Booleans and integral floats
    # compare equal to integers but must still receive the builder's validation.
    if (not isinstance(nelecas, (list, tuple)) or len(nelecas) != 2
            or any(not isinstance(value, Integral) or isinstance(value, bool)
                   for value in (ncas, *nelecas, spin2))):
        return make_csf_space(ncas, nelecas, spin2, max_memory=max_memory)
    budget = validate_memory_mb(max_memory)
    # Serialize misses so concurrent facades do not build the same large graph.
    with _CSF_SPACE_LOCK:
        space = _cached_csf_space(int(ncas), tuple(map(int, nelecas)), int(spin2), budget)
        workspace_bytes(budget, resident_bytes=space.storage_bytes, device="cpu")
    return space


@dataclass(frozen=True)
class MCResult:
    """Molecular energy, orbital frame and state in its fixed-spin CSF basis."""

    total_energy: jax.Array
    active_energy: jax.Array
    mo_coeff: jax.Array
    ci_coeff: jax.Array
    space: object
    core: tuple[int, ...]
    active: tuple[int, ...]
    external: tuple[int, ...]
    ci_residual: jax.Array
    orbital_gradient: jax.Array | None
    converged: bool
    gap: jax.Array
    iterations: jax.Array | None
    response_valid: bool

    def make_rdm12(self):
        """Return the spin-summed active one- and ordered two-particle RDM."""
        self._check_state()
        return self.space.rdm12(self.ci_coeff)

    def make_rdm1s(self):
        """Return active alpha and beta one-particle RDMs in the input M_S."""
        self._check_state()
        return self.space.rdm1s(self.ci_coeff)

    def _check_state(self):
        if not self.converged:
            raise RuntimeError("Converge the multireference state before requesting density matrices")


class CASCI:
    """Fixed-orbital active CI on a converged GradSCF RHF/ROHF source.

    The activity order is significant. Supply explicit input arrays to
    :func:`solve_casci` when transforming this calculation with JAX.
    Facades share a process-local cache of 16 static CSF topologies and up to
    64 MB of their host arrays. The key includes the topology and memory budget.
    """

    def __init__(self, mf, ncas, nelecas, *, spin2, core=None, active=None,
                 solver="davidson", ci_tol=1e-10, max_memory=None):
        if not isinstance(ncas, Integral) or isinstance(ncas, bool) or ncas < 1:
            raise ValueError("ncas must be a positive integer")
        try:
            nelecas = tuple(nelecas)
        except TypeError as error:
            raise ValueError("nelecas must contain active alpha and beta electron counts") from error
        if (len(nelecas) != 2
                or any(not isinstance(n, Integral) or isinstance(n, bool)
                       or not 0 <= n <= ncas for n in nelecas)):
            raise ValueError("nelecas must count active alpha and beta electrons within ncas")
        self.mf = mf
        self.ncas = int(ncas)
        self.nelecas = tuple(map(int, nelecas))
        self.spin2 = spin2
        self.core = core
        self.active = active
        self.max_memory = resolve_memory_mb(max_memory, mf)
        self.config = EigenSolverConfig(method=solver, atol=ci_tol)
        self.result = None

    @property
    def e_tot(self):
        return None if self.result is None else self.result.total_energy

    @property
    def converged(self):
        return False if self.result is None else self.result.converged

    def run(self, *, ci0=None):
        """Solve in the supplied MO frame and store an independent result."""
        from gradscf.scf.facade import RKS
        from gradscf.scf.roks import ROKS

        self.result = None
        mf = self.mf
        if not isinstance(mf, (RKS, ROKS)) or str(mf.xc).strip().lower() != "hf":
            raise TypeError("CASCI and CASSCF require a common-orbital RHF or ROHF source")
        mf._check_reference_source()
        inputs = mf._scf_inputs
        if inputs is None:
            raise RuntimeError("Converged HF source does not contain solved AO inputs")
        c = jnp.asarray(mf.mo_coeff)
        s = jnp.asarray(inputs.overlap)
        if c.ndim != 2 or jnp.iscomplexobj(c):
            raise ValueError("HF source must have real common spatial orbitals")
        if not np.allclose(np.asarray(c.T @ s @ c), np.eye(c.shape[1]), atol=1e-8, rtol=0):
            raise ValueError("HF orbitals must be S-orthonormal")
        ncore, unpaired = divmod(int(mf.mol.nelectron) - sum(self.nelecas), 2)
        if ncore < 0 or unpaired:
            raise ValueError("nelecas must leave a nonnegative even number of nonactive electrons")
        core = ncore if self.core is None else self.core
        core, candidates = orbital_selection(c.shape[1], core, self.active)
        active = candidates[:self.ncas] if self.active is None else candidates
        if len(core) != ncore:
            raise ValueError("core orbital count must match the nonactive electron count")
        if len(active) != self.ncas:
            raise ValueError("active orbital count must equal ncas")
        external = tuple(p for p in range(c.shape[1]) if p not in core and p not in active)
        space = _facade_csf_space(self.ncas, self.nelecas, self.spin2, max_memory=self.max_memory)
        ao = read_ao_representation(inputs, c, max_memory=self.max_memory, owner=mf)
        state, c, gradient = self._solve(
            s, inputs.hcore, c, inputs.nuclear_repulsion, space, core, active, ao, ci0,
        )
        self.result = MCResult(
            state.total_energy, state.active_energy, c, state.ci_coeff, space,
            core, active, external, state.ci_residual, gradient,
            bool(state.converged), state.gap, state.iterations, bool(state.response_valid),
        )
        return self

    def _solve(self, s, h, c, enuc, space, core, active, ao, ci0):
        hcas, gcas, core_energy = transform_active(
            h, c, core=core, active=active, ecore=enuc, max_memory=self.max_memory, reserved_bytes=space.storage_bytes, **ao,
        )
        state = solve_casci(hcas, gcas, space, ecore=core_energy, config=self.config,
                            ci0=ci0, max_memory=self.max_memory,
                            reserved_bytes=tree_storage_bytes(h, c, enuc, ao))
        return state, c, None

    def _state(self):
        if self.result is None:
            raise RuntimeError("Run the multireference solver before requesting state properties")
        return self.result

    def make_rdm12(self):
        return self._state().make_rdm12()

    def make_rdm1s(self):
        return self._state().make_rdm1s()

    def spin_square(self):
        state = self._state()
        state._check_state()
        spin2 = state.space.graph.spin2
        return spin2 * (spin2 + 2) / 4., spin2 + 1


class CASSCF(CASCI):
    """Joint orbital and fixed-spin CI optimization in a specified active space."""

    def __init__(self, mf, ncas, nelecas, *, max_cycle=100, conv_tol=1e-10,
                 conv_tol_grad=1e-7, **kwargs):
        super().__init__(mf, ncas, nelecas, **kwargs)
        self.max_cycle = max_cycle
        self.conv_tol = conv_tol
        self.conv_tol_grad = conv_tol_grad

    def _solve(self, s, h, c, enuc, space, core, active, ao, ci0):
        from .orbitals import solve_casscf

        state = solve_casscf(
            s, h, c, enuc, space, core=core, active=active, config=self.config,
            ci0=ci0, max_cycle=self.max_cycle, conv_tol=self.conv_tol,
            conv_tol_grad=self.conv_tol_grad, max_memory=self.max_memory, **ao,
        )
        return state, state.mo_coeff, state.orbital_gradient
