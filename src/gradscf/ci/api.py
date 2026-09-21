"""PySCF-style facades backed by GradSCF's JAX CI kernels."""
import numpy as np

from .integrals import reference_from_source
from .space import make_ci_space
from .solver import solve_ci, solve_cis
from .solver import restricted_fock
from .corrections import cis_d_correction
from .types import CIConfig


class CI:
    """Single-reference CI through max_excitation, in the M_s=0 sector.

    kernel() returns (e_corr, ci). One root gives a scalar and a vector;
    multiple roots give an energy array and coefficient columns. result always
    retains a root axis. This facade performs eager input validation; use the
    functional solve_ci interface under jit/grad.
    """
    def __init__(self, mf, *, max_excitation=2, frozen=None, nroots=1,
                 solver="davidson", conv_tol=1e-9, max_cycle=100, max_space=None,
                 gradient_mode="eigenvalue_only", adjoint_tol=1e-10,
                 adjoint_max_cycle=100, max_determinants=5000):
        self.mf = mf
        self.max_excitation = max_excitation
        self.frozen = frozen
        self.nroots = nroots
        self.solver = solver
        self.conv_tol = conv_tol
        self.max_cycle = max_cycle
        self.max_space = max_space
        self.gradient_mode = gradient_mode
        self.adjoint_tol = adjoint_tol
        self.adjoint_max_cycle = adjoint_max_cycle
        self.max_determinants = max_determinants
        self.result = self.reference = self.space = None
        self.e_tot = self.e_corr = self.ci = self.converged = None

    def _config(self):
        return CIConfig(nroots=self.nroots, solver=self.solver, conv_tol=self.conv_tol,
                        max_cycle=self.max_cycle, max_space=self.max_space,
                        gradient_mode=self.gradient_mode, adjoint_tol=self.adjoint_tol,
                        adjoint_max_cycle=self.adjoint_max_cycle)

    def kernel(self):
        self.reference = ref = reference_from_source(self.mf)
        self.space = make_ci_space(ref.h1.shape[0], ref.nocc, max_excitation=self.max_excitation,
                                   frozen=self.frozen, max_determinants=self.max_determinants)
        self.result = solve_ci(ref.h1, ref.eri, self.space,
                               nuclear_repulsion=ref.nuclear_repulsion, config=self._config())
        self.e_tot = self.result.total_energies
        self.e_corr = self.result.correlation_energies
        self.ci = self.result.coefficients
        self.converged = np.asarray(self.result.converged)
        if self.nroots == 1:
            self.e_tot, self.e_corr, self.ci = self.e_tot[0], self.e_corr[0], self.ci[:, 0]
            self.converged = bool(self.converged[0])
        return self.e_corr, self.ci

    def run(self):
        self.kernel()
        return self


class CISD(CI):
    def __init__(self, mf, **kwargs):
        super().__init__(mf, max_excitation=2, **kwargs)


class CISDT(CI):
    def __init__(self, mf, **kwargs):
        super().__init__(mf, max_excitation=3, **kwargs)


class CISDTQ(CI):
    def __init__(self, mf, **kwargs):
        super().__init__(mf, max_excitation=4, **kwargs)


class CIS(CI):
    """Spin-adapted HF singles. e is an array of excitation energies in Hartree."""
    def __init__(self, mf, *, singlet=True, **kwargs):
        super().__init__(mf, max_excitation=1, **kwargs)
        self.singlet = singlet
        self.e = self.amplitudes = None

    def kernel(self):
        self.reference = ref = reference_from_source(self.mf)
        fock = np.asarray(restricted_fock(ref.h1, ref.eri, ref.nocc))
        if np.max(np.abs(fock[:ref.nocc, ref.nocc:]), initial=0.) > 1e-7:
            raise ValueError("CIS requires a stationary HF reference (occupied-virtual Fock block must vanish)")
        self.result = solve_cis(ref.h1, ref.eri, nocc=ref.nocc, singlet=self.singlet,
                                frozen=self.frozen, config=self._config())
        self.e, self.amplitudes = self.result.excitation_energies, self.result.amplitudes
        self.converged = np.asarray(self.result.converged)
        return self.e, self.amplitudes


class CIS_D(CIS):
    """Canonical RHF singlet CIS(D), with complete first-order CIS-vector response."""
    def __init__(self, mf, *, singlet=True, denominator_tol=1e-10, **kwargs):
        if not singlet:
            raise NotImplementedError("The CIS(D) facade currently supports singlet roots only")
        if kwargs.get("gradient_mode", "implicit_eigenvector") != "implicit_eigenvector":
            raise ValueError("CIS(D) requires gradient_mode='implicit_eigenvector'")
        kwargs["gradient_mode"] = "implicit_eigenvector"
        super().__init__(mf, singlet=True, **kwargs)
        self.denominator_tol = denominator_tol
        self.e_cis = self.correction = self.cis_result = None

    def kernel(self):
        super().kernel()
        self.cis_result = self.result
        ref = self.reference
        if ref.mo_energy is None:
            raise ValueError("CIS(D) requires canonical HF mo_energy")
        fock = np.asarray(restricted_fock(ref.h1, ref.eri, ref.nocc))
        eps = np.asarray(ref.mo_energy)
        if eps.shape != (ref.h1.shape[0],) or not np.allclose(
                fock, np.diag(eps), atol=1e-7, rtol=0):
            raise ValueError("CIS(D) requires canonical HF orbitals and consistent mo_energy")
        self.result = cis_d_correction(ref.eri, ref.mo_energy, self.cis_result,
                                       nocc=ref.nocc, frozen=self.frozen,
                                       denominator_tol=self.denominator_tol)
        self.e_cis, self.correction = self.result.cis_energies, self.result.corrections
        self.e = self.result.excitation_energies
        self.converged = np.asarray(self.result.valid)
        return self.e, self.amplitudes
