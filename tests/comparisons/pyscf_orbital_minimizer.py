"""Test-only fixed-occupation reference using PySCF Fock and SciPy Frechet AD.

This module never calls GradSCF energy/gradient code or PySCF Newton kernels.
Stationarity is a local first-order test, not a ground-state/stability claim.
All gradient norms are the true final-chart angle-gradient norm divided by 2.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator, minres


@dataclass
class ReferenceOrbitalResult:
    total_energy: float
    density_matrix: np.ndarray
    mo_coeff: np.ndarray
    mo_occ: np.ndarray
    stationary: bool
    gradient_norm: float
    optimizer_success: bool
    optimizer_message: str
    iterations: int
    evaluations: int
    polish_steps: int
    polish_history: list[dict]


class OrbitalObjective:
    """Exact energy and first derivative in one exponential orbital chart."""

    def __init__(self, mf, mo_coeff, mo_occ, *, method):
        self.mf = mf
        self.method = method.replace('_ncol', '').upper()
        self.coeff = np.asarray(mo_coeff, dtype=complex if np.iscomplexobj(mo_coeff) else float).copy()
        self.occ = np.asarray(mo_occ, dtype=float).copy()
        self.unrestricted = self.method in {'UHF', 'UKS'}
        self.restricted_open = self.method in {'ROHF', 'ROKS'}
        self.generalized = self.method in {'GHF', 'GKS'}
        if not (self.unrestricted or self.restricted_open or self.generalized):
            raise ValueError(f'Unsupported reference method: {method}')
        self.hcore = np.asarray(mf.get_hcore())
        self.complex_angles = self.generalized or np.iscomplexobj(self.coeff)
        if self.complex_angles:
            self.coeff = self.coeff.astype(complex)
        nmo = self.coeff.shape[-1]
        expected_occ_shape = (nmo,) if self.generalized else (2, nmo)
        if self.occ.shape != expected_occ_shape:
            raise ValueError(f'{method} requires occupations with shape {expected_occ_shape}')
        expected_coeff_ndim = 3 if self.unrestricted else 2
        if self.coeff.ndim != expected_coeff_ndim:
            raise ValueError('Incompatible coefficient dimensions')
        overlap = np.asarray(mf.get_ovlp())
        metric = self.coeff.conj().swapaxes(-1, -2) @ overlap @ self.coeff
        if not np.allclose(metric, np.eye(nmo), atol=1e-8, rtol=0):
            raise ValueError('Initial orbitals must be overlap-orthonormal')
        self.pairs = []
        occupation_sets = self.occ if self.unrestricted else [self.occ]
        for occupations in occupation_sets:
            different = occupations[..., :, None] != occupations[..., None, :]
            if different.ndim == 3:
                different = np.any(different, axis=0)
            self.pairs.append(np.where(np.triu(different, k=1)))
        self.sizes = [len(i)*(2 if self.complex_angles else 1) for i, _ in self.pairs]
        self.size = sum(self.sizes)
        self.evaluations = 0

    def generators(self, x):
        matrices = []
        offset = 0
        for (i, j), size in zip(self.pairs, self.sizes):
            count = len(i)
            values = np.asarray(x[offset:offset + count])
            if self.complex_angles:
                values = values + 1j*np.asarray(x[offset + count:offset + size])
            k = np.zeros((self.coeff.shape[-1],)*2, dtype=self.coeff.dtype)
            k[i, j] = values
            k[j, i] = -values.conj()
            matrices.append(k)
            offset += size
        return matrices

    def coefficients(self, x):
        bases = self.coeff if self.unrestricted else [self.coeff]
        transformed = [c @ expm(k) for c, k in zip(bases, self.generators(x))]
        return np.stack(transformed) if self.unrestricted else transformed[0]

    def density(self, coefficients):
        if self.restricted_open:
            return np.stack([(coefficients * o) @ coefficients.conj().T for o in self.occ])
        return (coefficients*self.occ[..., None, :]) @ coefficients.conj().swapaxes(-1, -2)

    def value_and_grad(self, x):
        self.evaluations += 1
        ks = self.generators(x)
        bases = self.coeff if self.unrestricted else [self.coeff]
        cs = [c @ expm(k) for c, k in zip(bases, ks)]
        coefficients = np.stack(cs) if self.unrestricted else cs[0]
        dm = self.density(coefficients)
        potential = self.mf.get_veff(dm=dm)
        energy = float(self.mf.energy_tot(dm=dm, h1e=self.hcore, vhf=potential))
        # For RO this is the physical pair of spin Focks, not Roothaan's
        # effective Fock; the latter is not an unrestricted density derivative.
        focks = self.hcore + np.asarray(potential)
        if self.restricted_open:
            g_rotations = [sum(2 * (self.coeff.conj().T @ f @ coefficients) * o
                               for f, o in zip(focks, self.occ))]
        elif self.unrestricted:
            g_rotations = [2 * (c0.conj().T @ f @ c) * o
                           for c0, f, c, o in zip(bases, focks, cs, self.occ)]
        else:
            g_rotations = [2 * (self.coeff.conj().T @ focks @ coefficients) * self.occ]
        pieces = []
        for k, g_rotation, (i, j) in zip(ks, g_rotations, self.pairs):
            # Frobenius-inner-product adjoint of D exp(K).
            gk = expm_frechet(k.conj().T, g_rotation, compute_expm=False)
            pieces.append((gk[i, j] - gk[j, i]).real)
            if self.complex_angles:
                pieces.append((gk[i, j] + gk[j, i]).imag)
        gradient = np.concatenate(pieces) if pieces else np.zeros(0)
        return energy, gradient


def minimize_pyscf_orbitals(mf, mo_coeff, mo_occ, *, method=None,
                            max_iterations=500, gradient_tolerance=1e-7,
                            polish=True, _allow_rebase=True):
    """Optimize from the supplied initial orbitals and unchanged occupations.

    RO methods require occupations of shape (2,nmo). Generalized methods use
    full complex spinor coefficients and occupation shape (2*nao,). Polishing
    uses bounded finite differences of PySCF Fock gradients and MINRES, only
    near an apparent stationary point; accepted steps must reduce the measured
    gradient and may raise the energy by at most a documented roundoff bound.
    """
    if method is None:
        method = type(mf).__name__
    if gradient_tolerance <= 0 or max_iterations < 1:
        raise ValueError('Positive gradient tolerance and iteration count required')
    objective = OrbitalObjective(mf, mo_coeff, mo_occ, method=method)
    initial = np.zeros(objective.size)
    if objective.size:
        opt = minimize(objective.value_and_grad, initial, jac=True, method='L-BFGS-B',
                       options={'maxiter': max_iterations, 'gtol': gradient_tolerance*.1,
                                'ftol': 1e-15, 'maxls': 50, 'maxcor': 30})
        coefficients = objective.coefficients(opt.x)
        optimizer_success, message, iterations = bool(opt.success), str(opt.message), int(opt.nit)
    else:
        coefficients = objective.coeff
        optimizer_success, message, iterations = True, 'No occupation-changing rotations', 0
    evaluations = objective.evaluations
    current = OrbitalObjective(mf, coefficients, mo_occ, method=method)
    energy, gradient = current.value_and_grad(np.zeros(current.size))
    history = []
    # Energy differences below float64 resolution can stop a line search before
    # the physically meaningful gradient passes. Do not use this as a global
    # optimizer, nor accept failed true-gradient tests as success.
    if polish and gradient_tolerance < np.linalg.norm(gradient)/2 < 1e-3:
        for _ in range(5):
            old_norm = float(np.linalg.norm(gradient)/2)
            if old_norm <= gradient_tolerance:
                break

            def hvp(vector):
                magnitude = np.linalg.norm(vector)
                if magnitude == 0:
                    return np.zeros_like(vector)
                step = 1e-4/magnitude
                gp = current.value_and_grad(step*vector)[1]
                gm = current.value_and_grad(-step*vector)[1]
                return (gp-gm)/(2*step)

            direction = -gradient
            status = -1
            linear_residual = float('inf')
            damping = 0.
            # Near-null or weak negative modes make an undamped Newton step
            # arbitrarily large. First relax stiff components with a modest
            # positive shift. A positive shift gives a bounded descent
            # solve; its shifted linear residual is checked explicitly.
            for damping in (1e-3, 0., 1e-7, 1e-5, .1, 1.):
                def shifted_hvp(vector):
                    return hvp(vector) + damping*vector
                hessian = LinearOperator((current.size, current.size), matvec=shifted_hvp, dtype=float)
                proposal, status = minres(hessian, -gradient, rtol=1e-12,
                                          maxiter=min(200, max(10, 2*current.size)))
                linear_residual = float(np.linalg.norm(shifted_hvp(proposal) + gradient))
                if (np.all(np.isfinite(proposal)) and gradient @ proposal < 0
                        and linear_residual <= .1*np.linalg.norm(gradient)
                        and np.linalg.norm(proposal) <= .05):
                    direction = proposal
                    break
            norm = np.linalg.norm(direction)
            if norm > .05:
                direction *= .05/norm
            allowance = 128*np.finfo(float).eps*max(1., abs(energy))
            accepted = False
            for factor in (1., .5, .25, .125, .0625):
                trial_coeff = current.coefficients(factor*direction)
                trial = OrbitalObjective(mf, trial_coeff, mo_occ, method=method)
                trial_energy, trial_gradient = trial.value_and_grad(np.zeros(trial.size))
                trial_norm = float(np.linalg.norm(trial_gradient)/2)
                evaluations += trial.evaluations
                if trial_norm < old_norm and trial_energy <= energy+allowance:
                    history.append(dict(gradient_before=old_norm, gradient_after=trial_norm,
                                        energy_change=trial_energy-energy, energy_allowance=allowance,
                                        step_norm=float(np.linalg.norm(factor*direction)),
                                        minres_status=int(status), damping=float(damping),
                                        linear_relative_residual=linear_residual/np.linalg.norm(gradient)))
                    evaluations += current.evaluations
                    current = trial
                    current.evaluations = 0
                    energy, gradient = trial_energy, trial_gradient
                    accepted = True
                    break
            if not accepted:
                break
    evaluations += current.evaluations
    gradient_norm = float(np.linalg.norm(gradient)/2)
    stationary = bool(np.isfinite(energy) and np.isfinite(gradient_norm)
                      and gradient_norm <= gradient_tolerance)
    result = ReferenceOrbitalResult(
        total_energy=energy, density_matrix=current.density(current.coeff), mo_coeff=current.coeff,
        mo_occ=objective.occ.copy(), stationary=stationary, gradient_norm=gradient_norm,
        optimizer_success=optimizer_success, optimizer_message=message, iterations=iterations,
        evaluations=evaluations, polish_steps=len(history), polish_history=history,
    )
    # A large original exponential chart can obscure weak physical directions.
    # One new chart uses the remaining optimizer budget; it remains independent
    # of native results, and no global-stability conclusion follows.
    remaining = max_iterations - iterations
    if (polish and _allow_rebase and not stationary and remaining > 0
            and np.isfinite(gradient_norm) and gradient_norm < 1e-3):
        candidate = minimize_pyscf_orbitals(
            mf, result.mo_coeff, mo_occ, method=method,
            max_iterations=remaining, gradient_tolerance=gradient_tolerance,
            polish=True, _allow_rebase=False,
        )
        allowance = 128*np.finfo(float).eps*max(1., abs(result.total_energy))
        if (candidate.gradient_norm < result.gradient_norm
                and candidate.total_energy <= result.total_energy + allowance):
            candidate.iterations += result.iterations
            candidate.evaluations += result.evaluations
            candidate.polish_history = result.polish_history + candidate.polish_history
            candidate.polish_steps += result.polish_steps
            candidate.optimizer_success = result.optimizer_success and candidate.optimizer_success
            candidate.optimizer_message = result.optimizer_message + '; rebase: ' + candidate.optimizer_message
            result = candidate
        else:
            result.evaluations += candidate.evaluations
            result.iterations += candidate.iterations
            result.optimizer_message += '; rebased candidate did not improve true residual and energy'
    return result
