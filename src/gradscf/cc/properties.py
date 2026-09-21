"""CC one-particle properties from the stationary amplitude Lagrangian."""

import jax
import jax.numpy as jnp
from .amplitudes import AmplitudeSpace
from .integrals import prepare_integrals
from .rccsd import residual, correlation_energy
from .lambda_equations import solve_lambda
from .types import CCConfig


def _density_from_lambda(h1, eri, result, left, *, nocc, frozen, config):
    no, nv = result.t1.shape
    space = AmplitudeSpace(no, nv, config.method)

    def lagrangian(raw_h):
        # Derivative for a real Hermitian perturbation; PySCF's public RDM1 is
        # also symmetrized. This is an orbital-unrelaxed, spin-summed MO density.
        h = 0.5 * (raw_h + raw_h.T)
        ints = prepare_integrals(h, eri, nocc=nocc, frozen=frozen)
        r = space.pack(*residual(result.t1, result.t2, ints, model=config.method))
        e = ints.reference_energy + correlation_energy(
            result.t1, result.t2, ints, model=config.method
        )
        return e + jnp.vdot(left.adjoint, r).real

    # T and lambda are held fixed for this *partial* derivative. Do not apply
    # stop_gradient: an outer derivative of the density must include both states'
    # implicit response, which is verified against reconverged finite differences.
    density = jax.grad(lagrangian)(jnp.asarray(h1))
    return jnp.where(result.converged & left.converged, density, jnp.nan)


def make_rdm1(h1, eri, result, *, nocc, frozen=None, config=None):
    """Real symmetric, spin-summed MO 1-RDM, including frozen occupied electrons.

    This includes T/Lambda response but not optimization of the reference
    orbitals. It is a CC-model density, not a CCSD(T)-corrected density.
    """
    cfg = CCConfig() if config is None else config
    left = solve_lambda(h1, eri, result, nocc=nocc, frozen=frozen, config=cfg)
    return _density_from_lambda(
        h1, eri, result, left, nocc=nocc, frozen=frozen, config=cfg
    )
