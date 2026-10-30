"""Fixed-reference parameterized hybrid XC energy adapter tests."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import dft
from gradscf.dft import iks
from gradscf.tools.features import MoleculeGridView, MoleculeLikeState


def one_ao_state():
    return MoleculeLikeState(
        ao=jnp.ones((1, 1)),
        ao_deriv1=jnp.zeros((4, 1, 1)),
        grid=MoleculeGridView(weights=jnp.ones(1)),
        rdm1=jnp.zeros((2, 1, 1)),
        mo_coeff=jnp.ones((2, 1, 1)),
        mo_occ=jnp.array([[1.0], [1.0]]),
        mo_energy=jnp.zeros((2, 1)),
        rep_tensor=jnp.full((1, 1, 1, 1), 0.8),
        h1e=jnp.zeros((1, 1)),
        overlap_matrix=jnp.ones((1, 1)),
    )


def test_fixed_reference_hybrid_xc_adds_exact_exchange_to_semilocal_energy():
    state = one_ao_state()
    semilocal = dft.custom_xc(
        input_fn=lambda features: features.density_matrix,
        energy_fn=lambda params, density: params * jnp.sum(density),
    )
    energy = iks.hybrid_xc_energy_function(
        state, semilocal, layout="rks", exact_exchange_fraction=0.25
    )

    value = energy(jnp.asarray(0.3), jnp.ones((1, 1)))

    # E_semilocal = 0.3 and E_HF = -0.25 * 0.2 = -0.05 for this 1-AO case.
    np.testing.assert_allclose(value, 0.25, atol=2e-13)


def test_fixed_reference_hybrid_xc_keeps_parameter_ad_live():
    state = one_ao_state()
    semilocal = dft.custom_xc(
        input_fn=lambda features: features.density_matrix,
        energy_fn=lambda params, density: params * jnp.sum(density),
    )
    energy = iks.hybrid_xc_energy_function(state, semilocal, layout="rks")

    derivative = jax.grad(lambda params: energy(params, jnp.ones((1, 1))))(0.3)

    np.testing.assert_allclose(derivative, 1.0, atol=2e-13)


def test_fixed_reference_hybrid_xc_rejects_spinor_layout():
    state = one_ao_state()
    semilocal = dft.custom_xc(
        input_fn=lambda features: features.density_matrix,
        energy_fn=lambda params, density: params * jnp.sum(density),
    )
    with pytest.raises(ValueError, match="RKS|UKS|spinor"):
        iks.hybrid_xc_energy_function(state, semilocal, layout="gks")
