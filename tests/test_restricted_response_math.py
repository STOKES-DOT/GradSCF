"""Restricted TD response shares XC without changing independent X/Y math."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.tddft import casida, response
from gradscf.tddft._semilocal_response import SemilocalResponseFunctional


class _Reference(SimpleNamespace):
    """Semilocal tensor caching requires a weak-referenceable molecule view."""


def _reference():
    ao = jnp.asarray([[.8, .2, -.1, .3], [.1, .7, .4, -.2],
                      [.4, -.3, .6, .5]])
    occupied = jnp.asarray([1., 1., 0., 0.])
    return _Reference(
        mo_coeff=jnp.eye(4), mo_occ=2*occupied, nocc=2,
        mo_energy=jnp.asarray([-1.4, -.9, .3, .8]),
        fock_matrix=jnp.asarray([[-1.1, .07, 0., 0.], [.07, -.6, 0., 0.],
                                 [0., 0., .5, -.04], [0., 0., -.04, 1.2]]),
        ao=ao, ao_deriv1=jnp.stack((ao, .12*ao, -.09*ao, .07*ao)),
        grid=SimpleNamespace(weights=jnp.asarray([.2, .4, .3])),
        rdm1=jnp.stack((jnp.diag(occupied),)*2),
        df_factors=jnp.asarray([[[.3, .1, -.2, .05], [.2, .4, .15, -.1],
                                 [.07, -.03, .2, .1], [.11, .05, -.04, .3]],
                                [[-.1, .08, .2, .1], [.04, .2, -.07, .06],
                                 [.13, .01, .3, -.08], [.02, -.12, .1, .2]]]),
        xc_density_floor=1e-12,
    )


def _values():
    # Two independent rows and independent X/Y expose invalid symmetry reuse.
    x = jnp.asarray([[[.21, -.17], [.08, .11]], [[-.09, .14], [.19, -.04]]])
    y = jnp.asarray([[[-.05, .13], [.18, -.12]], [[.16, -.07], [.03, .22]]])
    return x, y


def _data(*, path, linear, calls):
    molecule = _reference()
    orbo, orbv = molecule.mo_coeff[:, :2], molecule.mo_coeff[:, 2:]

    def xc(values):
        calls.append(values)
        coupled = values + .4*jnp.sum(values, axis=(-2, -1), keepdims=True)
        return .23*coupled if linear else jnp.clip(coupled, -.13, .17)

    kwargs = {}
    if path == "raw_df":
        kwargs["ao_mo_response_action_fn"] = response._restricted_df_mo_response_action(
            molecule.df_factors, orbo, orbv, .25,
            include_exchange=True, dtype=jnp.float64)
    else:
        # Nonlocal A/B terms also select the generic path, even with AO J/K.
        kwargs["nonlocal_xc_a_action_fn"] = lambda values: .11*values
        kwargs["nonlocal_xc_b_action_fn"] = lambda values: -.07*values[..., ::-1]
    return response._RestrictedResponseOperatorData(
        delta_eps=jnp.asarray([[1.6, 2.3], [1.1, 1.8]]),
        orbo=orbo, orbv=orbv,
        ao_response_action_fn=response._restricted_ao_response_action(
            molecule, .25, include_exchange=True, dtype=jnp.float64),
        xc_response_action_fn=xc, linear_xc_response=linear,
        fock_occupied=molecule.fock_matrix[:2, :2],
        fock_virtual=molecule.fock_matrix[2:, 2:],
        **kwargs,
    )


def _independent_rows(data, x, y):
    return (response._restricted_a_action(data, x) + response._restricted_b_action(data, y),
            response._restricted_b_action(data, x) + response._restricted_a_action(data, y))


@pytest.mark.parametrize("path", ["raw_df", "ao_nonlocal"])
@pytest.mark.parametrize("linear", [False, True])
def test_generic_full_td_shares_xc_between_rows(path, linear):
    calls = []
    data = _data(path=path, linear=linear, calls=calls)
    x, y = _values()
    expected = _independent_rows(data, x, y)
    calls.clear()
    actual = response._restricted_tdhf_action(data, x, y)
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_allclose(left, right, atol=2e-13, rtol=2e-13)
    assert len(calls) == (1 if linear else 2), (
        "Full TD must share the local XC result between its two rows; "
        "opaque callbacks still require separate X and Y evaluations")
    if not linear:
        np.testing.assert_allclose(calls[0], x, atol=0., rtol=0.)
        np.testing.assert_allclose(calls[1], y, atol=0., rtol=0.)


@pytest.mark.parametrize("path", ["raw_df", "ao_nonlocal"])
@pytest.mark.parametrize("linear", [False, True])
def test_shared_full_td_xc_preserves_jit_jvp_and_vjp(path, linear):
    data = _data(path=path, linear=linear, calls=[])
    x, y = _values()
    values = jnp.stack((x, y))
    direction = jnp.flip(values, axis=0)*.37

    def evaluate(values, independent):
        action = _independent_rows if independent else response._restricted_tdhf_action
        return jnp.stack(action(data, values[0], values[1]))

    actual = lambda values: evaluate(values, False)
    expected = lambda values: evaluate(values, True)
    np.testing.assert_allclose(jax.jit(actual)(values), expected(values),
                               atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(jax.jvp(actual, (values,), (direction,))[1],
                               jax.jvp(expected, (values,), (direction,))[1],
                               atol=2e-13, rtol=2e-13)
    probe = jnp.flip(expected(values), axis=-1)
    np.testing.assert_allclose(jax.vjp(actual, values)[1](probe)[0],
                               jax.vjp(expected, values)[1](probe)[0],
                               atol=3e-13, rtol=3e-13)


def test_full_casida_uses_noncanonical_builder_gap(monkeypatch):
    molecule = _reference()
    expected = (jnp.diag(molecule.fock_matrix)[2:][None, :]
                - jnp.diag(molecule.fock_matrix)[:2, None])
    seen = []
    sentinel = object()

    def solve(gaps, operator, **kwargs):
        seen.append(gaps)
        np.testing.assert_allclose(gaps, expected, atol=0., rtol=0.)
        # The preconditioner must describe the same noncanonical Fock action.
        assert callable(operator)
        return sentinel

    monkeypatch.setattr(casida, "solve_casida_from_tdhf_operator", solve)
    result = casida.RestrictedCasidaTDDFT(molecule)._kernel_impl(nstates=1)
    assert result is sentinel
    assert len(seen) == 1


@pytest.mark.parametrize("xc_spec", ["pbe", "pbe0"])
def test_standard_semilocal_builder_skips_extra_density_contraction(monkeypatch, xc_spec):
    molecule = _reference()

    def forbidden(*args, **kwargs):
        raise AssertionError("Standard semilocal tensor construction already evaluates density")

    monkeypatch.setattr(response, "_density_on_grid", forbidden)
    data = response._build_restricted_response_operator_data(
        molecule, SemilocalResponseFunctional(xc_spec))
    assert callable(data.xc_response_action_fn)
    assert data.linear_xc_response
    np.testing.assert_allclose(data.hybrid_fraction, 0.25 if xc_spec == "pbe0" else 0.,
                               atol=0., rtol=0.)


def test_arbitrary_local_callbacks_receive_original_density():
    molecule = _reference()
    expected = jnp.einsum("spq,gp,gq->g", molecule.rdm1, molecule.ao, molecule.ao)
    seen = []

    class Functional:
        exact_exchange_fraction = .2

        def local_hf_fraction(self, density):
            seen.append(("hf", density))
            return jnp.full_like(density, .2)

        def local_kernel(self, density):
            seen.append(("kernel", density))
            return .1 + .05*density

    data = response._build_restricted_response_operator_data(molecule, Functional())
    assert [label for label, _ in seen] == ["hf", "kernel"]
    for _, density in seen:
        np.testing.assert_allclose(density, expected, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(data.hybrid_fraction, .2, atol=0., rtol=0.)
