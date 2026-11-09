"""Screened SCF must not escape into an unmatched dense response definition.

These are boundary/control-flow checks, not molecular accuracy claims. Real
facades/configs are used; spies stop numerical preparation or Hessian work so
the positive-cutoff rejection is checked at the public entry point.
"""
import jax.numpy as jnp
import pytest

from gradscf import dft, gto
from gradscf.scf import builders, diagnostics, stability
from gradscf.scf.rks import RKSConfig
from gradscf.scf.uks import UKSConfig, UKSResult


class _NumericalWorkReached(Exception):
    pass


def _numerical_work(*args, **kwargs):
    raise _NumericalWorkReached


def _molecule():
    return gto.M(atom="H 0 0 0; H 0 0 1.4", unit="Bohr", basis="sto-3g")


def _uks_result():
    """Finite solved-state metadata for testing facade storage and routing."""
    eye = jnp.eye(2)
    occ = jnp.asarray([1., 0.])
    levels = jnp.asarray([-.5, .2])
    return UKSResult(
        converged=True, total_energy=-1., electronic_energy=-1.7,
        nuclear_repulsion=.7, xc_energy=-.2, exact_exchange_fraction=0.,
        mo_coeff_alpha=eye, mo_coeff_beta=eye,
        mo_occ_alpha=occ, mo_occ_beta=occ,
        mo_energy_alpha=levels, mo_energy_beta=levels,
        density_matrix_alpha=jnp.diag(occ), density_matrix_beta=jnp.diag(occ),
        fock_matrix_alpha=jnp.diag(levels), fock_matrix_beta=jnp.diag(levels),
        overlap_matrix=eye, hcore_matrix=jnp.diag(levels), cycles=1)


@pytest.mark.parametrize("cutoff", [0., 1e-10])
def test_uks_kernel_does_not_publish_an_unmatched_screened_reference(monkeypatch, cutoff):
    source = dft.UKS(_molecule(), xc="pbe", grid_screen_tol=cutoff)
    result = _uks_result()
    monkeypatch.setattr(source, "_configure_jax_cache", lambda: None)
    monkeypatch.setattr(source, "_build_scf_result", lambda spec: result)
    monkeypatch.setattr(source, "_build_reference", _numerical_work)
    if cutoff == 0:
        # Preserve the established unscreened UKS reference-building route.
        with pytest.raises(_NumericalWorkReached):
            source.kernel()
        return
    assert source.kernel() == result.total_energy
    assert source.converged
    assert source.scf_result is result
    assert source.reference is None
    assert source.mo_coeff.shape == (2, 2, 2)
    assert source.make_rdm1().shape == (2, 2, 2)
    with pytest.raises(NotImplementedError, match="(?i)screened.*response|response.*screened"):
        source.to_reference()


@pytest.mark.parametrize("spin", ["restricted", "unrestricted"])
@pytest.mark.parametrize("cutoff", [0., 1e-10])
def test_reference_builders_reject_screening_before_preparing_inputs(monkeypatch, spin, cutoff):
    restricted = spin == "restricted"
    config = (RKSConfig if restricted else UKSConfig)(xc_spec="pbe", grid_screen_tol=cutoff)
    build = builders._build_restricted_reference if restricted else builders._build_unrestricted_reference
    input_builder = "build_rks_integral_inputs" if restricted else "build_uks_integral_inputs"
    config_name = "rks_config" if restricted else "uks_config"
    monkeypatch.setattr(builders, input_builder, _numerical_work)
    expected = NotImplementedError if cutoff else _NumericalWorkReached
    with pytest.raises(expected, match="(?i)screen" if cutoff else None):
        build(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", unit="Bohr", spin=0,
              xc_spec="pbe", **{config_name: config})


@pytest.mark.parametrize("method", [dft.RKS, dft.UKS])
@pytest.mark.parametrize("operation", ["stability", "stabilize"])
@pytest.mark.parametrize("cutoff", [0., 1e-10])
def test_facade_stability_rejects_screening_before_reusing_a_snapshot(
        monkeypatch, method, operation, cutoff):
    source = method(_molecule(), xc="pbe", grid_screen_tol=cutoff)
    source.e_tot = -1.  # Select the existing-snapshot branch of stabilize().
    state_reader = "_restricted_scf_state" if method is dft.RKS else "_unrestricted_scf_state"
    monkeypatch.setattr(diagnostics, state_reader, _numerical_work)
    expected = NotImplementedError if cutoff else _NumericalWorkReached
    with pytest.raises(expected, match="(?i)screen" if cutoff else None):
        getattr(source, operation)()


@pytest.mark.parametrize("cutoff", [0., 1e-10])
def test_integral_uks_stability_rejects_screening_before_constructing_curvature(monkeypatch, cutoff):
    config = UKSConfig(xc_spec="pbe", grid_screen_tol=cutoff)
    monkeypatch.setattr(stability, "_problem_functions", _numerical_work)
    expected = NotImplementedError if cutoff else _NumericalWorkReached
    with pytest.raises(expected, match="(?i)screen" if cutoff else None):
        stability.uks_stability(_uks_result(), config=config,
            eri=jnp.zeros((2, 2, 2, 2)), ao=jnp.ones((1, 2)),
            ao_deriv1=jnp.zeros((4, 1, 2)), grid_weights=jnp.ones(1))


@pytest.mark.parametrize("cutoff", [0., 1e-10])
def test_integral_uks_stabilization_rejects_screening_before_running_scf(monkeypatch, cutoff):
    config = UKSConfig(xc_spec="pbe", grid_screen_tol=cutoff)
    monkeypatch.setattr(stability, "_stabilize", _numerical_work)
    expected = NotImplementedError if cutoff else _NumericalWorkReached
    with pytest.raises(expected, match="(?i)screen" if cutoff else None):
        stability.stabilize_uks_from_integrals(config=config,
            eri=jnp.zeros((2, 2, 2, 2)), ao=jnp.ones((1, 2)),
            ao_deriv1=jnp.zeros((4, 1, 2)), grid_weights=jnp.ones(1))
