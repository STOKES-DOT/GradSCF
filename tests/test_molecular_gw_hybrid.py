"""Complete hybrid KS -> exact-pair GW geometry maps, CPU float64.

H2/STO-3G, Cartesian AOs, grid level 0; coordinates are in bohr and energies
in Hartree. These small finite-grid tests validate response composition, not
basis/frequency convergence. Native auxiliary-basis RI is a separate capability.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import dft, gto
from gradscf_tools.gw_response import MolecularGW


@pytest.fixture(scope="module", params=["pbe0", "wb97xd"])
def reference(request):
    mol = gto.M(atom="H 0 0 -.7; H 0 0 .7", basis="sto-3g", unit="Bohr", cart=True)
    mf = dft.RKS(mol, xc=request.param, grids_level=0,
                 conv_tol=1e-12, conv_tol_grad=1e-10, max_cycle=100).run()
    assert mf.converged
    return mf


def test_hybrid_constructor(reference):
    model = MolecularGW(reference)
    assert model.method == "g0w0"
    np.testing.assert_array_equal(model.coordinates, reference.mol.to_spec().coords_bohr)


def test_outer_tolerance_controls_the_converged_root(reference):
    for method in ("evgw", "qsgw", "scgw"):
        model = MolecularGW(reference, method=method, conv_tol=1e-12)
        assert model.conv_tol == 1e-12
        for tolerance in (0., -1e-12, np.nan, np.inf):
            with pytest.raises(ValueError, match="conv_tol"):
                MolecularGW(reference, method=method, conv_tol=tolerance)
    with pytest.raises(ValueError, match="outer|G0W0"):
        MolecularGW(reference, method="g0w0", conv_tol=1e-12)


def _observables(result, method):
    if method == "scgw":
        return jnp.array([result.total_energy, result.chemical_potential,
                          result.density_matrix[0, 0], result.density_matrix[0, 1]])
    if method == "qsgw":
        occupied = result.mo_coeff[:, :1]
        density = 2 * occupied @ occupied.T
        return jnp.concatenate([result.mo_energy, jnp.array([density[0, 0], density[0, 1]])])
    return jnp.concatenate([result.mo_energy, result.screening_energy])


@pytest.mark.parametrize("method", ["g0w0", "evgw", "qsgw", "scgw"])
def test_hybrid_exact_pair_geometry_response(reference, method):
    """Moving native inputs, relaxed KS, and the complete G/W response."""
    options = dict(method=method, nw=24, max_cycle=200)
    if method == "scgw":
        options.update(beta=20., mixing=.5, max_cycle=1600, conv_tol=1e-12)
    else:
        options.update(eta=1e-3, damp=.3)
    model = MolecularGW(reference, **options)
    coordinates = model.coordinates
    direction = jnp.array([[0., 0., -.5], [0., 0., .5]])

    def values(r):
        return _observables(model(r), method)

    def values_and_state(r):
        result = model(r)
        return _observables(result, method), result

    primal, tangent, result = jax.jit(lambda r: jax.jvp(
        values_and_state, (r,), (direction,), has_aux=True))(coordinates)
    assert bool(result.converged)
    assert np.isfinite(primal).all() and np.isfinite(tangent).all()
    assert np.linalg.norm(np.asarray(tangent)) > 1e-3
    if method == "scgw":
        assert result.mo_energy is None
        assert result.total_energy is not None
        assert np.isfinite(result.self_energy_iw).all()
        assert abs(float(result.particle_number_error)) < 1e-9
        assert float(result.fixed_point_residual) < 1e-8
    else:
        assert np.asarray(result.qp_computed_mask).all()
        assert np.asarray(result.converged_mask).all()
        if method == "evgw":
            np.testing.assert_allclose(result.screening_energy, result.mo_energy, atol=1e-10, rtol=0)

    weights = jnp.arange(1, primal.size + 1, dtype=float) / primal.size
    reverse = jax.jit(jax.grad(lambda r: weights @ values(r)))(coordinates)
    assert np.isfinite(reverse).all()
    np.testing.assert_allclose(jnp.sum(reverse * direction), weights @ tangent,
                               atol=2e-8, rtol=2e-7)
    np.testing.assert_allclose(reverse.sum(axis=0), 0., atol=1e-8, rtol=0)

    evaluate = jax.jit(values)
    for step in (3e-4, 1e-4):
        # Each call rebuilds the same exact-pair approximation and reconverges
        # the functional/KS state and G/W at its own displaced geometry.
        plus = evaluate(coordinates + step * direction)
        minus = evaluate(coordinates - step * direction)
        assert np.isfinite(plus).all() and np.isfinite(minus).all()
        finite = (plus - minus) / (2 * step)
        np.testing.assert_allclose(tangent, finite, atol=3e-6, rtol=4e-5)


@pytest.mark.parametrize("method", ["qsgw", "scgw"])
def test_matrix_methods_require_complete_orbital_space(reference, method):
    for window in (dict(orbs=(0,)), dict(g_orbitals=(0,)),
                   dict(screening_occupied=()), dict(screening_virtual=())):
        with pytest.raises(ValueError, match="complete|full"):
            MolecularGW(reference, method=method, **window)


def test_scgw_rejects_qp_consumers(reference):
    model = MolecularGW(reference, method="scgw", beta=20.)
    with pytest.raises(NotImplementedError):
        model.optical(model.coordinates)
    with pytest.raises(NotImplementedError):
        model.pole_seeds(model.coordinates)
