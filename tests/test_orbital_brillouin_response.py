"""Real Brillouin residuals use variational density gradients, not reported Fock."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("method", ["roks", "uks"])
def test_brillouin_matches_zero_chart_and_stationary_linearization(method):
    from gradscf.scf import orbital_optimization as orbital

    occupations = ((1., 1., 0.), (1., 0., 0.))
    dimension, rotate, density_from = orbital._rotation_functions(method, occupations)
    base = jnp.broadcast_to(jnp.eye(3), (1 if method == "roks" else 2, 3, 3))
    origin = jnp.zeros(dimension)
    forcing = jnp.array([[0., .4, -.2], [.4, 0., .3], [-.2, .3, 0.]])
    # This antisymmetric extension vanishes on physical symmetric densities;
    # the density gradient must be symmetrized before orbital projection.
    skew = jnp.array([[0., .11, -.05], [-.11, 0., .07], [.05, -.07, 0.]])

    def evaluate(density, args, *, compute_potential=True):
        h = jnp.diag(jnp.array([-1., .2, .9])) + skew + args["parameter"]*forcing
        energy = (jnp.sum(h*density) + .1*jnp.sum(density**2)
                  + .07*jnp.sum(density[0]*density[1]))
        reported = jnp.clip(h+.2*density+.07*density[::-1], -.02, .02)
        return energy, reported if compute_potential else None

    gradient = orbital._brillouin_gradient

    def brillouin(angles, parameter):
        return gradient(method, occupations, rotate(angles, base), density_from,
                        evaluate, {"parameter": parameter})

    def energy(angles, parameter):
        return evaluate(density_from(rotate(angles, base)),
                        {"parameter": parameter}, compute_potential=False)[0]

    chart_gradient = jax.grad(energy)
    # At theta=0 this equality holds even away from stationarity. Clipping the
    # reported Fock would change these nonzero residuals.
    np.testing.assert_allclose(brillouin(origin, .13), chart_gradient(origin, .13),
                               atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(chart_gradient(origin, 0.), 0., atol=2e-14, rtol=0)
    np.testing.assert_allclose(brillouin(origin, 0.), 0., atol=2e-14, rtol=0)
    # Body-frame Brillouin and chart gradients need not have equal Jacobians
    # at nonstationary points; compare their linearizations at the root only.
    np.testing.assert_allclose(jax.jacfwd(brillouin, 0)(origin, 0.),
        jax.jacfwd(chart_gradient, 0)(origin, 0.), atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(jax.jacfwd(brillouin, 1)(origin, 0.),
        jax.jacfwd(chart_gradient, 1)(origin, 0.), atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize("xc", ["pbe0", "wb97xd"])
def test_zero_diis_geometry_response_avoids_angle_gradient_residual(monkeypatch, xc):
    from gradscf import dft, gto
    from gradscf_tools.molecular_response import MolecularEnergy

    mf = dft.RKS(gto.M(atom="H 0 0 -.7; H 0 0 .7", basis="sto-3g", unit="Bohr", cart=True),
        xc=xc, grids_level=0, max_cycle=200, conv_tol=1e-12,
        conv_tol_density=1e-11, conv_tol_grad=1e-10).run()
    assert mf.converged
    original_grad = jax.grad

    def checked_grad(function, *args, **kwargs):
        if (getattr(function, "__module__", "") == "gradscf.scf.orbital_optimization"
                and getattr(function, "__qualname__", "") in {
                    "_problem_functions.<locals>.energy", "_minimize.<locals>.tangent_energy"}):
            raise AssertionError("Implicit stationarity/response must not differentiate the Cayley energy chart")
        return original_grad(function, *args, **kwargs)

    monkeypatch.setattr(jax, "grad", checked_grad)
    energy = MolecularEnergy(mf, forward="scf", jk_backend="direct", gradient_tolerance=1e-9)
    coordinates, field = energy.coordinates, jnp.array([0., 0., .001])
    direction = jnp.array([[0., 0., -.5], [0., 0., .5]])
    state = jax.jit(lambda r: energy.solve(r, field))(coordinates)
    assert bool(state.stationary) and int(state.iterations) == 0
    scalar = lambda q: energy(coordinates+q*direction, field)
    calculate = jax.jit(jax.value_and_grad(scalar))
    value, gradient = calculate(jnp.asarray(0.))
    hvp = jax.jit(lambda q: jax.jvp(jax.grad(scalar), (q,), (jnp.ones_like(q),))[1])(
        jnp.asarray(0.))
    assert np.isfinite(np.asarray([value, gradient, hvp])).all()
    for step in (3e-4, 1e-4):
        plus, minus = calculate(jnp.asarray(step)), calculate(jnp.asarray(-step))
        np.testing.assert_allclose(gradient, (plus[0]-minus[0])/(2*step), atol=3e-6, rtol=2e-5)
        np.testing.assert_allclose(hvp, (plus[1]-minus[1])/(2*step), atol=4e-6, rtol=2e-5)


@pytest.mark.parametrize("regularization,require_converged", [(1e-3, True), (0., False)],
                         ids=["regularized", "unchecked"])
def test_legacy_chart_response_policy_is_preserved(monkeypatch, regularization, require_converged):
    from gradscf import scf

    original_grad = jax.grad
    chart_calls = []

    def record_grad(function, *args, **kwargs):
        if (getattr(function, "__module__", "") == "gradscf.scf.orbital_optimization"
                and getattr(function, "__qualname__", "") == "_minimize.<locals>.tangent_energy"):
            chart_calls.append(True)
        return original_grad(function, *args, **kwargs)

    monkeypatch.setattr(jax, "grad", record_grad)
    config = scf.SCFDifferentiationConfig(mode="implicit", regularization=regularization,
                                         require_converged=require_converged)

    def solve():
        return scf.minimize_roks_from_integrals(
            overlap=jnp.eye(2), hcore=jnp.diag(jnp.array([-1., .5])),
            eri=jnp.zeros((2,)*4), nuclear_repulsion=0., ao=jnp.zeros((0, 2)),
            ao_deriv1=jnp.zeros((4, 0, 2)), grid_weights=jnp.zeros(0),
            mo_coeff=jnp.eye(2), mo_occ=np.array([[1., 0.], [1., 0.]]),
            max_iterations=0, gradient_tolerance=1e-9, differentiation=config)

    state = jax.jit(solve)()
    assert bool(state.stationary) and int(state.iterations) == 0
    assert chart_calls, "Legacy unchecked/regularized policies retain the Cayley-chart residual"
