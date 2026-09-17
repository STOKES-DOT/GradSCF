"""Shared convergence contract, independent of an XC backend."""
import jax
import jax.numpy as jnp
import pytest

from gradscf.scf.convergence import convergence_reached


def check(de=0., dd=0., grad=0., **kwargs):
    return convergence_reached(de, dd, grad, conv_tol=kwargs.pop('conv_tol', 1e-8),
                               conv_tol_density=1e-6, conv_tol_grad=1e-7, **kwargs)


@pytest.mark.parametrize('metrics', [(1e-7, 0., 0.), (0., 1e-5, 0.), (0., 0., 1e-6)])
def test_every_threshold_is_required(metrics):
    assert not bool(check(*metrics))


@pytest.mark.parametrize('index', range(3))
@pytest.mark.parametrize('bad', [jnp.nan, jnp.inf])
def test_energy_only_still_rejects_nonfinite(index, bad):
    metrics = [0., 0., 0.]
    metrics[index] = bad
    assert not bool(check(*metrics, energy_only=True))


def test_energy_only_and_fixed_cycle_contract_under_jit():
    assert bool(jax.jit(lambda: check(dd=1., grad=1., energy_only=True))())
    assert not bool(check(conv_tol=0.))
    assert not bool(check(has_prior_cycle=False))
    assert bool(check())


def _rks_toy_run(density_tol, *, conv_tol=1., override=None, grad_tol=10.):
    from gradscf.scf.rks import RKSConfig, _run_scf_iterations_lax_core
    h = jnp.diag(jnp.array([-1., 1.]))
    eye = jnp.eye(2)
    zero = jnp.zeros((2, 2))
    occ = jnp.array([2., 0.])
    def builder(dm, *_):
        # Flat energy does not imply a stationary density.
        off = .1 + .2 * dm[0, 0]
        fock = h + jnp.array([[0., off], [off, 0.]])
        return jnp.array(0.), jnp.array(0.), fock, zero, zero
    return _run_scf_iterations_lax_core(
        h=h, s=eye, x=eye, energy_and_fock_builder=builder,
        cfg=RKSConfig(max_cycle=2, conv_tol=conv_tol, conv_tol_density=density_tol,
                      conv_tol_grad=grad_tol),
        mo_occ_fixed=occ, diis_basis=eye, skip_first_fock_damping=True,
        density=zero, mo_coeff=eye, mo_occ=occ, mo_energy=jnp.diag(h),
        raw_fock=h, j_mat=zero, k_mat=zero, density_convergence_tol=override,
    )


def test_rks_reads_density_tolerance_and_override_never_bypasses_energy():
    assert bool(_rks_toy_run(10.)[0])
    assert not bool(_rks_toy_run(1e-14)[0])
    assert not bool(_rks_toy_run(10., grad_tol=1e-14)[0])
    assert not bool(_rks_toy_run(10., conv_tol=0., override=10.)[0])


def test_level_shift_reporting_preserves_selected_occupied_branch():
    from gradscf.scf.rks import RKSConfig, _maybe_run_extra_final_cycle
    eye = jnp.eye(2)
    # Occupying the higher orbital is stationary, and must not be refilled.
    coeff = eye[:, ::-1]
    density = jnp.diag(jnp.array([0., 2.]))
    fock = jnp.diag(jnp.array([-1., 1.]))
    output = _maybe_run_extra_final_cycle(
        density=density, mo_coeff=coeff, mo_energy=jnp.array([1., 4.]),
        energy=jnp.array(2.), xc_energy=jnp.array(0.), raw_fock=fock,
        converged=jnp.array(True), x=eye, jk_builder=None, ao=None,
        ao_deriv1=None, weights=None, h=fock, s=eye, enuc=jnp.array(0.),
        cfg=RKSConfig(level_shift=3.), alpha=jnp.array(1.), xc_kind='HF',
        mo_occ_fixed=jnp.array([2., 0.]), j_mat=None, k_mat=None, traceable=True,
    )
    assert jnp.array_equal(output[0], density)
    assert jnp.array_equal(output[1], coeff)
    assert jnp.array_equal(output[2], jnp.array([1., -1.]))
    assert bool(output[-1])


@pytest.mark.parametrize('method', ['roks', 'gks'])
@pytest.mark.parametrize('metric', ['energy', 'energy_and_residual'])
def test_open_shell_fixed_count_zero_tolerance(method, metric):
    from gradscf.scf.roks import ROKSConfig, run_roks_from_integrals
    from gradscf.scf.gks import GKSConfig, run_gks_from_integrals
    inputs = dict(overlap=jnp.eye(2), hcore=jnp.diag(jnp.array([-1., 1.])),
                  eri=jnp.zeros((2, 2, 2, 2)), nuclear_repulsion=0.,
                  ao=jnp.zeros((0, 2)), ao_deriv1=jnp.zeros((4, 0, 2)),
                  grid_weights=jnp.zeros(0))
    cfg = dict(xc_spec='hf', max_cycle=3, conv_tol=0., convergence_metric=metric)
    if method == 'roks':
        result = run_roks_from_integrals(**inputs, nalpha=1, nbeta=0, config=ROKSConfig(**cfg))
    else:
        result = run_gks_from_integrals(**inputs, nelectron=1, config=GKSConfig(**cfg))
    assert not bool(result.converged)
    assert int(result.cycles) == 3
