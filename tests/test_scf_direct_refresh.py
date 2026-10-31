"""An accumulated direct J/K estimate must pass a fresh convergence check."""
import jax
import jax.numpy as jnp
import pytest

from gradscf.scf.rks import RKSConfig, _run_scf_iterations_lax_core


def run(kind, *, backend="direct", fallback=False):
    h = jnp.diag(jnp.array([-1., 1.]))
    eye, zero = jnp.eye(2), jnp.zeros((2, 2))
    occ = jnp.array([2., 0.])
    def builder(density, coeff, occupation, energies, previous, j, k):
        energy, fock = jnp.asarray(0.), h
        if previous is None:
            if kind == "fock":
                fock = fock+jnp.array([[0., 1e-6], [1e-6, 0.]])
            elif kind == "energy":
                energy = jnp.asarray(1e-6)
            elif kind == "nonfinite":
                energy = jnp.asarray(jnp.nan)
        return energy, jnp.asarray(0.), fock, zero, zero
    options = dict(refresh_on_convergence=True) if fallback else {}
    return _run_scf_iterations_lax_core(
        h=h, s=eye, x=eye, energy_and_fock_builder=builder,
        cfg=RKSConfig(xc_spec="hf", jk_backend=backend, max_cycle=3,
            conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10),
        mo_occ_fixed=occ, diis_basis=eye, skip_first_fock_damping=True,
        density=jnp.diag(occ), mo_coeff=eye, mo_occ=occ, mo_energy=jnp.diag(h),
        raw_fock=h, j_mat=zero, k_mat=zero, **options)


@pytest.mark.parametrize("kind", ["fock", "energy", "nonfinite"])
def test_incremental_false_convergence_is_rejected_under_jit(kind):
    result = jax.jit(lambda: run(kind))()
    assert not bool(result[0]), "The fresh physical Fock/energy must meet the original gates"


def test_full_backend_is_unchanged_and_matching_direct_build_converges():
    assert bool(jax.jit(lambda: run("fock", backend="full"))()[0])
    assert bool(jax.jit(lambda: run("matched"))()[0])


def test_native_provider_in_full_facade_can_request_the_same_refresh():
    assert not bool(jax.jit(lambda: run("fock", backend="full", fallback=True))()[0])
