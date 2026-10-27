"""Fresh same-signature solves reuse executables without capturing Hamiltonians.

All examples use float64, Hartree, and deterministic tiny array Hamiltonians.
Compilation is counted at the backend boundary, rather than inferred from time.
"""

from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@contextmanager
def _backend_compilations(monkeypatch):
    from jax._src import compiler

    original = compiler.backend_compile_and_load
    calls = []

    def counted(*args, **kwargs):
        calls.append(str(args[1] if len(args) > 1 else kwargs.get("module", "compile")))
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(compiler, "backend_compile_and_load", counted)
        yield calls


def _rks_inputs(hcore, enuc):
    return dict(
        overlap=jnp.eye(1),
        hcore=jnp.asarray([[hcore]]),
        eri=jnp.full((1, 1, 1, 1), .6),
        nelectron=2,
        nuclear_repulsion=jnp.asarray(enuc),
        ao=jnp.asarray([[1.], [.7], [.2]]),
        ao_deriv1=jnp.asarray([[[1.], [.7], [.2]],
                              [[.1], [.2], [.1]],
                              [[.2], [.1], [.1]],
                              [[.1], [.1], [.2]]]),
        grid_weights=jnp.asarray([.3, .4, .3]),
    )


@pytest.mark.parametrize("xc", ["hf", "pbe"])
def test_fresh_rks_inputs_reuse_compilation_and_keep_hamiltonian_dynamic(monkeypatch, xc):
    from gradscf.scf.rks import RKSConfig, run_rks_from_integrals

    first_inputs = _rks_inputs(-1., .3)
    second_inputs = _rks_inputs(-.8, .4)
    first = run_rks_from_integrals(**first_inputs,
        config=RKSConfig(xc_spec=xc, max_cycle=5, conv_tol=1e-12))
    assert first.converged
    with _backend_compilations(monkeypatch) as compilations:
        second = run_rks_from_integrals(**second_inputs,
            config=RKSConfig(xc_spec=xc, max_cycle=5, conv_tol=1e-12))
        jax.block_until_ready(second)
    assert second.converged
    np.testing.assert_allclose(second.total_energy-first.total_energy, .5, atol=2e-12)
    assert not compilations, f"Fresh RKS inputs triggered {len(compilations)} backend compilations"


def _cc_inputs(scale):
    h = jnp.asarray([[-1., .012], [.012, .7]])
    factors = jnp.asarray([[[.5, .04], [.04, .35]],
                           [[.1, .02], [.02, .2]]])
    eri = scale*jnp.einsum("Lpq,Lrs->pqrs", factors, factors)
    return h, eri


def test_rks_dense_eri_spectral_factorization_precedes_compiled_iterations():
    from gradscf.scf.rks import RKSConfig, run_rks_from_integrals

    result = run_rks_from_integrals(**_rks_inputs(-1., .3),
        config=RKSConfig(xc_spec="hf", jk_backend="df", max_cycle=5, conv_tol=1e-12))
    assert result.converged
    np.testing.assert_allclose(result.total_energy, -1.1, atol=2e-12)


def test_fresh_cc_inputs_reuse_compilation_and_keep_integrals_dynamic(monkeypatch):
    from gradscf.cc import CCConfig, run_cc

    h1, g1 = _cc_inputs(1.)
    h2, g2 = _cc_inputs(1.1)
    first = run_cc(h1, g1, nocc=1,
        config=CCConfig(conv_tol=1e-12, residual_tol=1e-11))
    jax.block_until_ready(first)
    assert first.converged
    with _backend_compilations(monkeypatch) as compilations:
        second = run_cc(h2, g2, nocc=1,
            config=CCConfig(conv_tol=1e-12, residual_tol=1e-11))
        jax.block_until_ready(second)
    assert second.converged
    assert float(second.residual_norm) < 1e-11
    assert abs(float(second.total_energy-first.total_energy)) > 1e-4
    assert np.max(np.abs(np.asarray(second.t2-first.t2))) > 1e-6
    assert not compilations, f"Fresh CC inputs triggered {len(compilations)} backend compilations"


def test_reused_cc_kernel_preserves_implicit_gradient_and_hvp():
    from gradscf.cc import CCConfig, run_cc

    h, eri = _cc_inputs(1.)
    cfg = CCConfig(conv_tol=1e-13, residual_tol=1e-12,
                   adjoint_tol=1e-12, adjoint_restart=8)

    def energy(scale):
        return run_cc(h, scale*eri, nocc=1, config=cfg).total_energy

    point = jnp.asarray(1.)
    gradient = jax.jit(jax.grad(energy))
    actual = gradient(point)
    step = 1e-4
    expected = (energy(point+step)-energy(point-step))/(2*step)
    np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-7)
    hvp = jax.jit(lambda x: jax.jvp(gradient, (x,), (jnp.ones_like(x),))[1])(point)
    fd_hvp = (gradient(point+step)-gradient(point-step))/(2*step)
    np.testing.assert_allclose(hvp, fd_hvp, atol=2e-8, rtol=2e-6)
