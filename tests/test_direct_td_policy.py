"""Direct TD admission preserves automatic cache policy and live NN inputs.

CPU float64, Hartree, H2 native Cartesian 6-31G integrals. The cache-size
metadata fixture exercises budget rejection without allocating a large ERI.
The response references are bounded deterministic arrays, not converged SCF.
"""
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import dft
from gradscf.integrals.backends.native.jk import NativeDirectBasis
from gradscf.tddft.execution import _cacheable
from test_direct_td_program_reuse import driver, reference
from test_td_execution_reuse import compilations, density_inputs, neural_energy


@pytest.mark.parametrize("full", [False, True], ids=["tda", "full"])
def test_automatic_zero_cutoff_without_cache_admission_reuses_direct_program(monkeypatch, full):
    # Only admission metadata changes: the actual native basis/integrals stay
    # small. Even two cache copies alone exceed the full 1024 MB budget.
    monkeypatch.setattr(NativeDirectBasis, "eri_cache_nbytes", property(lambda self: 10**12))
    first_ref, moved_ref = reference(), reference(.08)
    first = driver(first_ref, full, cutoff=0., cache="auto")
    solver = first._build_solver()
    assert solver._memory_plan(2, rpa=full).eri_cache_bytes == 0
    assert _cacheable(solver), (
        "Automatic exact shell-direct work with an inadmissible ERI cache must reuse its program")

    prepared = []
    original = NativeDirectBasis.prepare_jk

    def prepare(basis, **kwargs):
        prepared.append(kwargs.get("cache_bytes", 0))
        assert kwargs.get("cache_bytes", 0) == 0, "Admission must preserve the eager zero-cache policy"
        return original(basis, **kwargs)

    monkeypatch.setattr(NativeDirectBasis, "prepare_jk", prepare)
    expected = jax.block_until_ready(driver(moved_ref, full, cutoff=0., cache="none").kernel())
    initial = jax.block_until_ready(first.kernel())
    moved = driver(moved_ref, full, cutoff=0., cache="auto")
    with compilations(monkeypatch) as events:
        repeated = jax.block_until_ready(first.kernel())
        updated = jax.block_until_ready(moved.kernel())
    # An already compiled matching program need not call this Python wrapper.
    assert not any(prepared)
    assert initial.converged and repeated.converged and updated.converged and expected.converged
    np.testing.assert_allclose(initial.excitation_energies, repeated.excitation_energies,
                               atol=2e-12, rtol=0.)
    np.testing.assert_allclose(updated.excitation_energies, expected.excitation_energies,
                               atol=2e-10, rtol=2e-10)
    assert float(jnp.min(updated.excitation_energies-initial.excitation_energies)) > .07
    assert not events, f"Auto no-cache direct response triggered {len(events)} warm compilations"


def _neural_driver(ref, model, weights, scale, full):
    calculation = driver(ref, full)
    calculation.nstates = 1
    calculation.xc_functional = model
    calculation.xc_params = {"weight": weights, "scale": scale}
    return calculation


@pytest.mark.parametrize("full", [False, True], ids=["tda", "full"])
def test_direct_custom_xc_parameter_jvp_vjp_fd_stay_dynamic(monkeypatch, full):
    ref = reference()
    width = ref.mo_coeff.shape[-1]
    weights = jnp.asarray(np.linspace(.1, .4, width*3).reshape(width, 3))
    model = dft.custom_xc(input_fn=density_inputs, energy_fn=neural_energy,
                          name="direct-parameter-policy")
    point, changed = jnp.asarray(.03), jnp.asarray(.04)
    assert _cacheable(_neural_driver(ref, model, weights, point, full)._build_solver())

    def observable(scale):
        return _neural_driver(ref, model, weights, scale, full).kernel().excitation_energies[0]

    def eager_observable(scale):
        solver = _neural_driver(ref, model, weights, scale, full)._build_solver()
        result = solver._kernel_impl(1) if full else solver._tda_impl(1)
        return result.excitation_energies[0]

    value = jax.jit(observable)
    forward = jax.jit(lambda scale: jax.jvp(observable, (scale,), (jnp.ones_like(scale),)))
    reverse = jax.jit(jax.value_and_grad(observable))
    initial = value(point)
    initial_forward = forward(point)
    initial_reverse = reverse(point)
    jax.block_until_ready((initial, initial_forward, initial_reverse))
    with compilations(monkeypatch) as events:
        updated = value(changed)
        updated_forward = forward(changed)
        updated_reverse = reverse(changed)
        jax.block_until_ready((updated, updated_forward, updated_reverse))
    assert not events, f"Changed direct NN parameters triggered {len(events)} warm compilations"
    assert abs(float(updated-initial)) > 1e-7
    for scale, primal, tangent, adjoint in (
            (point, initial, initial_forward, initial_reverse),
            (changed, updated, updated_forward, updated_reverse)):
        np.testing.assert_allclose(primal, eager_observable(scale), atol=2e-10, rtol=2e-10)
        np.testing.assert_allclose(tangent[0], primal, atol=2e-10, rtol=2e-10)
        np.testing.assert_allclose(adjoint[0], primal, atol=2e-10, rtol=2e-10)
        assert np.isfinite(np.asarray((primal, tangent[1], adjoint[1]))).all()
        assert abs(float(adjoint[1])) > 1e-7
        step = 1e-4
        finite_difference = (value(scale+step)-value(scale-step))/(2*step)
        np.testing.assert_allclose(tangent[1], adjoint[1], atol=3e-8, rtol=3e-6)
        np.testing.assert_allclose(adjoint[1], finite_difference, atol=3e-7, rtol=3e-5)


def test_native_provider_subclass_keeps_eager_fallback():
    class OpaqueNative(NativeDirectBasis):
        pass

    ref = reference()
    basis = ref.direct_basis
    opaque = OpaqueNative(basis.plan, basis.parameters)
    assert not _cacheable(driver(replace(ref, direct_basis=opaque))._build_solver())


def test_unknown_hashable_physical_callback_keeps_eager_fallback():
    @dataclass(frozen=True)
    class GridHF:
        exact_exchange_fraction: float = 0.

        def grid_hf_fraction(self, molecule):
            return jnp.full_like(molecule.grid.weights, .25)

        def local_kernel(self, density):
            return jnp.zeros_like(density)

    calculation = driver(reference())
    calculation.xc_functional = GridHF()
    assert hash(calculation.xc_functional) is not None
    assert not _cacheable(calculation._build_solver())
