"""Generalized consumers share spatial kernels without discarding spin mixing.

Small deterministic float64/complex128 arrays test exact operator contracts.
The direct-provider double supplies the identical spatial kernel; native
integral generation is an independent integration gate, not an RI accuracy claim.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.scf import gks


def _arrays():
    factors = np.sin(np.arange(12).reshape(3, 2, 2)+1.)*.2
    factors = .5*(factors+factors.swapaxes(1, 2))
    source = np.cos(np.arange(12).reshape(4, 3)+.4) + 1j*np.sin(np.arange(12).reshape(4, 3)+.7)
    density = source@source.conj().T/8.
    return jnp.asarray(factors), jnp.asarray(density)


def _eri(factors):
    # All layouts represent symmetric spatial AO pairs, including VJP directions.
    factors = .5*(factors+factors.swapaxes(1, 2))
    return jnp.einsum("Qpq,Qrs->pqrs", factors, factors)


def _spatial_jk(eri, density):
    return (jnp.einsum("pqrs,...sr->...pq", eri, density),
            jnp.einsum("prqs,...rs->...pq", eri, density))


def _dense_spin_jk(eri, density, with_k=True):
    n = density.shape[-1]//2
    j = _spatial_jk(eri, density[:n, :n]+density[n:, n:])[0]
    zero = jnp.zeros_like(j)
    spin_j = jnp.block([[j, zero], [zero, j]])
    blocks = density.reshape(2, n, 2, n).transpose(0, 2, 1, 3)
    k = _spatial_jk(eri, blocks)[1].transpose(0, 2, 1, 3).reshape(density.shape)
    hermitian = lambda a: .5*(a+a.conj().T)
    return hermitian(spin_j), hermitian(k) if with_k else jnp.zeros_like(density)


class _DirectKernel:
    def __init__(self, eri, eri_lr=None):
        self.eri, self.eri_lr = eri, eri_lr


def _selected_spy(monkeypatch):
    from gradscf.integrals.backends.native import jk
    calls = []

    def selected(provider, density, *, with_j=True, with_k=True, omega=0., **kwargs):
        calls.append((bool(with_j), bool(with_k), tuple(density.shape), float(omega)))
        kernel = provider.eri_lr if omega else provider.eri
        j, k = _spatial_jk(kernel, density)
        return (j if with_j else jnp.zeros_like(density),
                k if with_k else jnp.zeros_like(density))

    monkeypatch.setattr(jk, "_selected_jk", selected)
    return calls


@pytest.mark.parametrize("backend", ["full", "packed", "df", "direct"])
@pytest.mark.parametrize("with_k", [False, True])
def test_generalized_jk_shared_full_packed_df_direct_value_and_array_ad(monkeypatch, backend, with_k):
    factors, density = _arrays()
    calls = _selected_spy(monkeypatch)

    def action(factors, density):
        factors = .5*(factors+factors.swapaxes(1, 2))
        eri = _eri(factors)
        kwargs = dict(with_k=with_k)
        if backend == "packed":
            row, col = np.tril_indices(factors.shape[-1])
            pair = eri[row[:, None], col[:, None], row[None, :], col[None, :]]
            eri = pair[np.tril_indices(len(row))]
        elif backend == "df":
            kwargs["df_factors"] = factors
            eri = None
        elif backend == "direct":
            kwargs["direct_basis"] = _DirectKernel(eri)
            eri = None
        return gks.generalized_jk(eri, density, **kwargs)

    expected = lambda b, d: _dense_spin_jk(_eri(b), d, with_k)
    actual = action(factors, density)
    for value, reference in zip(actual, expected(factors, density)):
        np.testing.assert_allclose(value, reference, atol=2e-12, rtol=2e-12)
    if with_k:
        assert np.linalg.norm(np.asarray(actual[1])[:2, 2:].imag) > 1e-4
    if backend == "direct":
        assert len(calls) == (2 if with_k else 1)
        assert calls[0][:3] == (True, False, (2, 2))
        if with_k:
            assert calls[1][0:2] == (False, True)
            assert np.prod(calls[1][2][:-2]) == 4

    delta_b = .03*jnp.cos(jnp.arange(factors.size).reshape(factors.shape))
    delta_b = .5*(delta_b+delta_b.swapaxes(1, 2))
    delta_d = .02*density
    actual_jvp = jax.jvp(action, (factors, density), (delta_b, delta_d))[1]
    expected_jvp = jax.jvp(expected, (factors, density), (delta_b, delta_d))[1]
    for value, reference in zip(actual_jvp, expected_jvp):
        np.testing.assert_allclose(value, reference, atol=3e-12, rtol=3e-12)
    objective = lambda fn, b, d: sum(jnp.sum(jnp.abs(x)**2) for x in fn(b, d))
    actual_vjp = jax.grad(lambda b, d: objective(action, b, d), argnums=(0, 1))(factors, density)
    expected_vjp = jax.grad(lambda b, d: objective(expected, b, d), argnums=(0, 1))(factors, density)
    for value, reference in zip(actual_vjp, expected_vjp):
        np.testing.assert_allclose(value, reference, atol=3e-11, rtol=3e-11)


class _ArrayGrid:
    def __init__(self, ao, derivative, weights):
        self.ao, self.derivative, self.weights = ao, derivative, weights
        self.block_size, self.calls = 2, 0

    def reduce(self, function, initial, *, deriv):
        assert deriv in (0, 1)
        for start in range(0, len(self.weights), self.block_size):
            part = slice(start, start+self.block_size)
            initial = function(initial, self.ao[part], self.derivative[:, part], None,
                               jnp.zeros((len(self.weights[part]), 3)), self.weights[part])
            self.calls += 1
        return initial


def _grid_arrays():
    ao = .2+.03*jnp.sin(jnp.arange(10).reshape(5, 2))
    derivative = .01*jnp.cos(jnp.arange(40).reshape(4, 5, 2))
    derivative = derivative.at[0].set(ao)
    return ao, derivative, jnp.asarray([.2, .3, .1, .4, .25])


@pytest.mark.parametrize("mode", ["ghf-grid-free", "gks-col-lda", "gks-ncol-lda", "gks-col-pbe"])
def test_generalized_grid_free_hf_and_streamed_gks_contract(monkeypatch, mode):
    if mode == "ghf-grid-free":
        from gradscf import gto, scf
        from gradscf.scf.inputs import assembly
        prepare = assembly._prepare_basis_grid_context

        def no_hf_grid(**kwargs):
            assert kwargs["needs_grid"] is False
            return prepare(**kwargs)

        monkeypatch.setattr(assembly, "_prepare_basis_grid_context", no_hf_grid)
        mf = scf.GHF(gto.M(atom="H .07 .11 .19", basis="sto-3g", spin=1), max_cycle=4)
        mf.kernel()
        assert mf.converged
        return
    pytest.importorskip("jax_xc")
    factors, density = _arrays()
    ao, derivative, weights = _grid_arrays()
    cfg = gks.GKSConfig(xc_spec="pbe" if mode.endswith("pbe") else "lda_x",
                        collinear="ncol" if "ncol" in mode else "col", grid_block_size=-1)
    common = dict(density=density, hcore=jnp.diag(jnp.asarray([-1., -.3])), eri=_eri(factors),
                  nuclear_repulsion=.2, grid_weights=weights)
    expected = gks.generalized_energy_and_fock(**common, ao=ao, ao_deriv1=derivative, config=cfg)
    provider = _ArrayGrid(ao, derivative, weights)
    actual = gks.generalized_energy_and_fock(**common, ao=jnp.zeros((0, 2)),
        ao_deriv1=jnp.zeros((4, 0, 2)), grid_ao=provider, config=replace(cfg, grid_block_size=2))
    assert provider.calls == 3
    for value, reference in zip(actual, expected):
        np.testing.assert_allclose(value, reference, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("backend", ["full", "direct", "ri-j-direct-k"])
def test_generalized_rsh_full_direct_and_ri_j_direct_k_contract(monkeypatch, backend):
    pytest.importorskip("jax_xc")
    from gradscf.scf import rsh
    factors, density = _arrays()
    eri, eri_lr = _eri(factors), _eri(.57*factors)
    provider = _DirectKernel(eri, eri_lr)
    calls = _selected_spy(monkeypatch)
    monkeypatch.setattr(rsh, "_require_direct_rsh_provider", lambda value: None)
    ao, derivative, weights = _grid_arrays()
    cfg = gks.GKSConfig(xc_spec="cam-b3lyp", jk_backend="df" if backend == "ri-j-direct-k" else backend,
                        only_dfj=backend == "ri-j-direct-k")
    kwargs = dict(eri=eri if backend == "full" else None,
                  eri_lr=eri_lr if backend == "full" else None)
    if backend != "full":
        kwargs["direct_basis"] = provider
    if backend == "ri-j-direct-k":
        kwargs["df_factors"] = .83*factors
    h = jnp.diag(jnp.asarray([-1., -.3, -1., -.3]))
    _, long_range, short_range = rsh.rsh_preset(cfg.xc_spec).to_range_separated_hybrid_coefficients()
    j = _dense_spin_jk(_eri(.83*factors) if backend == "ri-j-direct-k" else eri, density)[0]
    k = short_range*_dense_spin_jk(eri, density)[1] + (long_range-short_range)*_dense_spin_jk(eri_lr, density)[1]
    exc, vxc = gks._generalized_xc(density, ao, derivative, weights, cfg, "GGA")
    expected_energy = jnp.einsum("pq,qp->", density, h+.5*(j-k)).real+exc+.2
    expected_fock = h+j-k+vxc
    energy, _, fock = gks.generalized_energy_and_fock(density=density, hcore=h,
        nuclear_repulsion=.2, ao=ao, ao_deriv1=derivative, grid_weights=weights, config=cfg, **kwargs)
    np.testing.assert_allclose(energy, expected_energy, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(fock, expected_fock, atol=2e-11, rtol=2e-11)
    if backend == "ri-j-direct-k":
        assert calls and all(not with_j for with_j, _, _, _ in calls)


@pytest.mark.parametrize("method", ["GKS", "GHF"])
def test_generalized_facade_preserves_execution_configuration(method):
    from gradscf import gto, scf
    mf = getattr(scf, method)(gto.M(atom="H 0 0 0", basis="sto-3g", spin=1))
    full_key = mf._scf_signature()
    assert mf.density_fit("weigend") is mf
    cfg = mf._config()
    assert cfg.jk_backend == "df" and cfg.auxbasis == "weigend"
    df_key = mf._scf_signature()
    assert df_key != full_key
    assert mf.density_fit("weigend", only_dfj=True) is mf
    assert mf._config().only_dfj is True
    assert mf._scf_signature() not in (full_key, df_key)
    assert mf.direct_scf() is mf
    assert mf._config().jk_backend == "direct"
    assert mf._scf_signature() not in (full_key, df_key)
