"""Facade reuse of static CSF topology across molecular calculations."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from time import sleep
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import _memory, gto, scf
from gradscf.mc import facade


@pytest.fixture(autouse=True)
def clear_space_cache():
    cache = getattr(facade, "_cached_csf_space", None)
    if cache is not None:
        cache.cache_clear()
    yield
    if cache is not None:
        cache.cache_clear()


def test_casscf_reuses_topology_across_distinct_geometries(monkeypatch):
    def solved_hf(distance, coefficients, enuc):
        mf = scf.RHF(gto.M(atom=f"H 0 0 0; H 0 0 {distance}", basis="sto-3g"))
        monkeypatch.setattr(mf, "_check_reference_source", lambda: None)
        mf.mo_coeff = jnp.asarray(coefficients)
        mf._scf_inputs = SimpleNamespace(
            overlap=jnp.eye(2), hcore=jnp.diag(jnp.array([-1., 0.])),
            nuclear_repulsion=enuc, eri=jnp.zeros((2,) * 4),
        )
        return mf

    first = facade.CASSCF(solved_hf(.74, np.eye(2), .5), 2, (1, 1), spin2=0, solver="dense").run()
    rotated = np.array([[.8, -.6], [.6, .8]])
    second = facade.CASSCF(solved_hf(1., rotated, .4), 2, (1, 1), spin2=np.int64(0), solver="dense").run()

    assert first.result.space is second.result.space
    assert first.result is not second.result
    np.testing.assert_allclose(first.e_tot, -1.5, atol=1e-12, rtol=0.)
    np.testing.assert_allclose(second.e_tot, -1.6, atol=1e-12, rtol=0.)


def test_csf_cache_preserves_validation_and_memory_budget(monkeypatch):
    space = facade._facade_csf_space(2, (1, 1), 0, max_memory=40.)
    with pytest.raises(ValueError, match="spin2"):
        facade._facade_csf_space(2, (1, 1), False, max_memory=40.)
    with pytest.raises(ValueError, match="memory budget"):
        facade._facade_csf_space(2, (1, 1), 0, max_memory=0.0001)
    with monkeypatch.context() as scope:
        scope.setattr(_memory, "available_memory_bytes", lambda device=None: 4)
        with pytest.raises(ValueError, match="memory budget"):
            facade._facade_csf_space(2, (1, 1), 0, max_memory=40.)
    assert facade._facade_csf_space(2, (1, 1), 0, max_memory=40.) is space


def test_csf_cache_has_bounded_array_payload(monkeypatch):
    def large_topology(*args, **kwargs):
        return SimpleNamespace(storage_bytes=40_000_000)

    monkeypatch.setattr(facade, "make_csf_space", large_topology)
    first = facade._facade_csf_space(2, (1, 1), 0, max_memory=1000.)
    facade._facade_csf_space(3, (1, 1), 0, max_memory=1000.)
    assert facade._cached_csf_space.cache_bytes() == 40_000_000
    assert facade._facade_csf_space(2, (1, 1), 0, max_memory=1000.) is not first


def test_csf_cache_coalesces_concurrent_cold_calls(monkeypatch):
    callers = Barrier(3)
    builds = []

    def build(*args, **kwargs):
        builds.append(1)
        sleep(.02)
        return SimpleNamespace(storage_bytes=1)

    def use_cache():
        callers.wait(timeout=5.)
        return facade._facade_csf_space(2, (1, 1), 0, max_memory=40.)

    monkeypatch.setattr(facade, "make_csf_space", build)
    with ThreadPoolExecutor(max_workers=3) as executor:
        pending = [executor.submit(use_cache) for _ in range(3)]
        spaces = [result.result(timeout=5.) for result in pending]
    assert builds == [1]
    assert all(space is spaces[0] for space in spaces)
