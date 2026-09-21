import os


os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")


import pytest


@pytest.fixture(scope="module")
def radical():
    import jax.numpy as jnp
    pyscf = pytest.importorskip("pyscf")
    from pyscf import ao2mo
    mol = pyscf.gto.M(atom="H 0 0 0; H 0 0 .85; H 0 0 1.9",
                      basis="sto-3g", spin=1, verbose=0)
    mf = mol.UHF().run(conv_tol=1e-13)
    ca, cb = mf.mo_coeff
    h = (ca.T @ mf.get_hcore() @ ca, cb.T @ mf.get_hcore() @ cb)
    g = tuple(ao2mo.general(mol, cs, compact=False).reshape((3,) * 4)
              for cs in ((ca, ca, ca, ca), (ca, ca, cb, cb), (cb, cb, cb, cb)))
    return mf, tuple(map(jnp.asarray, h)), tuple(map(jnp.asarray, g))
