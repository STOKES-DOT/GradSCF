from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import dft, gto, scf
from gradscf.scf.inputs import build_rks_integral_inputs


ATOM = "Li 0 0 0; H 0 0 1.6"
BASIS = "6-31g*"


def test_molecular_ao_representation_defaults_to_spherical():
    assert gto.M(atom=ATOM, basis=BASIS).cart is False
    assert gto.M(atom=ATOM, basis=BASIS, cart=True).cart is True


def _pyscf_molecule(*, charge=0, spin=0, cart=False):
    pytest.importorskip("pyscf")
    from pyscf import gto as pgto

    return pgto.M(atom=ATOM, basis=BASIS, cart=cart, charge=charge, spin=spin, verbose=0)


@pytest.mark.parametrize("block_size", [-1, 32], ids=["dense", "streamed"])
def test_spherical_rks_input_ao_and_derivatives_match_pyscf(block_size):
    reference = _pyscf_molecule()
    from pyscf import dft as pdft

    assert _pyscf_molecule(cart=True).nao > reference.nao
    inputs = build_rks_integral_inputs(
        atom=ATOM, basis=BASIS, cart=False, grids_level=0, include_dipole_integrals=False,
        config=scf.RKSConfig(xc_spec="pbe", jk_backend="direct", grid_block_size=block_size),
    )

    assert inputs.overlap.shape == inputs.hcore.shape == (reference.nao, reference.nao)
    assert inputs.direct_basis.nao == reference.nao
    if block_size == -1:
        assert inputs.grid_ao is None
        derivatives = inputs.ao_deriv1
        np.testing.assert_array_equal(inputs.ao, derivatives[0])
    else:
        assert inputs.grid_ao.nao == reference.nao
        derivatives = jnp.moveaxis(
            inputs.grid_ao.map(
                lambda ao, derivatives, laplacian, coords, weights: jnp.moveaxis(derivatives, 0, 1),
                deriv=1,
            ),
            1, 0,
        )
    expected = pdft.numint.eval_ao(reference, np.asarray(inputs.coords), deriv=1)
    np.testing.assert_allclose(np.asarray(derivatives), expected, atol=2e-10, rtol=0)


def test_streamed_spherical_ao_preserves_geometry_jvp_and_vjp():
    mol = gto.M(atom=ATOM, basis=BASIS, cart=False)
    spec = mol.to_spec()
    config = scf.RKSConfig(xc_spec="pbe", jk_backend="direct", grid_block_size=32)
    transform = jnp.asarray(_pyscf_molecule().cart2sph_coeff())

    def block(coords, cart):
        inputs = build_rks_integral_inputs(
            atom=replace(spec, coords_bohr=coords), basis=BASIS, cart=cart,
            config=config, include_dipole_integrals=False,
        )
        values = inputs.grid_ao.block(0, 7, deriv=1)[1]
        return values @ transform if cart else values

    coords = spec.coords_bohr
    direction = jnp.asarray([[.1, -.2, .3], [.1, .4, .2]])
    for fn, ref in zip(
        jax.jvp(lambda x: block(x, False), (coords,), (direction,)),
        jax.jvp(lambda x: block(x, True), (coords,), (direction,)),
        strict=True,
    ):
        np.testing.assert_allclose(fn, ref, atol=1e-11, rtol=0)
    values, pull = jax.vjp(lambda x: block(x, False), coords)
    _, reference_pull = jax.vjp(lambda x: block(x, True), coords)
    cotangent = jnp.linspace(-.1, .2, values.size).reshape(values.shape)
    np.testing.assert_allclose(pull(cotangent)[0], reference_pull(cotangent)[0],
                               atol=1e-11, rtol=0)


@pytest.mark.parametrize(
    ("method", "charge", "spin", "block_size"),
    [("RKS", 0, 0, -1), ("UKS", 1, 1, 32), ("GKS", 0, 0, 32), ("ROKS", 1, 1, -1)],
    ids=["rks", "uks", "gks", "roks"],
)
def test_spherical_pbe_facade_matches_pyscf_on_shared_grid(method, charge, spin, block_size):
    pytest.importorskip("jax_xc")
    mol = _pyscf_molecule(charge=charge, spin=spin)
    from pyscf import dft as pdft

    options = {"jk_backend": "direct"} if method != "ROKS" else {}
    mf = getattr(dft, method)(
        gto.M(atom=ATOM, basis=BASIS, charge=charge, spin=spin),
        xc="pbe", grids_level=0, grid_block_size=block_size, max_cycle=100,
        init_guess="hcore", conv_tol=1e-10, conv_tol_density=1e-8,
        conv_tol_grad=1e-7, **options,
    ).run()
    assert mf.converged
    inputs = mf._scf_inputs
    nao = mol.nao
    assert inputs.overlap.shape == (nao, nao)
    if block_size > 0:
        assert inputs.grid_ao.nao == nao
    else:
        assert inputs.ao.shape[-1] == nao
    density = np.asarray(mf.make_rdm1())
    shape = {"RKS": (nao, nao), "UKS": (2, nao, nao), "GKS": (2 * nao, 2 * nao),
             "ROKS": (2, nao, nao)}[method]
    assert density.shape == shape

    reference = getattr(pdft, "RKS" if method == "GKS" else method)(mol, xc="pbe")
    reference.grids.coords = np.asarray(inputs.coords)
    reference.grids.weights = np.asarray(inputs.grid_weights)
    reference.small_rho_cutoff = 0.0
    reference.init_guess = "1e"
    reference.conv_tol = 1e-12
    reference.conv_tol_grad = 1e-9
    reference.max_cycle = 100
    reference.kernel()
    assert reference.converged
    np.testing.assert_allclose(mf.e_tot, reference.e_tot, atol=5e-7, rtol=0)
