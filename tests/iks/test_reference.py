"""SCF snapshot metadata must preserve representation and complete XC recipes."""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import dft, gto


def api():
    assert hasattr(dft, 'iks'), 'Missing public gradscf.dft.iks infrastructure'
    return dft.iks


def source(layout, xc):
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', unit='Bohr')
    cls = {'rks': dft.RKS, 'uks': dft.UKS, 'gks': dft.GKS}[layout]
    mf = cls(mol, xc=xc, max_memory=128.)
    mf.converged = True
    data = dict(overlap_matrix=jnp.eye(2), hcore_matrix=-jnp.eye(2))
    if layout == 'rks': data['density_matrix'] = jnp.eye(2)
    elif layout == 'uks': data.update(density_matrix_alpha=.5*jnp.eye(2), density_matrix_beta=.5*jnp.eye(2))
    else: data['density_matrix'] = .5*jnp.eye(4, dtype=complex)
    # This fixture tests snapshots only, not SCF convergence or XC evaluation.
    mf.scf_result = SimpleNamespace(**data)
    return mf


@pytest.mark.parametrize('layout', ['rks', 'uks', 'gks'])
@pytest.mark.parametrize('xc', ['pbe0', 'hse06'])
def test_reference_preserves_full_exchange_metadata_without_integral_build(layout, xc, monkeypatch):
    module = api()
    functional = dft.custom_xc(xc)
    mf = source(layout, functional)
    def forbidden(*a, **k): pytest.fail('A snapshot rebuilt SCF or complete ERIs')
    monkeypatch.setattr(mf, 'kernel', forbidden)
    monkeypatch.setattr(mf, '_build_reference', forbidden, raising=False)
    reference = module.Reference.from_source(mf)
    assert reference.layout == layout and reference.functional is functional
    assert reference.functional.recipe.exact_exchange == functional.recipe.exact_exchange
    assert reference.max_memory == 128.
    assert reference.target().valid
    if xc == 'hse06': assert reference.functional.recipe.exact_exchange.omega is not None


@pytest.mark.parametrize('layout', ['rks', 'uks', 'gks'])
def test_reference_keeps_custom_callback_and_live_parameter_identity(layout):
    module = api()
    functional = dft.custom_xc(input_fn=lambda x: x.density_matrix,
                              energy_fn=lambda p, d: p['a']*jnp.sum(jnp.abs(d)**2))
    mf = source(layout, functional)
    mf.params = {'a': jnp.asarray(.03)}
    ref = module.Reference.from_source(mf, max_memory=64.)
    assert ref.functional is functional and ref.params is mf.params
    assert ref.max_memory == 64.
    # A snapshot is not a promise that the current GKS facade executes custom XC.
    np.testing.assert_array_equal(ref.density, ref.target().density)


def test_reference_rejects_unsolved_or_invalid_target_source():
    module = api()
    mf = source('rks', 'pbe')
    mf.converged = False
    with pytest.raises(ValueError, match='converged'):
        module.Reference.from_source(mf)
    with pytest.raises(TypeError, match='RKS|UKS|GKS'):
        module.Reference.from_source(SimpleNamespace(converged=True))


def test_actual_gks_snapshot_keeps_spinor_overlap_and_detects_changed_xc():
    module = api()
    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g', unit='Bohr', cart=True)
    mf = dft.GKS(mol, xc='hf', integral_backend='jax', execution_device='cpu',
                 conv_tol=1e-11, conv_tol_grad=1e-9, max_cycle=80).run()
    assert mf.converged
    ref = module.Reference.from_source(mf)
    assert ref.overlap.shape == (2, 2) and ref.density.shape == (4, 4)
    np.testing.assert_allclose(ref.target().electron_numbers, [2.], atol=1e-10)
    mf.xc = 'pbe0'
    with pytest.raises(ValueError, match='changed|source'):
        module.Reference.from_source(mf)
