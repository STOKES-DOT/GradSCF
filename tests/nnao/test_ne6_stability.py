"""The accepted Ne6 basis must not fail on clustered positive curvature roots."""
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.initialization import bind_pool_candidate
from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool


def test_ne6_clustered_stability_and_implicit_coefficient_gradient(tmp_path):
    fixture = json.loads((Path(__file__).with_name('data') / 'ne6-clustered-curvature.json').read_text())
    row = fixture['structure']
    layout = prepare_direct_basis(list(zip(row['symbols'], row['coords_angstrom'])),
                                  basis_family='nnao_dzp_s', core_primitives='6-31g')
    layout = bind_pool_candidate(layout, fixture['candidate'])
    problem, _ = prepare_fixed_pool(row, layout, tmp_path/'integrals.npz', storage='mmap')
    # Keep the real fused forward/implicit-backward path, without MACE or FD.
    (energy, info), gradient = problem._value_grad(jnp.asarray(fixture['coefficients']), *problem.raw)
    assert bool(info['converged'])
    np.testing.assert_allclose(float(energy), fixture['provenance']['reference_energy_hartree'],
                               atol=2e-9, rtol=0)
    assert bool(info['stable']) and bool(info['stability_resolved'])
    assert float(info['stability_eigen_residual']) <= 1e-7
    np.testing.assert_allclose(float(info['minimum_curvature']),
                               fixture['provenance']['dense_minimum_curvature'], atol=1e-7, rtol=0)
    assert not bool(info['stability_dense_fallback'])
    assert np.isfinite(np.asarray(gradient)).all()
