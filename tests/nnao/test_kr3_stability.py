"""The real Kr3 orbital Hessian must resolve a missed iterative spectrum."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.initialization import bind_pool_candidate
from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool
from gradscf.model.nnao.training.initial_guess import atomic_guess_data
from gradscf.model.nnao.training.objective import _df_rhf_solver
from gradscf.model.nnao.training.pool import FreeContractionRHF


def test_kr3_dense_stability_rescue_preserves_implicit_gradient(tmp_path, monkeypatch):
    from gradscf.scf import stability
    fixture=json.loads((Path(__file__).with_name('data')/'kr3-clustered-curvature.json').read_text())
    row=fixture['structure']
    layout=prepare_direct_basis(list(zip(row['symbols'],row['coords_angstrom'])),
                                basis_family='nnao_dzp_s',core_primitives='6-31g')
    layout=bind_pool_candidate(layout,fixture['candidate'])
    problem,_=prepare_fixed_pool(row,layout,tmp_path/'integrals.npz',storage='mmap')
    coefficients=jnp.asarray(fixture['coefficients'])
    natural=problem._value_grad(coefficients,*problem.raw)
    real=stability.solve_hermitian
    def missed_iteration(operator,*,config):
        solved=real(operator,config=config)
        return solved if config.method=='dense' else solved._replace(converged=jnp.zeros_like(solved.converged))
    with monkeypatch.context() as patch:
        patch.setattr(stability,'solve_hermitian',missed_iteration)
        # Fresh functions isolate this injected miss from the shared JIT cache.
        value,gradient=_df_rhf_solver.__wrapped__(layout.nelectron,1e-8,.2)
        forced=FreeContractionRHF(layout,problem.raw,problem.enuc,value,gradient,
            row['structure_id'],problem.diagnostics,atomic_guess=atomic_guess_data(layout))
        (energy,info),derivative=forced._value_grad(coefficients,*problem.raw)
        jax.block_until_ready((energy,info,derivative))
    assert bool(info['converged']) and bool(info['stable']) and bool(info['stability_resolved'])
    assert bool(info['stability_dense_fallback'])
    assert float(info['stability_eigen_residual'])<=1e-7
    np.testing.assert_allclose(energy,fixture['provenance']['reference_energy_hartree'],atol=2e-9,rtol=0)
    np.testing.assert_allclose(info['minimum_curvature'],fixture['provenance']['dense_minimum_curvature'],atol=1e-7,rtol=0)
    assert np.isfinite(np.asarray(derivative)).all()
    np.testing.assert_allclose(energy,natural[0][0],atol=1e-9,rtol=0)
    np.testing.assert_allclose(derivative,natural[1],atol=2e-7,rtol=2e-6)
