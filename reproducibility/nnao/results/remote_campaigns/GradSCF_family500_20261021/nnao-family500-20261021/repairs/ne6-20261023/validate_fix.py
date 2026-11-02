"""Validate the exact failed basis through implicit SCF and the MACE pullback."""
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gradscf.data.molecule import atomic_number
from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.graph import build_graph
from gradscf.model.nnao.initialization import bind_pool_candidate
from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool
from gradscf.model.nnao.runtime import atomic_json, cache_location, scientific_source_hash, verify_source_imports
from gradscf.model.nnao.training.trainer import make_mace_predictor


ROOT = Path('/home/user_yuanjiao/private/homefile/GradSCF_family500_20261021/artifacts/nnao-family500-20261021')
HERE = ROOT/'repairs/ne6-20261023'


def main():
    deployment = json.loads((HERE/'deployment.json').read_text())
    assert scientific_source_hash(HERE/'source') == deployment['new_source_sha256']
    imports = verify_source_imports(HERE/'source')
    fixture = json.loads((HERE/'ne6-clustered-curvature.json').read_text())
    row = fixture['structure']
    with jax.default_device(jax.devices('cpu')[0]):
        layout = prepare_direct_basis(list(zip(row['symbols'],row['coords_angstrom'])),
                                      basis_family='nnao_dzp_s',core_primitives='6-31g')
        layout = bind_pool_candidate(layout,fixture['candidate'])
    problem, _ = prepare_fixed_pool(row,layout,cache_location(ROOT/'cache/s',row),storage='mmap')
    coefficients = jnp.asarray(fixture['coefficients'])
    energy, gradient, info = problem.evaluate_device(coefficients)
    np.testing.assert_allclose(energy,fixture['provenance']['reference_energy_hartree'],atol=2e-9,rtol=0)
    np.testing.assert_allclose(info['minimum_curvature'],fixture['provenance']['dense_minimum_curvature'],atol=1e-7,rtol=0)
    assert info['stable'] and info['stability_resolved'] and info['stability_eigen_residual']<=1e-7
    assert bool(jnp.all(jnp.isfinite(gradient)))
    config=dict(json.loads((ROOT/'inputs/model-config.json').read_text()),basis_family='nnao_dzp_s')
    _,predict,backward=make_mace_predictor(0,config)
    with np.load(ROOT/'runs/nnao_dzp_s/train/last.npz',allow_pickle=False) as saved:
        parameters=jnp.asarray(saved['parameters'])
    graph=build_graph([atomic_number(s) for s in row['symbols']],row['coords_angstrom'],
                      element_order=config['elements'],cutoff=config.get('cutoff',5.))
    np.testing.assert_allclose(predict(parameters,graph),coefficients,atol=1e-13,rtol=1e-13)
    parameter_gradient=backward(parameters,graph,gradient)
    assert bool(jnp.all(jnp.isfinite(parameter_gradient)))
    report=dict(status='passed',slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        device_kinds=[d.device_kind for d in jax.devices()],jax=jax.__version__,
        source_sha256=deployment['new_source_sha256'],imports=imports,
        energy_hartree=energy,scf=info,
        coefficient_gradient_norm=float(jnp.linalg.norm(gradient)),
        parameter_gradient_norm=float(jnp.linalg.norm(parameter_gradient)),
        full_mace_backward_finite=True,finite_differences=False)
    atomic_json(HERE/'validation.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':main()
