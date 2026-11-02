"""One exact failed-basis SCF/implicit contraction/MACE backward; no FD."""
import argparse
import json
import os
from pathlib import Path
import time

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


ROOT=Path('/home/user_yuanjiao/private/homefile/GradSCF_family500_20261021/artifacts/nnao-family500-20261021')
HERE=ROOT/'repairs/c6cl6-20261026'


def main(output_name='validation.json'):
    start=time.perf_counter()
    deployment=json.loads((HERE/'deployment.json').read_text())
    assert scientific_source_hash(HERE/'source')==deployment['new_source_sha256']
    imports=verify_source_imports(HERE/'source')
    row=next(r for r in json.loads((ROOT/'inputs/references.json').read_text())['structures']
             if r['structure_id']=='GMTKN55-v2/DC13/C6Cl6')
    with jax.default_device(jax.devices('cpu')[0]):
        layout=prepare_direct_basis(list(zip(row['symbols'],row['coords_angstrom'])),
                                   basis_family='nnao_tzp_l',core_primitives='6-31g')
        layout=bind_pool_candidate(layout,json.loads((ROOT/'inputs/l-candidate.json').read_text()))
    assert layout.topology.nao==240
    problem,_=prepare_fixed_pool(row,layout,cache_location(ROOT/'cache/l',row),storage='mmap')
    config=dict(json.loads((ROOT/'inputs/model-config.json').read_text()),basis_family='nnao_tzp_l')
    initial,predict,backward=make_mace_predictor(0,config)
    with np.load(HERE/'before-restart/failed-update-000001.npz',allow_pickle=False) as saved:
        parameters=jnp.asarray(saved['parameters'])
    np.testing.assert_allclose(parameters,initial,atol=5e-14,rtol=5e-14)
    graph=build_graph([atomic_number(s) for s in row['symbols']],row['coords_angstrom'],
                      element_order=config['elements'],cutoff=config.get('cutoff',5.))
    coefficients=predict(parameters,graph)
    energy,gradient,info=problem.evaluate_device(coefficients)
    np.testing.assert_allclose(energy,-2983.735668098414,atol=2e-8,rtol=0)
    np.testing.assert_allclose(info['min_overlap_eigenvalue'],3.6384739159885476e-7,atol=2e-11,rtol=0)
    assert info['stable'] and info['converged'] and info['stability_resolved']
    assert info['stability_eigen_residual']<=1e-7
    parameter_gradient=backward(parameters,graph,gradient)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert bool(jnp.all(jnp.isfinite(parameter_gradient)))
    assert float(jnp.linalg.norm(parameter_gradient))>0
    report=dict(status='passed',slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        source_sha256=deployment['new_source_sha256'],imports=imports,jax=jax.__version__,
        device_kinds=[d.device_kind for d in jax.devices()],nao=240,nelectron=138,
        energy_hartree=energy,scf=info,finite_differences=False,
        coefficient_gradient_norm=float(jnp.linalg.norm(gradient)),
        parameter_gradient_norm=float(jnp.linalg.norm(parameter_gradient)),
        full_mace_backward_finite=True,elapsed_seconds=time.perf_counter()-start)
    atomic_json(HERE/output_name,report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-name',default='validation.json')
    main(parser.parse_args().output_name)
