"""Exact accepted-update-3 clean-process memory probe; no parameter changes."""
import json
import os
from pathlib import Path
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from gradscf.data.molecule import atomic_number
from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.graph import build_graph
from gradscf.model.nnao.initialization import bind_pool_candidate
from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool
from gradscf.model.nnao.runtime import atomic_json,cache_location,file_hash,scientific_source_hash
from gradscf.model.nnao.training.objective import BasisRHF
from gradscf.model.nnao.training.trainer import make_mace_predictor

ROOT=Path('/home/user_yuanjiao/private/homefile/GradSCF_family500_20261021/artifacts/nnao-family500-20261021')
HERE=ROOT/'repairs/bn-corannulene-20261027'
SOURCE=ROOT/'repairs/c6cl6-20261026/source'


def main(output_name='probe.json'):
    start=time.perf_counter()
    assert scientific_source_hash(SOURCE)=='c28a091033269e2a50f935050ab481e2480fa5566ef2a45099a6e5a47703179c'
    device=jax.devices('gpu')[0]
    checkpoint=ROOT/'runs/nnao_tzp_l/train/last.npz'
    checkpoint_hash=file_hash(checkpoint)
    with np.load(checkpoint,allow_pickle=False) as saved:
        metadata=json.loads(saved['metadata'].item())
        assert metadata['update_step']==3 and metadata['batch_cursor']==3
        parameters=jnp.asarray(saved['parameters'])
    row=next(r for r in metadata['identity']['references']['structures']
             if r['structure_id']=='GMTKN55-v2/INV24/BN_Corannulene')
    report=dict(status='preparing',job_id=os.environ.get('SLURM_JOB_ID'),jax=jax.__version__,
        device=device.device_kind,checkpoint_update=3,checkpoint_sha256=checkpoint_hash,
        structure_id=row['structure_id'],source_sha256=metadata['identity']['scientific_source_sha256'],
        finite_differences=False,optimizer_reset=False)
    output=HERE/output_name
    def save(phase):
        report.update(phase=phase,elapsed_seconds=time.perf_counter()-start,
                      device_memory=device.memory_stats())
        atomic_json(output,report);print('PROBE',json.dumps(report),flush=True)
    try:
        with jax.default_device(jax.devices('cpu')[0]):
            layout=prepare_direct_basis(list(zip(row['symbols'],row['coords_angstrom'])),
                basis_family='nnao_tzp_l',core_primitives='6-31g')
            layout=bind_pool_candidate(layout,metadata['identity']['candidate'])
        problem,manifest=prepare_fixed_pool(row,layout,cache_location(ROOT/'cache/l',row),storage='mmap')
        report.update(nao=layout.topology.nao,df_shape=manifest['df_shape'],cache_bytes=manifest['bytes'])
        _,predict,backward=make_mace_predictor(0,metadata['identity']['model_config'])
        graph=build_graph([atomic_number(s) for s in row['symbols']],row['coords_angstrom'],
                         element_order=metadata['identity']['model_config']['elements'],cutoff=5.)
        coefficients=jax.block_until_ready(predict(parameters,graph));save('predicted')
        lowered=problem._value_grad.lower(coefficients,*problem.raw)
        compiled=lowered.compile()
        memory=compiled.memory_analysis()
        report['compiled_memory']={name:getattr(memory,name,None) for name in (
            'argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes',
            'alias_size_in_bytes','generated_code_size_in_bytes')}
        save('compiled')
        (energy,info),gradient=jax.block_until_ready(compiled(coefficients,*problem.raw))
        details=BasisRHF._checked_info(float(energy),info)
        assert bool(jnp.all(jnp.isfinite(gradient)))
        report.update(energy_hartree=float(energy),scf=details,
                      coefficient_gradient_norm=float(jnp.linalg.norm(gradient)))
        save('scf_and_implicit_backward_passed')
        parameter_gradient=jax.block_until_ready(backward(parameters,graph,gradient))
        assert bool(jnp.all(jnp.isfinite(parameter_gradient)))
        report.update(status='passed',full_mace_backward_finite=True,
                      parameter_gradient_norm=float(jnp.linalg.norm(parameter_gradient)))
        save('complete')
    except Exception as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc())
        save('failed');raise
    finally:
        assert file_hash(checkpoint)==checkpoint_hash,'The probe must never change the accepted checkpoint.'


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-name',default='probe.json')
    main(parser.parse_args().output_name)
