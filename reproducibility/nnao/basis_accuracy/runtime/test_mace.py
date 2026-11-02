"""Check pinned MACE inference against the already frozen final-model basis."""
from pathlib import Path
import json
import os
import time
import numpy as np

import jax
import jax.numpy as jnp
from gradscf.model.nnao.training.trainer import make_mace_predictor
from gradscf.model.nnao.graph import build_graph
from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.initialization import bind_pool_candidate

ROOT=Path(__file__).resolve().parent.parent


def main():
    start=time.perf_counter();backend=jax.default_backend()
    requested='gpu' if os.environ.get('JAX_PLATFORMS')=='cuda' else 'cpu'
    assert backend==requested,(backend,requested)
    basis=json.loads((ROOT/'frozen/nnao_szp_s.json').read_text())
    with np.load(basis['checkpoint'],allow_pickle=False) as saved:
        meta=json.loads(saved['metadata'].item());vector=np.array(saved['parameters'])
    identity=meta['identity'];rows=json.loads((ROOT/'inputs/structures.json').read_text())
    row=next(r for r in rows if r['structure_id']=='S22/02/A')
    expected=next(r for r in basis['records'] if r['structure_id']==row['structure_id'])
    initial,predict,_=make_mace_predictor(identity['seed'],identity['model_config'])
    assert initial.shape==vector.shape
    graph=build_graph(row['atomic_numbers'],row['coords_angstrom'],
        element_order=identity['model_config']['elements'],cutoff=identity['model_config']['cutoff'])
    prediction=predict(jnp.asarray(vector),graph);jax.block_until_ready(prediction)
    layout=bind_pool_candidate(prepare_direct_basis(list(zip(row['symbols'],row['coords_angstrom'])),
        basis_family='nnao_szp_s',core_primitives=identity['references']['core_primitives'],cart=False),identity['candidate'])
    actual=layout.atom_shells(layout.bind(prediction));error=0.
    assert len(actual)==len(expected['atom_shells'])
    assert np.isfinite(np.asarray(prediction)).all()
    for a,b in zip(actual,expected['atom_shells']):
        assert len(a)==len(b)
        for sa,sb in zip(a,b):
            assert sa[0]==sb[0]
            aa,bb=np.asarray(sa[1:]),np.asarray(sb[1:])
            assert aa.shape==bb.shape and np.isfinite(aa).all() and np.isfinite(bb).all()
            error=max(error,float(np.max(abs(aa-bb))))
    assert error<1e-10,error
    import mace_jax,e3nn_jax
    report=dict(status='passed',backend=backend,device=str(jax.devices()[0]),jax=jax.__version__,
        mace_source=mace_jax.__file__,e3nn_jax_source=e3nn_jax.__file__,checkpoint=basis['checkpoint'],
        update_step=basis['update_step'],max_basis_difference=error,tolerance=1e-10,
        output_shape=list(prediction.shape),elapsed_seconds=time.perf_counter()-start)
    report['loaded_cuda_libraries']=sorted({line.split()[-1]
        for line in Path('/proc/self/maps').read_text().splitlines()
        if any(name in line for name in ['libcublas.so','libcusolver.so','libcudart.so','libcudnn.so'])})
    (ROOT/'runtime'/f'mace-validation-{backend}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
