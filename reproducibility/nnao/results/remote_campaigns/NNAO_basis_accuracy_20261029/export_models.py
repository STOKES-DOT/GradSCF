"""CPU-only frozen-basis export, using each checkpoint's original source tree."""
import os
os.environ['JAX_PLATFORMS']='cpu'
os.environ['JAX_ENABLE_X64']='1'
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
HOME=Path('/home/user_yuanjiao/private/homefile')
ROOT=HOME/'GradSCF_family500_20261021'
CAMPAIGN=ROOT/'artifacts/nnao-family500-20261021'
FAMILIES=[f'nnao_{z}p_{s}' for z in ['sz','dz','tz'] for s in ['s','m','l']]
SOURCES={
 '1d249956d1e90296fe83aa697453c8b849d8a26bcceafbfc7bd01a909b52fee5':ROOT,
 '1b5cef38a4fb336b5f3c0bda987c1af534ad2592cc835ca798089173acb83da6':CAMPAIGN/'repairs/ne6-tzp-20261023/source',
 '7e5a22c4a17e3d727c03c121be4484e902080cef7b39b164783ce5fde23cdec2':CAMPAIGN/'repairs/kr3-20261024/source',
 'c28a091033269e2a50f935050ab481e2480fa5566ef2a45099a6e5a47703179c':CAMPAIGN/'repairs/c6cl6-20261026/source',
 'd86bd78ed807b3f5aae93bcf63988739311edc1858efe4c6c4b72b0a132a095b':HERE/'sources/lszp',
}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as s:
        for block in iter(lambda:s.read(1048576),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def export(family):
    started=time.perf_counter()
    checkpoint=(HOME/'GradSCF_mace500_20261014/artifacts/mace500-20261014/train/last.npz'
                if family=='nnao_szp_l' else CAMPAIGN/'runs'/family/'train/last.npz')
    checkpoint_hash=digest(checkpoint)
    with np.load(checkpoint,allow_pickle=False) as saved:
        metadata=json.loads(saved['metadata'].item());vector=np.array(saved['parameters'])
    identity=metadata['identity'];source=SOURCES[identity['scientific_source_sha256']]
    sys.path[:0]=[str(source/'src'),str(source/'src/gradscf/model/nnao/vendor/mace-jax')]
    import jax
    import jax.numpy as jnp
    from gradscf.model.nnao.runtime import scientific_source_hash
    from gradscf.model.nnao.training.trainer import make_mace_predictor
    from gradscf.model.nnao.basis import prepare_direct_basis
    from gradscf.model.nnao.initialization import bind_pool_candidate
    from gradscf.model.nnao.integrals.pool import active_pool_basis
    from gradscf.model.nnao.graph import build_graph
    from protocols import basis_counts
    assert scientific_source_hash(source)==identity['scientific_source_sha256']
    assert identity['model_config']['basis_family']==family
    kwargs=dict(basis_family=family,core_primitives=identity['references']['core_primitives'],cart=False)
    # Validate the basis pool and fixed core against actual training cache metadata.
    training=identity['references']['structures'][0]
    layout=bind_pool_candidate(prepare_direct_basis(list(zip(training['symbols'],training['coords_angstrom'])),**kwargs),identity['candidate'])
    _,p=active_pool_basis(layout)
    expected=identity['caches'][training['structure_id']]['identity']['pool']
    for actual,wanted in zip(p.exponents,expected['exponents']):np.testing.assert_allclose(actual,wanted,atol=1e-12,rtol=0)
    for actual,wanted in zip(p.coefficients,expected['coefficients']):np.testing.assert_allclose(actual,wanted,atol=1e-12,rtol=0)
    initial,predict,_=make_mace_predictor(identity['seed'],identity['model_config'])
    assert vector.shape==initial.shape and len(vector)==identity['parameter_count']
    rows=json.loads((HERE/'inputs/structures.json').read_text());records=[]
    for i,row in enumerate(rows):
        layout=bind_pool_candidate(prepare_direct_basis(list(zip(row['symbols'],row['coords_angstrom'])),**kwargs),identity['candidate'])
        graph=build_graph(row['atomic_numbers'],row['coords_angstrom'],element_order=identity['model_config']['elements'],
            cutoff=identity['model_config']['cutoff'],total_charge=0,spin=0)
        tick=time.perf_counter();coefficients=predict(jnp.asarray(vector),graph);jax.block_until_ready(coefficients)
        prediction_seconds=time.perf_counter()-tick
        shells=layout.atom_shells(layout.bind(coefficients))
        count=basis_counts(shells);assert count['nao_spherical']==layout.topology.nao
        records.append(dict(structure_id=row['structure_id'],atom_shells=shells,counts=count,prediction_seconds=prediction_seconds))
        print('EXPORT',family,i+1,len(rows),row['structure_id'],count['nao_spherical'],flush=True)
    assert digest(checkpoint)==checkpoint_hash
    write(HERE/'frozen'/f'{family}.json',dict(basis_key=family,kind='nnao',checkpoint=str(checkpoint),
        checkpoint_sha256=checkpoint_hash,parameter_sha256=hashlib.sha256(vector.tobytes()).hexdigest(),
        epoch=metadata['epoch'],update_step=metadata['update_step'],source=str(source),
        scientific_source_sha256=identity['scientific_source_sha256'],model_config=identity['model_config'],
        core_primitives=kwargs['core_primitives'],pool_and_core_checked_against_training=True,
        jax=jax.__version__,backend=jax.default_backend(),seconds=time.perf_counter()-started,records=records))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--family');args=parser.parse_args()
    if args.family:
        export(args.family);return
    from protocols import basis_counts
    from pyscf import gto
    settings=json.loads((HERE/'inputs/settings.json').read_text())
    rows=json.loads((HERE/'inputs/structures.json').read_text())
    for index,basis in enumerate(settings['classical_bases']+[settings['reference_basis']]):
        key=f't{index:02d}' if index<18 else 'reference'
        records=[]
        for row in rows:
            shells=[gto.basis.load(basis,s) for s in row['symbols']]
            records.append(dict(structure_id=row['structure_id'],atom_shells=shells,counts=basis_counts(shells)))
        write(HERE/'frozen'/f'{key}.json',dict(basis_key=key,kind='classical',name=basis,records=records))
    for family in FAMILIES:
        if (HERE/'frozen'/f'{family}.json').exists():continue
        subprocess.run([sys.executable,'-u',str(Path(__file__)), '--family',family],check=True)
    files=list((HERE/'frozen').glob('*.json'))
    assert len(files)==28
    write(HERE/'export-gate.json',dict(status='passed',basis_files=28,physical_structures=30,
        checkpoint_policy='final last.npz; old L-SZP update 5969; not historical update 3351',
        files={f.name:digest(f) for f in files}))


if __name__=='__main__':main()
