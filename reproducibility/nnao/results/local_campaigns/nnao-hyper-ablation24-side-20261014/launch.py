"""Three controlled ablations in a source-isolated checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import time

OUT=Path(__file__).resolve().parent

def settings():
    from gradscf.model.nnao.runtime import file_hash,scientific_source_hash,repository_root
    d=json.loads((OUT/'experiment.json').read_text())
    assert scientific_source_hash(repository_root())==d['source_sha256']
    old=Path(d['baseline_directory'])
    assert json.loads((old/'comparison.json').read_text())['complete']
    for name in ['references.json','candidate.json']:
        assert file_hash(OUT/name)==file_hash(old/name)
    return d

def preflight(name):
    import jax
    import numpy as np
    from flax import nnx
    from gradscf.model.nnao.encoder import MACEBasisModel
    from gradscf.model.nnao.training.trainer import make_mace_predictor
    from gradscf.model.nnao.basis import prepare_direct_basis
    from gradscf.model.nnao.initialization import bind_pool_candidate
    from gradscf.model.nnao.graph import build_graph
    from gradscf.model.nnao.runtime import atomic_json,file_hash
    assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64
    d=settings();variant=d['variants'][name];config=variant['model_config']
    root=OUT/name;root.mkdir(exist_ok=True)
    start=time.perf_counter()
    baseline=json.loads((Path(d['baseline_directory'])/'wide/metadata.json').read_text())['identity']
    # The unchanged default radial network must reproduce the original seeded parameter vector exactly.
    initial,_,_,mask=make_mace_predictor(d['seed'],d['baseline_model'],return_mask=True)
    assert int(initial.size)==baseline['parameter_count']==222755
    assert hashlib.sha256(np.asarray(initial).tobytes()).hexdigest()==baseline['initialization_sha256']
    del initial,mask
    model=MACEBasisModel(**config,rngs=nnx.Rngs(d['seed']))
    assert tuple(model.backbone.radial_MLP)==tuple(config['radial_MLP'])
    del model
    vector,predict,_,mask=make_mace_predictor(d['seed'],config,return_mask=True)
    if name=='lr3e4':
        assert hashlib.sha256(np.asarray(vector).tobytes()).hexdigest()==baseline['initialization_sha256']
    else:
        assert vector.size>baseline['parameter_count']
    data=json.loads((OUT/'references.json').read_text())
    candidate=json.loads((OUT/'candidate.json').read_text())
    for row in data['structures']:
        with jax.default_device(jax.devices('cpu')[0]):
            layout=prepare_direct_basis(list(zip(row['symbols'],row['coords_angstrom'])),
                basis_family=data['basis_family'],core_primitives=data['core_primitives'])
            layout=bind_pool_candidate(layout,candidate)
            expected=np.array(layout.reference_outputs(),copy=True)
            expected/=np.maximum(np.linalg.norm(expected,axis=-1,keepdims=True),1e-30)
        graph=build_graph(row['atomic_numbers'],row['coords_angstrom'],element_order=config['elements'])
        np.testing.assert_allclose(predict(vector,graph),expected,atol=1e-13,rtol=1e-13)
    result=dict(passed=True,variant=name,parameter_count=int(vector.size),
        trainable_parameter_count=int(mask.sum()),baseline_parameter_bytes_preserved=True,
        reference_initialization_passed=24,model_config=config,learning_rate=variant['learning_rate'],
        references_sha256=file_hash(OUT/'references.json'),candidate_sha256=file_hash(OUT/'candidate.json'),
        experiment_sha256=file_hash(OUT/'experiment.json'),launcher_sha256=file_hash(__file__),
        device=jax.devices()[0].device_kind,elapsed_seconds=time.perf_counter()-start)
    atomic_json(root/'preflight.json',result);print(json.dumps(result),flush=True)

def train(name):
    import jax
    import gradscf.model.nnao.config as configuration
    from gradscf.model.nnao.runtime import file_hash,atomic_json
    from gradscf.model.nnao.workflows.fixed_pool import run_fixed_pool
    assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64
    d=settings();variant=d['variants'][name];root=OUT/name
    gate=json.loads((root/'preflight.json').read_text())
    assert gate['passed']
    assert gate['experiment_sha256']==file_hash(OUT/'experiment.json')
    assert gate['launcher_sha256']==file_hash(__file__)
    configuration.MODEL_CONFIG=dict(variant['model_config'])
    output=root/'training';output.mkdir(exist_ok=True)
    assert not (output/'last.npz').exists(),'Fresh experiment only'
    atomic_json(root/'requested.json',dict(variant,epochs=d['epochs'],seed=d['seed'],
        source_sha256=d['source_sha256'],launcher_sha256=file_hash(__file__)))
    result=run_fixed_pool(json.loads((OUT/'references.json').read_text()),
        json.loads((OUT/'candidate.json').read_text()),OUT/'cache',output,
        epochs=d['epochs'],seed=d['seed'],learning_rate=variant['learning_rate'],storage=d['storage'])
    return int(result['status']!='complete')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['preflight','train'])
    parser.add_argument('variant',choices=['lr3e4','correlation3','radial64'])
    args=parser.parse_args()
    if args.action=='preflight':preflight(args.variant)
    else:raise SystemExit(train(args.variant))
