"""Process-local model configuration; reuse the verified fixed-pool trainer."""
import argparse
import inspect
import json
from pathlib import Path
import time

OUT=Path(__file__).resolve().parent

def settings():
    from gradscf.model.nnao.runtime import scientific_source_hash,repository_root
    from gradscf.model.nnao.encoder import MACEBasisModel
    experiment=json.loads((OUT/'experiment.json').read_text())
    assert scientific_source_hash(repository_root())==experiment['source_sha256'],'Scientific source changed'
    parameters=inspect.signature(MACEBasisModel.__init__).parameters
    for key in ['channels','max_ell','num_interactions','correlation','cutoff','zero_init']:
        assert experiment['variants']['default'][key]==parameters[key].default,(key,parameters[key].default)
    return experiment

def preflight():
    import jax
    import numpy as np
    from gradscf.model.nnao.training.trainer import make_mace_predictor
    from gradscf.model.nnao.basis import prepare_direct_basis
    from gradscf.model.nnao.graph import build_graph
    from gradscf.model.nnao.initialization import bind_pool_candidate
    from gradscf.model.nnao.runtime import atomic_json,file_hash
    assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64
    experiment=settings();data=json.loads((OUT/'references.json').read_text())
    candidate=json.loads((OUT/'candidate.json').read_text())
    started=time.perf_counter();result={};predictions={}
    for name,config in experiment['variants'].items():
        vector,predict,_,mask=make_mace_predictor(experiment['seed'],config,return_mask=True)
        predictions[name]={}
        for row in data['structures']:
            graph=build_graph(row['atomic_numbers'],row['coords_angstrom'],element_order=config['elements'])
            with jax.default_device(jax.devices('cpu')[0]):
                layout=prepare_direct_basis(list(zip(row['symbols'],row['coords_angstrom'])),
                    basis_family=data['basis_family'],core_primitives=data['core_primitives'])
                layout=bind_pool_candidate(layout,candidate)
                expected=np.array(layout.reference_outputs(),copy=True)
                expected/=np.maximum(np.linalg.norm(expected,axis=-1,keepdims=True),1e-30)
            actual=np.asarray(predict(vector,graph))
            np.testing.assert_allclose(actual,expected,atol=1e-13,rtol=1e-13)
            predictions[name][row['structure_id']]=actual
        result[name]=dict(parameter_count=int(vector.size),trainable_parameter_count=int(mask.sum()),
            model_config=config,reference_initialization_passed=len(predictions[name]))
        del vector,predict,mask
        jax.clear_caches()
    for identifier in predictions['small']:
        np.testing.assert_array_equal(predictions['small'][identifier],predictions['default'][identifier])
    atomic_json(OUT/'preflight.json',dict(passed=True,variants=result,
        identical_initial_contractions=24,elapsed_seconds=time.perf_counter()-started,
        references_sha256=file_hash(OUT/'references.json'),candidate_sha256=file_hash(OUT/'candidate.json'),
        experiment_sha256=file_hash(OUT/'experiment.json'),device=jax.devices()[0].device_kind))
    print(json.dumps(result),flush=True)

def train(name):
    import jax
    import gradscf.model.nnao.config as config_module
    from gradscf.model.nnao.runtime import atomic_json,file_hash
    from gradscf.model.nnao.workflows.fixed_pool import run_fixed_pool
    assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64
    experiment=settings();gate=json.loads((OUT/'preflight.json').read_text())
    assert gate['passed'] and gate['experiment_sha256']==file_hash(OUT/'experiment.json')
    assert gate['references_sha256']==file_hash(OUT/'references.json')
    assert gate['candidate_sha256']==file_hash(OUT/'candidate.json')
    # This assignment affects only this fresh Python process, never repository files.
    config_module.MODEL_CONFIG=dict(experiment['variants'][name])
    output=OUT/name;output.mkdir(exist_ok=True)
    assert not (output/'last.npz').exists(),'Fresh comparison run requested; checkpoint already exists'
    atomic_json(output/'requested.json',dict(model_config=config_module.MODEL_CONFIG,
        seed=experiment['seed'],epochs=experiment['epochs'],launcher_sha256=file_hash(__file__)))
    result=run_fixed_pool(json.loads((OUT/'references.json').read_text()),
        json.loads((OUT/'candidate.json').read_text()),OUT/'cache',output,
        epochs=experiment['epochs'],learning_rate=experiment['learning_rate'],
        seed=experiment['seed'],storage=experiment['storage'])
    return int(result['status']!='complete')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['preflight','small','default'])
    args=parser.parse_args()
    if args.action=='preflight':preflight()
    else:raise SystemExit(train(args.action))
