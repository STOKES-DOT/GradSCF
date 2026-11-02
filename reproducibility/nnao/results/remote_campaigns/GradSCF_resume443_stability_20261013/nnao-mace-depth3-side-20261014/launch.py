"""Depth-only pilot; preserve the fixed-pool SCF and training implementation."""
import argparse
import importlib.util
import inspect
import json
from pathlib import Path
import time

OUT=Path(__file__).resolve().parent
BASE=OUT.parent/'nnao-mace-width64-side-20261014'
TRAINER=OUT.parent/'nnao-mace-capacity24-side-20261014'

def settings():
    from gradscf.model.nnao.runtime import file_hash,scientific_source_hash,repository_root
    from gradscf.model.nnao.encoder import MACEBasisModel
    experiment=json.loads((OUT/'experiment.json').read_text())
    assert scientific_source_hash(repository_root())==experiment['source_sha256']
    assert file_hash(TRAINER/'launch.py')==experiment['baseline_launcher_sha256']
    old=json.loads((BASE/'experiment.json').read_text())
    current=experiment['variants']['deep'];baseline=old['variants']['wide']
    assert current==dict(baseline,num_interactions=3) and baseline['num_interactions']==2
    for key in ['epochs','seed','learning_rate','storage','source_sha256']:
        assert experiment[key]==old[key]
    for name in ['references.json','candidate.json']:
        assert file_hash(OUT/name)==file_hash(BASE/name)
    signature=inspect.signature(MACEBasisModel.__init__).parameters
    for key in ['max_ell','correlation','cutoff','zero_init']:
        assert current[key]==signature[key].default
    assert json.loads((BASE/'comparison.json').read_text())['complete']
    return experiment

def preflight():
    import jax
    import numpy as np
    from gradscf.model.nnao.training.trainer import make_mace_predictor
    from gradscf.model.nnao.graph import build_graph
    from gradscf.model.nnao.basis import prepare_direct_basis
    from gradscf.model.nnao.initialization import bind_pool_candidate
    from gradscf.model.nnao.runtime import atomic_json,file_hash
    assert jax.default_backend()=='gpu' and jax.config.jax_enable_x64
    experiment=settings();config=experiment['variants']['deep']
    previous=json.loads((BASE/'preflight.json').read_text())
    assert previous['passed'] and previous['reference_initialization_passed']==24
    for name,key in [('references.json','references_sha256'),('candidate.json','candidate_sha256')]:
        assert previous[key]==file_hash(OUT/name)
    data=json.loads((OUT/'references.json').read_text());candidate=json.loads((OUT/'candidate.json').read_text())
    started=time.perf_counter()
    vector,predict,_,mask=make_mace_predictor(experiment['seed'],config,return_mask=True)
    for row in data['structures']:
        with jax.default_device(jax.devices('cpu')[0]):
            layout=prepare_direct_basis(list(zip(row['symbols'],row['coords_angstrom'])),
                basis_family=data['basis_family'],core_primitives=data['core_primitives'])
            layout=bind_pool_candidate(layout,candidate)
            expected=np.array(layout.reference_outputs(),copy=True)
            expected/=np.maximum(np.linalg.norm(expected,axis=-1,keepdims=True),1e-30)
        graph=build_graph(row['atomic_numbers'],row['coords_angstrom'],element_order=config['elements'])
        np.testing.assert_allclose(predict(vector,graph),expected,atol=1e-13,rtol=1e-13)
    result=dict(passed=True,parameter_count=int(vector.size),trainable_parameter_count=int(mask.sum()),
        model_config=config,reference_initialization_passed=24,baseline_reference_initialization_passed=24,
        references_sha256=file_hash(OUT/'references.json'),candidate_sha256=file_hash(OUT/'candidate.json'),
        experiment_sha256=file_hash(OUT/'experiment.json'),elapsed_seconds=time.perf_counter()-started,
        device=jax.devices()[0].device_kind)
    atomic_json(OUT/'preflight.json',result);print(json.dumps(result),flush=True)

def train():
    settings()
    spec=importlib.util.spec_from_file_location('matched_capacity_launcher',TRAINER/'launch.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.OUT=OUT;module.settings=settings
    return module.train('deep')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['preflight','train'])
    args=parser.parse_args()
    if args.action=='preflight':preflight()
    else:raise SystemExit(train())
