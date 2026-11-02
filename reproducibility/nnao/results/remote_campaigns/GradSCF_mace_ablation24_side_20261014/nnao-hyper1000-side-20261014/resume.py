"""Continue each accepted epoch-200 state using its original scientific source."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

OUT=Path(__file__).resolve().parent

def main(name):
    import jax
    import gradscf.model.nnao.config as config
    from gradscf.model.nnao.runtime import atomic_json,file_hash,repository_root,scientific_source_hash
    from gradscf.model.nnao.workflows.fixed_pool import run_fixed_pool
    assert jax.config.jax_enable_x64 and jax.default_backend()=='gpu'
    manifest=json.loads((OUT/'continuation.json').read_text())
    run=manifest['runs'][name];output=Path(run['output_dir'])
    assert repository_root().resolve()==Path(run['checkout']).resolve()
    assert scientific_source_hash(repository_root())==run['source_sha256']
    assert file_hash(output/'last.npz')==run['copied_checkpoint_sha256']['last.npz']
    assert file_hash(output/'best.npz')==run['copied_checkpoint_sha256']['best.npz']
    with np.load(output/'last.npz',allow_pickle=False) as saved:
        metadata=json.loads(str(saved['metadata'].item()))
        assert metadata['epoch']==run['start_epoch']==200
        for key in ['parameters','gradient','optimizer']:
            expected=run[{'parameters':'parameter_sha256','gradient':'gradient_sha256','optimizer':'optimizer_sha256'}[key]]
            assert hashlib.sha256(saved[key].tobytes()).hexdigest()==expected
    identity=metadata['identity']
    assert identity['model_config']==run['model_config']
    assert identity['learning_rate']==run['learning_rate']
    assert len(identity['references']['structures'])==24
    config.MODEL_CONFIG=dict(identity['model_config'])
    atomic_json(OUT/name/'resume-check.json',dict(passed=True,start_epoch=200,target_epochs=1000,
        parameter_bytes_preserved=True,gradient_bytes_preserved=True,optimizer_bytes_preserved=True,
        checkpoint_sha256=file_hash(output/'last.npz'),source_sha256=run['source_sha256'],
        launcher_sha256=file_hash(__file__),model_config=config.MODEL_CONFIG,
        learning_rate=identity['learning_rate'],device=jax.devices()[0].device_kind))
    result=run_fixed_pool(identity['references'],identity['candidate'],run['cache_dir'],output,
        epochs=1000,learning_rate=identity['learning_rate'],seed=identity['seed'],
        storage=identity['storage'],resume=True)
    return int(result['status']!='complete')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant',choices=['lr3e4','radial64','correlation3','baseline'])
    raise SystemExit(main(parser.parse_args().variant))
