"""Continue the latest Adam state directly with one evaluation per batch."""
from pathlib import Path
import json
import jax
from gradscf.model.nnao.runtime import scientific_source_hash,repository_root
from gradscf.model.nnao.workflows.fixed_pool import run_fixed_pool

ROOT=Path(__file__).resolve().parent
record=json.loads((ROOT/'single-pass-switch.json').read_text())
source=json.loads((ROOT/'source.json').read_text())
assert record['status']=='applied'
assert record['source_sha256']==source['source_sha256']==scientific_source_hash(repository_root())
assert jax.config.jax_enable_x64 and jax.default_backend()=='gpu'
print('Single-pass resume:',record['resume'],'; retain Adam; no startup replay, candidate check, epoch scoring or finite differences.',flush=True)
run_fixed_pool(json.loads((ROOT/'inputs/references.json').read_text()),
    json.loads((ROOT/'selection/candidate.json').read_text()),ROOT/'cache',ROOT/'train',
    epochs=1000,learning_rate=1e-3,seed=0,storage='mmap',resume=True,
    model_config=json.loads((ROOT/'inputs/model-config.json').read_text()),
    batch_size=50,decay_epochs=100,check_parameter_gradients=False)
