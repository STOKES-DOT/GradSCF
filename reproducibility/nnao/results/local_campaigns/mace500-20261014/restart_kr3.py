"""Resume accepted update 29; finite-difference validation disabled by request."""
from pathlib import Path
import hashlib
import json
import jax
import numpy as np
from gradscf.model.nnao.runtime import scientific_source_hash,repository_root
from gradscf.model.nnao.workflows.fixed_pool import run_fixed_pool

ROOT=Path(__file__).resolve().parent
source=json.loads((ROOT/'source.json').read_text())['source_sha256']
gate=json.loads((ROOT/'kr3-restart-verification.json').read_text())
assert gate['passed'] and gate['finite_differences'] is False
assert gate['source_sha256']==source==scientific_source_hash(repository_root())
assert jax.config.jax_enable_x64 and jax.default_backend()=='gpu'
with np.load(ROOT/'train/last.npz',allow_pickle=False) as a:
    checkpoint=json.loads(a['metadata'].item())
    assert checkpoint['identity']['scientific_source_sha256']==source
    assert checkpoint['update_step']==29 and checkpoint['batch_cursor']==9 and checkpoint['epoch']==2
    assert hashlib.sha256(a['parameters'].tobytes()).hexdigest()==gate['parameter_sha256']['last']
data=json.loads((ROOT/'inputs/references.json').read_text())
candidate=json.loads((ROOT/'selection/candidate.json').read_text())
model=json.loads((ROOT/'inputs/model-config.json').read_text())
print('Resume accepted update 29; retain Adam; skip finite-difference checks by explicit user request.',flush=True)
run_fixed_pool(data,candidate,ROOT/'cache',ROOT/'train',epochs=1000,
    learning_rate=1e-3,seed=0,storage='mmap',resume=True,model_config=model,
    batch_size=50,decay_epochs=100,check_parameter_gradients=False)
