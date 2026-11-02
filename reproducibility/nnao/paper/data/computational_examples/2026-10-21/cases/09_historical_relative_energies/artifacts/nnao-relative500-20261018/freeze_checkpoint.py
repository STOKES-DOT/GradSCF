"""Copy one atomic checkpoint inode while training continues, then record lineage."""
from pathlib import Path
import hashlib
import json
import shutil
import datetime
import numpy as np

OUT=Path(__file__).resolve().parent
TRAIN=OUT.parent/'mace500-20261014'
target=OUT/'checkpoint-current.npz'
assert not target.exists(),'Use the existing frozen snapshot; do not replace it.'
digest=hashlib.sha256()
with (TRAIN/'train/last.npz').open('rb') as source,target.open('xb') as destination:
    inode=__import__('os').fstat(source.fileno()).st_ino
    for block in iter(lambda:source.read(1024*1024),b''):
        digest.update(block);destination.write(block)
with np.load(target,allow_pickle=False) as a:
    metadata=json.loads(a['metadata'].item());parameters=a['parameters']
    assert np.isfinite(parameters).all()
    identity=metadata['identity']
    record=dict(frozen_at=datetime.datetime.now().astimezone().isoformat(),checkpoint_sha256=digest.hexdigest(),
        parameter_sha256=hashlib.sha256(parameters.tobytes()).hexdigest(),source_inode=inode,
        epoch=metadata['epoch'],update_step=metadata['update_step'],batch_cursor=metadata['batch_cursor'],
        parameters_evaluated=metadata.get('parameters_evaluated'),loss_hartree=metadata['loss_hartree'],
        loss_kind=metadata.get('loss_kind'),source_sha256=identity['scientific_source_sha256'],
        basis_family=identity['references']['basis_family'],core_primitives=identity['references']['core_primitives'],
        model_config=identity['model_config'],seed=identity['seed'],parameter_count=parameters.size,
        training_structures=len(identity['references']['structures']),
        method='DF-RHF',auxbasis=identity['auxbasis'])
(OUT/'snapshot.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2),flush=True)
