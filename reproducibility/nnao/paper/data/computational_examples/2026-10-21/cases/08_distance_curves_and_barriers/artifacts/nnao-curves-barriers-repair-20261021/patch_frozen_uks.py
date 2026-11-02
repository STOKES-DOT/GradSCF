"""Apply only the reviewed UKS density/orbital-hint fix to the frozen checkout."""
from datetime import datetime, timezone
from pathlib import Path
import difflib
import hashlib
import json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hash(root):
    result = hashlib.sha256()
    package = root/'src/gradscf'
    for path in sorted(package.rglob('*')):
        if path.is_file() and path.suffix in {'.py', '.json', '.dat'}:
            result.update(str(path.relative_to(package)).encode())
            result.update(digest(path).encode())
    return result.hexdigest()


root = Path.cwd()
output = root/'artifacts/nnao-curves-barriers-repair-20261021'
original_snapshot = root/'artifacts/nnao-relative500-20261018/snapshot.json'
snapshot = json.loads(original_snapshot.read_text())
target = root/'src/gradscf/scf/uks.py'
expected = '7568ef38d928c70de0a352889624ea005d5662bdbbb0ce636ba9d5b0a0cd314e'
if digest(target) != expected or source_hash(root) != snapshot['source_sha256']:
    raise ValueError('Frozen source changed; inspect it before applying the repair.')
checkpoint = root/'artifacts/nnao-relative500-20261018/checkpoint-current.npz'
if digest(checkpoint) != snapshot['checkpoint_sha256']:
    raise ValueError('Frozen checkpoint changed.')
if 'def _df_jk_with_orbital_hint(' not in target.with_name('rks.py').read_text():
    raise ValueError('The existing safe shared DF helper is missing.')
before = target.read_text()
old = '''                weighted = jnp.asarray(coeff, dtype=factors.dtype) * jnp.sqrt(
                    jnp.maximum(jnp.asarray(occ, dtype=factors.dtype), 0.0)
                )[None, :]
                b_occ = jnp.einsum("Qpr,ri->Qpi", factors, weighted, precision=Precision.HIGHEST)
                k_mat = jnp.einsum("Qpi,Qqi->pq", b_occ, b_occ, precision=Precision.HIGHEST)
'''
new = '''                _, k_mat = _df_jk_with_orbital_hint(
                    factors, density, jnp.asarray(coeff, dtype=factors.dtype),
                    jnp.asarray(occ, dtype=factors.dtype), int(coeff.shape[-1]),
                )
'''
anchor = '    _build_jk,\n'
if before.count(old) != 1 or before.count(anchor) != 1:
    raise ValueError('UKS patch context differs from the reviewed source.')
after = before.replace(anchor, anchor+'    _df_jk_with_orbital_hint,\n').replace(old, new)
compile(after, str(target), 'exec')
backup = output/'source-before/uks.py'
backup.parent.mkdir(parents=True, exist_ok=True)
with backup.open('x') as stream:
    stream.write(before)
target.write_text(after)
patch = output/'uks.patch'
patch.write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
    fromfile='a/src/gradscf/scf/uks.py', tofile='b/src/gradscf/scf/uks.py')))
provenance = dict(timestamp=datetime.now(timezone.utc).isoformat(),
    parent_snapshot_sha256=digest(original_snapshot), parent_source_sha256=snapshot['source_sha256'],
    source_sha256=source_hash(root), checkpoint_sha256=digest(checkpoint),
    changed_file='src/gradscf/scf/uks.py', before_sha256=digest(backup), after_sha256=digest(target),
    patch_sha256=digest(patch), reason='DF exchange must use the input density when orbital hints disagree; reuse the existing physical-density JVP.')
(output/'source-patch.json').write_text(json.dumps(provenance, indent=2)+'\n')
snapshot.update(source_sha256=provenance['source_sha256'], source_repair=provenance)
(output/'snapshot.json').write_text(json.dumps(snapshot, indent=2)+'\n')
print(json.dumps(provenance, indent=2))
