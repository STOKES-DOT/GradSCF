"""Copy scientific inputs/results without rebuildable caches or training states."""
import argparse
import gzip
import hashlib
import json
import os
import io
from pathlib import Path
import shutil
import zipfile


PRUNE = {'.git', '.venv', '__pycache__', '.pytest_cache', 'source', 'source_snapshot',
         'source_snapshots', 'sources', 'src', 'vendor', 'native', 'build', 'lib',
         'cache', 'caches', 'jit-cache', 'jit_cache', '.jax_cache', 'plot-cache',
         'conda-pkgs','conda_pkgs','pkgs','site-packages','.cache','conda-envs','wheels'}
SUFFIXES = {'.json', '.jsonl', '.csv', '.tsv', '.xyz', '.npz', '.npy', '.md',
            '.txt', '.log', '.out', '.err', '.png', '.pdf', '.svg', '.html',
            '.py', '.sh', '.slurm', '.yaml', '.yml', '.sha256', '.bib', '.sdf',
            '.res', '.resrc', '.msgpack', '.patch', '.dat', '.mol', '.mol2',
            '.cube', '.inp', '.toml', '.gin', '.cff', '.rst', '.jpg', '.jpeg', '.webp'}
SPECIAL_NAMES={'SHA256SUMS','CHECKSUMS','LICENSE','NOTICE','CITATION','README'}


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def excluded_npz(path):
    with zipfile.ZipFile(path) as archive:
        keys = {Path(name).stem for name in archive.namelist()}
    if any('optimizer' in key or 'adam' in key or 'opt_state' in key for key in keys) or ('parameters' in keys and 'gradient' in keys):
        return 'optimizer_or_training_checkpoint'
    if {'overlap', 'hcore', 'df_factors'} <= keys or {'eri', 'overlap'} <= keys:
        return 'rebuildable_integral_cache'
    if 'signature' in keys and (keys & {'eri', 'df_factors', 'ps', 'pb'}):
        return 'rebuildable_integral_cache'
    return None


def collect(plan, output, *, manifest_name='collection-manifest.json'):
    output = Path(output)
    records, excluded = [], []
    for source in plan.get('sources', plan.get('local_sources', [])):
        root = Path(source['path'])
        if not root.is_dir():
            raise FileNotFoundError(root)
        before = len(records)
        only_files=set(source['files']) if 'files' in source else None
        prune={'.git','__pycache__','.pytest_cache'} if source.get('preserve_layout') else PRUNE
        for current, directories, names in os.walk(root):
            directories[:] = sorted(d for d in directories if d not in prune
                and not d.startswith(('.git', 'jit-cache', 'xla-cache'))
                and not (Path(current)/d).is_symlink())
            for name in sorted(names):
                path = Path(current)/name
                if name=='.DS_Store' or name.startswith('._') or path.is_symlink():
                    continue
                if only_files is not None and str(path.relative_to(root)) not in only_files:
                    continue
                if path.suffix.lower() not in SUFFIXES and name not in SPECIAL_NAMES:
                    continue
                if any(word in name.lower() for word in ('local-cache-', 'integrals.npz')):
                    continue
                reason = excluded_npz(path) if path.suffix == '.npz' else None
                mixed=False
                if reason=='optimizer_or_training_checkpoint':
                    with zipfile.ZipFile(path) as archive:
                        mixed='outputs.npy' in archive.namelist()
                relative = Path(source['destination'])/path.relative_to(root)
                if reason and not mixed:
                    excluded.append(dict(source=str(path), reason=reason, bytes=path.stat().st_size))
                    continue
                stat = path.stat()
                target = output/relative
                compression = None
                if stat.st_size >= 95*1024*1024 and path.suffix in {'.json', '.jsonl', '.csv', '.log', '.out'}:
                    target = target.with_name(target.name+'.gz')
                    compression = 'gzip'
                target.parent.mkdir(parents=True, exist_ok=True)
                original_hash = digest(path)
                if mixed:
                    import numpy as np
                    with np.load(path,allow_pickle=False) as saved:
                        arrays={key:saved[key] for key in saved.files
                            if not any(s in key.lower() for s in ('optimizer','adam','opt_state'))}
                    buffer=io.BytesIO();np.savez(buffer,**arrays);payload=buffer.getvalue()
                    if target.exists() and target.read_bytes()!=payload:
                        raise RuntimeError(f'Existing archive differs from transformed source: {target}')
                    target.write_bytes(payload)
                elif target.exists():
                    if compression is None and digest(target) != original_hash:
                        raise RuntimeError(f'Existing archive differs from source: {target}')
                elif compression:
                    with path.open('rb') as incoming, target.open('wb') as raw:
                        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as outgoing:
                            shutil.copyfileobj(incoming, outgoing)
                else:
                    shutil.copy2(path, target)
                if path.stat().st_size != stat.st_size or path.stat().st_mtime_ns != stat.st_mtime_ns:
                    raise RuntimeError(f'Source changed during collection: {path}')
                record = dict(path=str(target.relative_to(output)), source=str(path),
                    source_sha256=original_hash, sha256=digest(target),
                    source_bytes=stat.st_size, bytes=target.stat().st_size)
                if compression:
                    record['compression'] = compression
                    with gzip.open(target, 'rb') as stream:
                        assert hashlib.file_digest(stream, 'sha256').hexdigest() == original_hash
                if mixed:record['transformation']='Optimizer arrays removed; scientific outputs and parameters preserved.'
                records.append(record)
        print(root.name, len(records)-before, 'files', flush=True)
    result = dict(version=1,files=records,excluded_training_states=excluded,
        file_count=len(records),bytes=sum(r['bytes'] for r in records))
    output.mkdir(parents=True,exist_ok=True)
    (output/manifest_name).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(files=len(records),bytes=result['bytes'],excluded_states=len(excluded))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--manifest-name',default='collection-manifest.json')
    args=parser.parse_args()
    collect(json.loads(args.plan.read_text()),args.output,manifest_name=args.manifest_name)


if __name__ == '__main__':
    main()
