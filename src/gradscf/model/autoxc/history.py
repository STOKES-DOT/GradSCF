"""Append-only attempt records, atomic checkpoints and immutable run identity."""
from dataclasses import asdict, is_dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path


def json_value(value):
    if is_dataclass(value): return json_value(asdict(value))
    if isinstance(value, dict): return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [json_value(v) for v in value]
    if hasattr(value, 'tolist'): return json_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value): return None
    if value is None or isinstance(value, (str, int, float, bool)): return value
    raise TypeError(f'Unsupported record type {type(value).__name__}')


def write_json(path, value, *, sort_keys=False):
    """Atomically publish a JSON record next to its temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as handle:
        json.dump(json_value(value), handle, sort_keys=sort_keys, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def source_digest():
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def versions():
    result = {}
    for name in ('jax', 'jaxlib', 'numpy', 'scipy', 'openai-codex'):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


class History:
    def __init__(self, directory, configuration):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.configuration = json_value(configuration)
        path = self.directory/'config.json'
        if path.exists():
            self.verify(configuration)
        elif (self.directory/'history.jsonl').exists():
            raise ValueError('History has no configuration; refuse an unverifiable restart')
        else:
            self.write('config.json', self.configuration)
        path = self.directory/'history.jsonl'
        self.records = ([json.loads(line) for line in path.read_text().splitlines()]
                        if path.exists() else [])
        if [r['round'] for r in self.records] != list(range(1, len(self.records)+1)):
            raise ValueError('History rounds are incomplete or duplicated')

    def verify(self, configuration):
        if json.loads((self.directory/'config.json').read_text()) != json_value(configuration):
            raise ValueError('Run configuration/fingerprint changed; use a new directory')

    def write(self, name, value):
        write_json(self.directory/name, value, sort_keys=True)

    def append(self, record):
        record = json_value(record)
        if record['round'] != len(self.records)+1:
            raise ValueError('Attempt round must follow the persisted history')
        with (self.directory/'history.jsonl').open('a') as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False)+'\n')
            handle.flush()
            os.fsync(handle.fileno())
        self.records.append(record)

    def pending(self):
        path = self.directory/'pending.json'
        return json.loads(path.read_text()) if path.exists() else None
