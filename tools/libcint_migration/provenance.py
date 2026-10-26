"""Post-run provenance from already imported modules and loaded libraries only."""
import hashlib
from pathlib import Path
import socket
import sys


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source(module):
    path = Path(vars(module)['__file__']).resolve()
    root, digest, count = path.parent, hashlib.sha256(), 0
    for item in sorted(root.rglob('*')):
        relative = item.relative_to(root)
        private = relative.parts[:3] == ('integrals', '_native', 'csrc')
        if item.is_file() and (item.suffix == '.py' or
                (private and item.suffix in ('.cc', '.cpp', '.cxx', '.h', '.hpp'))):
            digest.update(str(relative).encode()+b'\0'+item.read_bytes())
            count += 1
    return dict(path=str(path), version=vars(module).get('__version__'),
                tree_sha256=digest.hexdigest(), hashed_files=count,
                hash_scope='package .py and private integrals/_native/csrc C++ headers/sources')


def collect(probe, *, modules=None, maps_path=Path('/proc/self/maps')):
    """Call once after all samples; optional ledgers are metadata-test inputs."""
    modules = sys.modules if modules is None else modules
    probes = (probe,) if isinstance(probe, (str, Path)) else probe
    libraries = {}
    def add(consumer, filename):
        if filename:
            path = Path(filename).resolve()
            if path.is_file() and (consumer, str(path)) not in libraries:
                libraries[consumer, str(path)] = dict(consumer=consumer, path=str(path), sha256=_sha(path))
    native = modules.get('gradscf.integrals._native')
    register = vars(native).get('register_integrals') if native is not None else None
    if register is not None and hasattr(register, 'cache_info') and register.cache_info().currsize > 0:
        add('gradscf', vars(register()).get('_name'))  # An existing LRU hit, never an initial load.
    if modules.get('pyscf') is not None:
        intor = modules.get('pyscf.gto.moleintor')
        handle = vars(intor).get('libcgto') if intor is not None else None
        if handle is not None:
            add('pyscf', vars(handle).get('_name'))
        if Path(maps_path).is_file():
            for line in Path(maps_path).read_text().splitlines():
                row = line.split(maxsplit=5)
                if len(row) == 6 and any(tag in Path(row[5]).name for tag in ('libcint', 'libcgto')):
                    add('pyscf', row[5])
    return dict(python_executable=sys.executable, python_version=sys.version, hostname=socket.gethostname(),
        collector_sha256=_sha(__file__),
        versions={name: vars(modules[name]).get('__version__') if modules.get(name) is not None else None
                  for name in ('jax', 'numpy', 'scipy')},
        sources={name: _source(modules[name]) for name in ('gradscf', 'pyscf')
                 if modules.get(name) is not None and vars(modules[name]).get('__file__')},
        libraries=[libraries[key] for key in sorted(libraries)],
        probes={str(Path(path).resolve()): _sha(path) for path in probes})
