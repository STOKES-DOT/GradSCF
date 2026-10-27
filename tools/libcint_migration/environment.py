"""Read-only runtime provenance inside the allocated Slurm step."""
import contextlib
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def command(*args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except Exception as exc:
        return dict(error=repr(exc))


def collect():
    data = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                host=platform.node(), affinity=sorted(os.sched_getaffinity(0)),
                cpu_count=os.cpu_count(), environment={k: v for k, v in os.environ.items()
                    if any(s in k for s in ('THREAD', 'XLA', 'JAX', 'CUDA', 'SLURM', 'GRADSCF', 'MOLGW'))},
                cpuinfo=Path('/proc/cpuinfo').read_text(), meminfo=Path('/proc/meminfo').read_text(),
                cgroups=Path('/proc/self/cgroup').read_text(), versions={}, libraries={})
    for name in ('jax', 'jaxlib', 'numpy', 'scipy', 'pyscf', 'pyscfad', 'jax-xc', 'flax', 'optax'):
        try:
            data['versions'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            data['versions'][name] = None
    import numpy as np
    import jax
    jax.config.update('jax_enable_x64', True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        np.show_config()
    data['numpy_config'] = buf.getvalue()
    data['jax_devices'] = [str(x) for x in jax.devices()]
    data['jax_probe'] = float(jax.jit(lambda x: x * x)(np.float64(2.0)).block_until_ready())
    import gradscf
    import pyscf
    data['gradscf_import'] = str(gradscf.__file__)
    data['pyscf_import'] = str(pyscf.__file__)
    for base in (Path(gradscf.__file__).parent/'integrals/_native/lib', Path(pyscf.__file__).parent/'lib'):
        for path in base.rglob('*.so*'):
            if path.is_file():
                data['libraries'][str(path)] = dict(bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    from pyscf import dft, gto, gw, mp
    data['capabilities'] = dict(pyscf_gw_cd=importlib.util.find_spec('pyscf.gw.gw_cd') is not None,
        pyscf_bse=importlib.util.find_spec('pyscf.bse') is not None,
        pyscf_libxc_version=dft.libxc.__version__)
    for filename in ('memory.max', 'memory.limit_in_bytes', 'cpuset.cpus.effective'):
        data[filename] = {str(p): p.read_text() for p in Path('/sys/fs/cgroup').glob('*/'+filename)
                          if p.is_file()}
    data['nvidia_smi'] = command('nvidia-smi', '--query-gpu=index,name,uuid,memory.total,driver_version', '--format=csv')
    data['mpi'] = command('mpirun', '--version')
    data['gcc'] = command('gcc', '--version')
    data['ldd'] = {p: command('ldd', p) for p in data['libraries'] if 'gradscf_integrals' in p}
    try:
        from threadpoolctl import threadpool_info
        data['threadpools'] = threadpool_info()
    except ImportError:
        pass
    return data


if __name__ == '__main__':
    Path(sys.argv[1]).write_text(json.dumps(collect(), indent=2)+'\n')
