"""Read-only fixed-primitive integral cache storage."""
from pathlib import Path
from functools import lru_cache
import math
import hashlib
import json
import struct
import zipfile
import numpy as np
from ..runtime import file_hash


_FIXED_CACHE_SCHEMA = "nnao-fixed-primitive-libcint-v3"


@lru_cache(maxsize=32)
def _hash_file(path, fingerprint):
    """Memoize content hashes, never numerical integrals, by file identity."""
    del fingerprint  # Its inode/timestamps/size participate in the LRU key.
    return file_hash(path)


def _source_hash(path):
    path = Path(path).resolve()
    stat = path.stat()
    fingerprint = (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
    return _hash_file(str(path), fingerprint)


def _array_identity(value):
    array = np.asarray(value)
    return dict(dtype=array.dtype.str, shape=array.shape, values=array.tolist())


def _basis_identity(topology, parameters):
    return dict(cart=bool(topology.cart), angular_momenta=topology.angular_momenta,
        primitive_counts=topology.primitive_counts, contraction_counts=topology.contraction_counts,
        nuclear_charges=topology.nuclear_charges,
        exponents=[_array_identity(x) for x in parameters.exponents],
        coefficients=[_array_identity(x) for x in parameters.coefficients],
        centers=_array_identity(parameters.centers), nuclear_coords=_array_identity(parameters.nuclear_coords))


def fixed_signature(layout, symbols, coords, basis_family, auxbasis, topology, parameters,
                    *, auxiliary_topology=None, auxiliary_parameters=None):
    """Bind fixed primitive data and the libcint/normalization implementation.

    Model theta and contracted coefficients in ``layout`` are applied after
    this cache; only the supplied physical primitive snapshot enters its key.
    Prepared auxiliary data are supplied by DF consumers, without evaluating
    an integral or parsing an unrelated auxiliary basis on direct paths.
    """
    from gradscf.integrals import _native
    from gradscf.integrals.basis import normalization

    if (auxiliary_topology is None) != (auxiliary_parameters is None):
        raise ValueError('Supply auxiliary topology and parameters together.')
    auxiliary = (None if auxiliary_topology is None else
                 _basis_identity(auxiliary_topology, auxiliary_parameters))
    payload = dict(format=_FIXED_CACHE_SCHEMA, symbols=symbols, coords=_array_identity(coords),
        basis=basis_family, auxbasis=auxbasis, primitive=_basis_identity(topology, parameters),
        auxiliary=auxiliary, native_sha256=_source_hash(_native.register_integrals()._name),
        normalization_sha256=_source_hash(normalization.__file__))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_fixed_cache(path, signature, storage):
    import jax.numpy as jnp
    names = ('overlap', 'hcore', 'df_factors')
    with np.load(path, allow_pickle=False) as cache:
        if str(cache['signature'].item()) != signature:
            raise ValueError('Integral cache identity does not match this geometry, basis or libcint implementation; '
                             'rebuild this cache.')
        if storage == 'mmap':
            return tuple(map_npz_numeric(path, name) for name in names)
        array = np.asarray if storage == 'host' else jnp.asarray
        return tuple(array(cache[name]) for name in names)


def write_fixed_cache(path, signature, overlap, hcore, factors):
    path = Path(path)
    if not str(path).endswith('.npz'):
        path = Path(str(path)+'.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, signature=np.asarray(signature), overlap=np.asarray(overlap),
             hcore=np.asarray(hcore), df_factors=np.asarray(factors))
    return path

def map_npz_numeric(path,name):
    """Read-only map a numeric ZIP_STORED NPY member of the original NPZ file."""
    with zipfile.ZipFile(path) as archive,Path(path).open('rb') as stream:
        member=archive.getinfo(name+'.npy')
        if member.compress_type!=zipfile.ZIP_STORED or member.flag_bits&1:
            raise ValueError('NPZ mmap requires unencrypted ZIP_STORED members; compressed arrays cannot be mapped.')
        stream.seek(member.header_offset)
        header=struct.unpack('<4s5H3I2H',stream.read(30))
        if header[0]!=b'PK\x03\x04' or header[3]!=zipfile.ZIP_STORED or header[2]&1:
            raise ValueError('Invalid ZIP_STORED local header for NPZ mmap.')
        start=member.header_offset+30+header[-2]+header[-1]
        stream.seek(start)
        version=np.lib.format.read_magic(stream)
        readers={(1,0):np.lib.format.read_array_header_1_0,
                 (2,0):np.lib.format.read_array_header_2_0}
        if version not in readers:raise ValueError(f'Unsupported NPY version for NPZ mmap: {version}.')
        shape,fortran,dtype=readers[version](stream)
        if dtype.hasobject or dtype.kind not in 'biufc':
            raise ValueError('NPZ mmap supports only numeric arrays without object dtype.')
        offset=stream.tell();size=math.prod(shape)*dtype.itemsize
        if offset+size!=start+member.file_size:
            raise ValueError('NPZ numeric payload size differs from its NPY header.')
    return np.memmap(path,dtype=dtype,mode='r',offset=offset,shape=shape,
                     order='F' if fortran else 'C')
