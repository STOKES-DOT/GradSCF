"""Persistent mixed-pool S/h/packed-DF cache for fixed-exponent training."""
import hashlib
import fcntl
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gradscf import integrals, scf
from gradscf.integrals._native import register_integrals
from ..runtime import atomic_json, file_hash
from .cache import read_fixed_cache, write_fixed_cache
from .pool import active_pool_basis


def prepare_fixed_pool(row, layout, path, *, storage='device',
                       auxbasis='def2-universal-jkfit', implicit_tolerance=1e-8):
    """Prepare once on CPU, then reuse numeric arrays without native AD calls.

    Core functions are already contracted in this pool. The manifest includes
    their coefficients, absolute exponents, geometry, auxiliary basis and native
    library identity. Existing incompatible/corrupt caches fail explicitly.
    """
    from ..training.objective import _df_rhf_solver
    from ..training.pool import FreeContractionRHF, overlap_diagnostics
    from ..training.initial_guess import atomic_guess_data
    from ..initialization import core_reference_identity
    if storage not in {'device', 'host', 'mmap'} or not jax.config.jax_enable_x64:
        raise ValueError('Fixed-pool training requires float64 and device/host/mmap storage.')
    if row.get('charge') != 0 or row.get('spin') != 0 or layout.nelectron % 2:
        raise ValueError('Fixed-pool pilot requires neutral closed-shell RHF.')
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    manifest_path = path.with_suffix('.json')
    with path.with_suffix('.lock').open('a') as lock,jax.default_device(jax.devices('cpu')[0]):
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        top, parameters = active_pool_basis(layout)
        atom = list(zip(row['symbols'], row['coords_angstrom']))
        at, ap = integrals.prepare_basis(atom, auxbasis, cart=False)
        def arrays(p):
            return dict(exponents=[np.asarray(x).tolist() for x in p.exponents],
                coefficients=[np.asarray(x).tolist() for x in p.coefficients],
                centers=np.asarray(p.centers).tolist(),
                nuclear_coords=np.asarray(p.nuclear_coords).tolist())
        identity = dict(format='nnao-fixed-active-pool-v2', dtype='float64',
            symbols=list(layout.symbols), roles=list(layout.roles),
            active_shells=[slot>=0 for slot in layout.slots],
            angular_momenta=list(top.angular_momenta), cart=top.cart,
            charges=list(top.nuclear_charges), pool=arrays(parameters),
            auxiliary=arrays(ap), auxiliary_angular_momenta=list(at.angular_momenta),
            auxbasis=auxbasis, native_sha256=file_hash(register_integrals()._name),
            projection_sha256=file_hash(Path(__file__).with_name('pool.py')))
        if core_reference_identity(layout) is not None:
            identity['core_reference'] = core_reference_identity(layout)
        signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if path.exists() or manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest['identity'] != identity or manifest['signature'] != signature:
                raise ValueError('Fixed-pool cache identity/signature changed; use a new cache directory.')
            if manifest['cache_sha256'] != file_hash(path):
                raise ValueError('Fixed-pool cache checksum/hash mismatch.')
        else:
            plan = integrals.make_plan(top)
            aux = integrals.make_auxiliary_plan(top, at)
            metric = aux.metric_factor(parameters, ap)
            s = plan.evaluate('overlap', parameters)
            h = plan.evaluate('kinetic', parameters)+plan.evaluate('nuclear', parameters)
            b = aux.factors(parameters, ap, metric_factor=metric)
            overlap_diagnostics(s)
            temporary = path.with_name(path.stem+'.tmp.npz')
            write_fixed_cache(temporary, signature, s, h, b)
            temporary.replace(path)
            manifest = dict(identity=identity, signature=signature, cache_sha256=file_hash(path),
                pool_nao=top.nao, creator_contracted_nao=layout.topology.nao,
                df_shape=list(b.shape), bytes=int(s.nbytes+h.nbytes+b.nbytes))
            atomic_json(manifest_path, manifest)
        manifest=dict(manifest,contracted_nao=layout.topology.nao)
        enuc = float(scf.nuclear_repulsion_energy(parameters.nuclear_coords,
                                                jnp.asarray(top.nuclear_charges)))
    raw = read_fixed_cache(path, signature, storage)
    diagnostics = overlap_diagnostics(raw[0])
    value, gradient = _df_rhf_solver(layout.nelectron, implicit_tolerance, .2)
    return (FreeContractionRHF(layout, raw, jnp.asarray(enuc), value, gradient,
                              row['structure_id'], diagnostics,atomic_guess=atomic_guess_data(layout)), manifest)
