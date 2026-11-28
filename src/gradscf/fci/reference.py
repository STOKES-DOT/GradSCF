"""Active-space Hamiltonians and eager common-orbital SCF adaptation."""
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
from gradscf._memory import (array_storage_bytes, resolve_memory_mb, tree_storage_bytes, workspace_bytes)
from gradscf.ao2mo import validate_integrals, transform_integrals
from gradscf.ao2mo.active import orbital_selection, read_ao_representation, transform_active
from ..scf.reference import _array_signature, reference_state_signature
from .cistring import unpack_nelec


@dataclass(frozen=True)
class FCIReference:
    h1: object
    eri: object
    nelec: object
    ecore: object = 0.


def fold_core(h1, eri, *, core=0, active=None, ecore=0.):
    """Return (h_active, eri_active, constant) for a doubly occupied core.

    Orbitals outside core+active are empty. Active orbital order is preserved.
    No electron count is inferred and no SCF or orbital optimization is run.
    """
    h,g = validate_integrals(h1,eri)
    core,active = orbital_selection(h.shape[0],core,active)
    c,a = jnp.asarray(core,dtype=jnp.int32),jnp.asarray(active,dtype=jnp.int32)
    hactive = h[jnp.ix_(a,a)]
    constant = jnp.asarray(ecore)
    if constant.shape != () or jnp.iscomplexobj(constant):
        raise ValueError("ecore must be a real scalar")
    if core:
        hactive = hactive + 2*jnp.einsum('pqii->pq',g[jnp.ix_(a,a,c,c)])
        hactive = hactive - jnp.einsum('piiq->pq',g[jnp.ix_(a,c,c,a)])
        gcc = g[jnp.ix_(c,c,c,c)]
        constant += 2*jnp.trace(h[jnp.ix_(c,c)]) + 2*jnp.einsum('iijj->',gcc) - jnp.einsum('ijji->',gcc)
    return hactive,g[jnp.ix_(a,a,a,a)],constant


def source_dimensions(source, spin=None):
    if isinstance(source,FCIReference):
        shape = np.shape(source.h1)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("FCIReference.h1 must be square")
        n = shape[0]
        return n,unpack_nelec(n,source.nelec,spin)
    from ..scf.facade import RKS
    from ..scf.roks import ROKS
    if not isinstance(source,(RKS,ROKS)):
        raise NotImplementedError("FCI accepts real common-orbital RKS/ROHF sources or FCIReference")
    if not source.converged or source.scf_result is None or source._scf_inputs is None:
        raise RuntimeError("Run and converge SCF before FCI")
    if np.ndim(source.mo_coeff) != 2:
        raise NotImplementedError("FCI requires a common spatial orbital frame")
    if source._cached_scf_key != source._scf_signature():
        raise RuntimeError("SCF inputs changed; run SCF again before FCI")
    n = source.mo_coeff.shape[1]
    return n,unpack_nelec(n,source.mol.nelectron,source.mol.spin if spin is None else spin)


def source_signature(source):
    if isinstance(source,FCIReference):
        return (unpack_nelec(np.shape(source.h1)[0],source.nelec),
                *(_array_signature(a) for a in (source.h1,source.eri,source.ecore)))
    return reference_state_signature(source)


def active_reference(source, core, active, *, max_memory=None):
    """Adapt an eager common frame; max_memory budgets active preparation."""
    max_memory = resolve_memory_mb(max_memory,source)
    if isinstance(source,FCIReference):
        shape = np.shape(source.h1)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("FCIReference.h1 must be square")
        core,active = orbital_selection(shape[0],core,active)
        arrays = (source.h1,source.eri,source.ecore)
        itemsize = max(8,*(np.dtype(getattr(a,'dtype',np.float64)).itemsize for a in arrays))
        resident = array_storage_bytes(*arrays)
        # NumPy/sequences may require a backend copy. This is a conservative
        # forecast: CPU jnp.asarray can alias NumPy storage. Existing JAX and
        # abstract/traced inputs require no additional full-array copy here.
        numerical = (jax.Array,jax.core.Tracer,jax.ShapeDtypeStruct)
        if not isinstance(source.h1,numerical):
            resident += shape[0]**2*itemsize
        if not isinstance(source.eri,numerical):
            resident += shape[0]**4*itemsize
        device = getattr(source.h1,'device',None)
        device = device() if callable(device) else device
        available = workspace_bytes(max_memory,resident_bytes=resident,device=device)
        nc,na = len(core),len(active)
        # fold_core gathers the active ERI, both active/core blocks, and the
        # closed-core ERI, while retaining one-body gather/update versions.
        required = itemsize*(na**4 + 3*na**2 + 2*na**2*nc**2 + nc**4 + nc**2)
        if required > available:
            raise ValueError("Explicit FCI core-folding arrays exceed the working-memory budget")
        return fold_core(source.h1,source.eri,core=core,active=active,ecore=source.ecore)
    inputs = source._scf_inputs
    coefficients = jnp.asarray(source.mo_coeff)
    overlap = np.asarray(inputs.overlap)
    c = np.asarray(coefficients)
    if (np.iscomplexobj(c) or not np.isfinite(c).all()
            or not np.allclose(c.T@overlap@c,np.eye(c.shape[1]),atol=1e-8,rtol=0)):
        raise ValueError("FCI requires finite real S-orthonormal orbitals")
    representation = read_ao_representation(inputs, coefficients, max_memory=max_memory, owner=source)
    numerical = (jax.Array,jax.core.Tracer,jax.ShapeDtypeStruct)
    child = (inputs.hcore,coefficients,inputs.nuclear_repulsion,*jax.tree.leaves(representation))
    reserved = tree_storage_bytes(source,exclude=tuple(x for x in child if isinstance(x,numerical)))
    return _active_from_orbitals(inputs.hcore,coefficients,core=core,active=active,
        ecore=inputs.nuclear_repulsion,max_memory=max_memory,reserved_bytes=reserved,**representation)


def _active_from_orbitals(hcore, coefficients, **kwargs):
    return transform_active(hcore, coefficients, _transform=transform_integrals, **kwargs)
