"""Static molecular GW working-array plans using the common decimal-MB budget."""
from dataclasses import dataclass

import numpy as np

from .._memory import (DEFAULT_MAX_MEMORY_MB, array_device, array_storage_bytes,
                      validate_memory_mb, validate_reserved_bytes, workspace_bytes)
from ..scf._pytree import pytree_dataclass


@pytree_dataclass(static_fields=("max_memory", "resident_bytes", "working_bytes",
                                 "frequency_batch", "rhs_batch", "projected_residue",
                                 "screening_config"))
@dataclass(frozen=True)
class GWExecutionPlan:
    max_memory: float
    resident_bytes: int
    working_bytes: int
    frequency_batch: int
    rhs_batch: int
    projected_residue: bool = True
    screening_config: object = None


def factor_promotion_bytes(factors, dtype):
    """Reserve a possible whole-factor cast retained by J/legacy contractions."""
    dtype = np.dtype(dtype)
    return factors.size*dtype.itemsize if np.dtype(factors.dtype) != dtype else 0


def plan_gw(factors, coefficients, *, nocc, n_targets, n_g, screening_shape, nw,
            max_memory=DEFAULT_MAX_MEMORY_MB, reserved_bytes=0, screening_config=None):
    """Reserve selected vertices, screening data and complex dense solve work.

    Caller-owned F/h/D/energies belong in reserved_bytes. Factor preparation
    and nonlinear response have different lifetimes; this plan covers the
    selected molecular forward kernels, not arbitrary outer AD/model state or
    compiler/allocator residency. Mandatory outputs are never truncated.
    """
    maximum = validate_memory_mb(max_memory)
    reserved = validate_reserved_bytes(reserved_bytes)
    nocc, n_targets, n_g, nw = (
        validate_reserved_bytes(x, name=name) for x, name in
        ((nocc, "nocc"), (n_targets, "n_targets"), (n_g, "n_g"), (nw, "nw")))
    no, nv = (validate_reserved_bytes(x, name="screening dimension") for x in screening_shape)
    q = factors.shape[0]
    item = np.dtype(np.result_type(factors.dtype, coefficients.dtype)).itemsize
    pairs = n_targets*n_g
    # Two independent Qpm vertices, OV factors, target exchange blocks, and
    # fixed-frequency projected W (including its zero-frequency subtraction).
    selected = q*(2*pairs+no*nv+2*n_targets*nocc)
    promoted = factor_promotion_bytes(factors, np.result_type(factors.dtype, coefficients.dtype))
    resident = reserved+array_storage_bytes(factors, coefficients)+promoted+item*(selected+2*(nw+1)*pairs)
    device = array_device(factors, coefficients)
    available = workspace_bytes(maximum, resident_bytes=resident, device=device)
    # Retarded Pi, epsilon, LU/copies and derivative solve temporaries. This
    # intentionally uses complex storage even for the cheaper imaginary axis.
    complex_item = 2*item
    # Include weighted primal/tangent factors and possible complex operand
    # promotion in the frequency slopes required by forward QP certification.
    minimum = (8*q*q+4*q*no*nv)*complex_item
    per_rhs = 4*q*complex_item
    iterative = screening_config is not None and screening_config.method == "gmres"
    if iterative:
        from ..solvers.memory import plan_linear
        screening_config, minimum = plan_linear(2*q, np.dtype(np.result_type(factors.dtype, coefficients.dtype)),
            screening_config, maximum, resident_bytes=resident,
            operator_min_bytes=4*(q+no*nv)*complex_item, device=device)
        per_rhs += 4*no*nv*complex_item
    if not pairs:
        minimum = per_rhs = 0
    if minimum+per_rhs > available:
        raise ValueError("GW minimum screening workspace exceeds the memory budget")
    rhs = min(max(1, pairs), max(1, (available-minimum)//max(1, per_rhs)))
    per_frequency = minimum+rhs*per_rhs
    frequencies = min(max(1, nw), max(1, available//max(1, per_frequency)))
    if iterative:
        frequencies = 1
    # Small GPU dense batches were faster than serial/tiled pole products in
    # measured probes. Preserve that schedule ONLY when the entire QP/G batch
    # fits; the projected path bounds larger layouts. CPU skips inactive poles.
    dense_residue = n_targets*n_g*(8*q*q+4*q*no*nv)*complex_item
    projected = (iterative or getattr(device, "platform", "cpu") == "cpu"
                 or dense_residue > available)
    working = max(per_frequency*frequencies, 0 if projected else dense_residue)
    return GWExecutionPlan(maximum, resident, working, frequencies, rhs, projected, screening_config)


def plan_qsgw(factors, occupied_virtual, *, nw, max_memory=DEFAULT_MAX_MEMORY_MB,
              reserved_bytes=0, screening_config=None, dtype=None):
    """Stream exact endpoint products into an MO matrix, retaining no W cube.

    The caller reserves AO/frame arrays; factors and the OV response remain
    live here. Imaginary RHS columns are tiled; the shared retarded own-pole
    solve retains a complete Q-by-MO block, including its complex workspace.
    """
    maximum = validate_memory_mb(max_memory)
    reserved = validate_reserved_bytes(reserved_bytes)
    nw = validate_reserved_bytes(nw, name="nw")
    q, n, _ = factors.shape
    if factors.shape != (q, n, n) or occupied_virtual.ndim != 3 or occupied_virtual.shape[0] != q:
        raise ValueError("qsGW requires full MO factors and matching OV response factors")
    no, nv = occupied_virtual.shape[1:]
    item = np.dtype(np.result_type(factors.dtype, occupied_virtual.dtype,
                                  factors.dtype if dtype is None else dtype)).itemsize
    resident = reserved + array_storage_bytes(factors, occupied_virtual) + item*(8*n*n+4*n+2*nw)
    device = array_device(factors, occupied_virtual)
    available = workspace_bytes(maximum, resident_bytes=resident, device=device)
    complex_item = 2*item
    per_rhs = (4*q+4*n)*complex_item
    # Independent non-own vertices and the shared full Q-by-MO own-pole RHS,
    # screened RHS, solution, and real/complex solve/update copies coexist.
    extra = 4*q*n*item+2*per_rhs+4*q*n*complex_item
    if screening_config is not None:
        from ..solvers import LinearSolverConfig
        from ..solvers.memory import plan_linear
        if not isinstance(screening_config, LinearSolverConfig) or screening_config.method != "gmres":
            raise ValueError("qsGW screening_config must be a shared GMRES LinearSolverConfig")
        per_rhs += 4*no*nv*complex_item
        screening_config, minimum = plan_linear(2*q, np.dtype(f'f{item}'), screening_config,
            maximum, resident_bytes=resident+extra+per_rhs,
            operator_min_bytes=4*(q+no*nv)*n*complex_item, nrhs=n, device=device)
    else:
        minimum = (8*q*q+4*q*no*nv)*complex_item
    minimum += extra
    if minimum+per_rhs > available:
        raise ValueError("qsGW minimum screening workspace exceeds the memory budget")
    rhs = min(max(1, n), max(1, (available-minimum)//max(1, per_rhs)))
    return GWExecutionPlan(maximum, resident, minimum+rhs*per_rhs, 1, rhs, True, screening_config)


def full_gw_working_bytes(naux, nmo, nw, *, transitions=None, screening_size=0, itemsize=8):
    """Conservative known forward arrays for the retained full/reference route."""
    pairs = nmo*nmo
    storage = itemsize*(2*naux*pairs+2*(nw+1)*pairs)
    if transitions is not None:
        storage += itemsize*(16*transitions**2+2*(nw+3)*transitions*pairs+3*naux*transitions)
    else:
        batched = nw*naux*naux*16*2 <= (1 << 31)
        storage += 16*(8*naux*naux+2*naux*screening_size)*(nw if batched else 1)
    return storage
