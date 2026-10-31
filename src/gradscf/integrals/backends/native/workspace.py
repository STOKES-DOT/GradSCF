"""Bind a method's remaining byte allowance to libcint accumulation work."""
from dataclasses import replace
from numbers import Integral

from gradscf._memory import workspace_bytes


def bind_jk_workspace(provider, *, workers, max_memory, resident_bytes=0,
                      overlap=3, cutoff=0.):
    """Keep physical leaves; partition remaining workspace across live calls.

    ``overlap=3`` retains the shared SCF allowance for a primal and both
    derivative orientations. Private outputs use at most a quarter of the
    remaining allowance, leaving room for solver/XC array work. Standalone
    providers keep their 64 MiB default;
    an explicitly smaller provider quota is always honored.
    """
    from .jk import (NativeDirectBasis, ProjectedNativeDirectBasis,
                     _DIRECT_JK_PRIVATE_MAX_BATCH, native_jk_workspace_bytes)
    if max_memory is None:
        return provider
    if isinstance(overlap, bool) or not isinstance(overlap, Integral) or overlap < 1:
        raise ValueError('overlap must be a positive integer.')
    primitive = provider.primitive_basis if type(provider) is ProjectedNativeDirectBasis else provider
    if type(primitive) is not NativeDirectBasis:
        return provider  # Opaque providers retain their caller-owned contract.
    base = native_jk_workspace_bytes(primitive, workers, cutoff, private_output_bytes=0)
    available = workspace_bytes(max_memory, resident_bytes=resident_bytes, device='cpu')-overlap*base
    if available < 0:
        raise ValueError('Native J/K fixed work exceeds max_memory; reduce workers or increase the budget.')
    cell = primitive.nao**2*8
    count = min(workers, len(primitive.plan.topology.angular_momenta))
    maximum = count*2*_DIRECT_JK_PRIVATE_MAX_BATCH*cell if count > 1 else 0
    quota = min(maximum, (available//(4*overlap)//cell)*cell)
    if primitive.private_output_bytes is not None:
        quota = min(quota, primitive.private_output_bytes)
    bounded = replace(primitive, private_output_bytes=int(quota))
    return replace(provider, primitive_basis=bounded) if type(provider) is ProjectedNativeDirectBasis else bounded
