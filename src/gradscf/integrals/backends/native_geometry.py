"""Geometry AD using native shell contractions and JAX FFI.

Primal basis data are separate from differentiable coordinates so unsupported
basis derivatives cannot silently become zero. First geometry derivatives are
differentiable in their direction/cotangent, enabling mixed model/geometry AD
without requiring second geometry derivatives. Neither path imports PySCF.
"""
from functools import lru_cache, partial

import jax
from jax.custom_derivatives import SymbolicZero
from jax.extend import core
from jax.interpreters import ad, batching, mlir
import jax.numpy as jnp
import numpy as np

from .native import evaluate as evaluate_values, _OPERATORS


@lru_cache(maxsize=64)
def _geometry_function(operator, atoms_key, basis_key, nao, cart):
    atoms = np.asarray(atoms_key, dtype=np.int32)
    basis = np.asarray(basis_key, dtype=np.int32)
    offsets = atoms[:, 1, None]+np.arange(3)[None, :]
    if np.unique(offsets).size != offsets.size:
        raise ValueError("Native geometry plans require distinct coordinate storage per atom/center")
    shape = (nao,)*4 if operator == "eri" else ((3, nao, nao) if operator == "dipole" else (nao, nao))

    def pack(env, coords, origin):
        env = env.at[offsets].set(coords, unique_indices=True)
        return env.at[1:4].set(origin)

    def ffi_product(target, env, vector, output_shape):
        call = jax.ffi.ffi_call(target, jax.ShapeDtypeStruct(output_shape, jnp.float64),
                                vmap_method="sequential")
        return call(jnp.asarray(atoms), jnp.asarray(basis), env, vector,
                    operator=np.int32(_OPERATORS[operator]), cart=np.int32(cart))

    # At fixed env these are mutually adjoint linear maps. Keep both behind
    # primitives so mixed AD differentiates their vector arguments without
    # reaching the raw FFI or materializing an integral Jacobian.
    linear = core.Primitive(f"gradscf_geometry_jvp_{operator}")
    adjoint = core.Primitive(f"gradscf_geometry_vjp_{operator}")
    product = lambda env, denv: ffi_product("gradscf_geometry_jvp_cpu_v1", env, denv, shape)
    adjoint_product = lambda env, cot: ffi_product("gradscf_geometry_vjp_cpu_v1", env, cot, env.shape)
    linear.def_impl(product)
    adjoint.def_impl(adjoint_product)
    linear.def_abstract_eval(lambda env, denv: jax.core.ShapedArray(shape, jnp.float64))
    adjoint.def_abstract_eval(lambda env, cot: jax.core.ShapedArray(env.shape, jnp.float64))
    mlir.register_lowering(linear, mlir.lower_fun(product, multiple_results=False), platform="cpu")
    mlir.register_lowering(adjoint, mlir.lower_fun(adjoint_product, multiple_results=False), platform="cpu")

    def transpose(other, cotangent, env, vector):
        if ad.is_undefined_primal(env):
            raise NotImplementedError("Native second geometry derivatives are not supported")
        if not ad.is_undefined_primal(vector):
            return None, None
        if isinstance(cotangent, ad.Zero):
            return None, ad.Zero(vector.aval)
        return None, other.bind(env, cotangent)

    def derivative(primitive, primals, tangents):
        env, vector = primals
        denv, dvector = tangents
        # Only symbolic inactivity proves env is fixed. An active numerical
        # zero must still raise: no coordinate Hessian has been implemented.
        if not isinstance(denv, ad.Zero):
            raise NotImplementedError("Native second geometry derivatives are not supported")
        primal = primitive.bind(env, vector)
        tangent = (ad.Zero.from_primal_value(primal) if isinstance(dvector, ad.Zero)
                   else primitive.bind(env, dvector))
        return primal, tangent

    def batch(primitive, args, axes):
        size = next(a.shape[axis] for a, axis in zip(args, axes) if axis is not None)
        batched = tuple(batching.bdim_at_front(a, axis, size) for a, axis in zip(args, axes))
        return jax.lax.map(lambda xs: primitive.bind(*xs), batched), 0

    for primitive, other in ((linear, adjoint), (adjoint, linear)):
        ad.primitive_transposes[primitive] = partial(transpose, other)
        ad.primitive_jvps[primitive] = partial(derivative, primitive)
        batching.primitive_batchers[primitive] = partial(batch, primitive)

    @jax.custom_jvp
    def value(env, coords, origin):
        return evaluate_values(operator, atoms, basis, pack(env, coords, origin), nao, cart)

    @partial(value.defjvp, symbolic_zeros=True)
    def value_jvp(primals, tangents):
        env, coords, origin = primals
        denv, dcoords, dorigin = tangents
        if not isinstance(denv, SymbolicZero):
            raise NotImplementedError("Native geometry AD does not yet support basis exponent/coefficient derivatives")
        if isinstance(dcoords, SymbolicZero):
            dcoords = jnp.zeros_like(coords)
        if isinstance(dorigin, SymbolicZero):
            dorigin = jnp.zeros_like(origin)
        primal = value(env, coords, origin)
        tangent = linear.bind(pack(env, coords, origin),
                              pack(jnp.zeros_like(env), dcoords, dorigin))
        return primal, tangent

    return value


def evaluate_geometry(operator, atm, bas, env, coords, origin, nao, cart=True):
    """Evaluate with geometry AD; env contains only fixed nongeometry values."""
    atoms_key = tuple(tuple(int(x) for x in row) for row in atm)
    basis_key = tuple(tuple(int(x) for x in row) for row in bas)
    return _geometry_function(operator, atoms_key, basis_key, nao, cart)(env, coords, origin)
