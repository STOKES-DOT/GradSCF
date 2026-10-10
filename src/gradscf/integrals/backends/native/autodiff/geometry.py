"""Geometry AD using native shell contractions and JAX FFI.

Primal basis data are separate from differentiable coordinates so unsupported
basis derivatives cannot silently become zero. Coordinate Hessians use native
shell contractions for bilinear JVPs and their adjoints, without constructing
an integral-coordinate Jacobian. Third coordinate derivatives are rejected.
"""
from functools import lru_cache, partial

import jax
from jax.custom_derivatives import SymbolicZero
from jax.extend import core
from jax.interpreters import ad, batching, mlir
import jax.numpy as jnp
import numpy as np

from ..ffi import evaluate as evaluate_values, _OPERATORS
from .common import batch_product


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

    second = core.Primitive(f"gradscf_geometry_hessian_jvp_{operator}")
    second_adjoint = core.Primitive(f"gradscf_geometry_hessian_vjp_{operator}")

    def second_product(target, env, left, right, output_shape):
        call = jax.ffi.ffi_call(target, jax.ShapeDtypeStruct(output_shape, jnp.float64),
                                vmap_method="sequential")
        return call(jnp.asarray(atoms), jnp.asarray(basis), env, left, right,
                    operator=np.int32(_OPERATORS[operator]), cart=np.int32(cart))

    hessian_product = lambda env, u, v: second_product(
        "gradscf_geometry_hessian_jvp_cpu_v1", env, u, v, shape)
    hessian_adjoint = lambda env, u, cot: second_product(
        "gradscf_geometry_hessian_vjp_cpu_v1", env, u, cot, env.shape)
    second.def_impl(hessian_product)
    second_adjoint.def_impl(hessian_adjoint)
    second.def_abstract_eval(lambda env, u, v: jax.core.ShapedArray(shape, jnp.float64))
    second_adjoint.def_abstract_eval(lambda env, u, cot: jax.core.ShapedArray(env.shape, jnp.float64))
    mlir.register_lowering(second, mlir.lower_fun(hessian_product, multiple_results=False), platform="cpu")
    mlir.register_lowering(second_adjoint, mlir.lower_fun(hessian_adjoint, multiple_results=False), platform="cpu")

    def transpose(other, cotangent, env, vector):
        if ad.is_undefined_primal(env):
            raise NotImplementedError("Geometry transposition requires a fixed environment")
        if not ad.is_undefined_primal(vector):
            return None, None
        if isinstance(cotangent, ad.Zero):
            return None, ad.Zero(vector.aval)
        return None, other.bind(env, cotangent)

    def derivative(primitive, primals, tangents):
        env, vector = primals
        denv, dvector = tangents
        primal = primitive.bind(env, vector)
        terms = []
        if not isinstance(denv, ad.Zero):
            hessian = second if primitive is linear else second_adjoint
            terms.append(hessian.bind(env, denv, vector))
        if not isinstance(dvector, ad.Zero):
            terms.append(primitive.bind(env, dvector))
        tangent = sum(terms) if terms else ad.Zero.from_primal_value(primal)
        return primal, tangent

    def second_derivative(primitive, primals, tangents):
        env, left, right = primals
        denv, dleft, dright = tangents
        if not isinstance(denv, ad.Zero):
            raise NotImplementedError("Native third geometry derivatives are not supported")
        primal = primitive.bind(*primals)
        terms = []
        if not isinstance(dleft, ad.Zero):
            terms.append(primitive.bind(env, dleft, right))
        if not isinstance(dright, ad.Zero):
            terms.append(primitive.bind(env, left, dright))
        return primal, sum(terms) if terms else ad.Zero.from_primal_value(primal)

    def second_transpose(primitive, cotangent, env, left, right):
        if ad.is_undefined_primal(env):
            raise NotImplementedError("Native third geometry derivatives are not supported")
        unknown_left, unknown_right = map(ad.is_undefined_primal, (left, right))
        if unknown_left and unknown_right:
            raise NotImplementedError("A bilinear Hessian transpose requires one fixed argument")
        if isinstance(cotangent, ad.Zero):
            return (None, ad.Zero(left.aval) if unknown_left else None,
                    ad.Zero(right.aval) if unknown_right else None)
        if primitive is second:
            return (None, second_adjoint.bind(env, right, cotangent) if unknown_left else None,
                    second_adjoint.bind(env, left, cotangent) if unknown_right else None)
        # Hessian symmetry exchanges its coordinate slots. The integral
        # cotangent slot instead transposes back to the bilinear JVP.
        return (None, second_adjoint.bind(env, cotangent, right) if unknown_left else None,
                second.bind(env, left, cotangent) if unknown_right else None)

    for primitive, other in ((linear, adjoint), (adjoint, linear)):
        ad.primitive_transposes[primitive] = partial(transpose, other)
        ad.primitive_jvps[primitive] = partial(derivative, primitive)
        batching.primitive_batchers[primitive] = partial(batch_product, primitive)
    for primitive in (second, second_adjoint):
        ad.primitive_transposes[primitive] = partial(second_transpose, primitive)
        ad.primitive_jvps[primitive] = partial(second_derivative, primitive)
        batching.primitive_batchers[primitive] = partial(batch_product, primitive)

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
