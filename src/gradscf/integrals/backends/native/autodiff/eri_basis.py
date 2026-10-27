"""Full/selected libcint ERI coefficient products and exponent JVP/VJP.

Only shell-local contractions are differentiated in native code. JAX owns raw
coefficient/primitive normalization. Pure coordinate products retain the existing
geometry family; higher exponent and mixed basis/geometry products fail closed.
"""
from functools import lru_cache, partial
import operator

import jax
from jax.custom_derivatives import SymbolicZero
from jax.extend import core
from jax.interpreters import ad, batching, mlir
import jax.numpy as jnp
import numpy as np

from ...._native import register_integrals
from ..ffi import _check_eri_capacity, _eri_shell_slice, evaluate as evaluate_values
from .common import batch_product, register_cpu_product, register_eager_dispatch
from .geometry import _geometry_function


def _products(atoms, basis, shape, cart, ncoeff, nexponents, shell_slice):
    active = basis if shell_slice is None else basis[np.unique(np.concatenate([
        np.arange(first, last) for first, last in zip(shell_slice[::2], shell_slice[1::2])]))]
    prefix = "eri" if shell_slice is None else "eri_slice"
    attributes = {} if shell_slice is None else {"shell_slice": np.asarray(shell_slice, np.int32)}

    def workspace(kind):
        angular = active[:, 1].astype(np.int64)
        ordinary = int(np.max(active[:, 3]*((angular+1)*(angular+2)//2)))
        if kind == "exponent":
            if np.any(angular > 10):
                raise NotImplementedError("ERI exponent angular raising requires l <= 10")
            largest = int(np.max(active[:, 3]*((angular+3)*(angular+4)//2)))
        elif kind == "adjoint":
            largest = int(np.max(active[:, 2]*((angular+1)*(angular+2)//2)))
        else:
            largest = ordinary
        if 3*largest*ordinary**3 >= np.iinfo(np.int32).max:
            raise ValueError("ERI basis internal Cartesian cache exceeds int32 indexing")

    def make(name, suffix, result_shape, kind):
        primitive = core.Primitive(f"gradscf_{prefix}_{name}_{suffix}")

        def call(*args):
            workspace(kind)
            register_integrals()
            target = f"gradscf_{prefix}_{name}_{suffix}_cpu_v1"
            invoke = jax.ffi.ffi_call(target, jax.ShapeDtypeStruct(result_shape, jnp.float64),
                                      vmap_method="sequential")
            return invoke(jnp.asarray(atoms), jnp.asarray(basis), *args,
                          cart=np.int32(cart), **attributes)

        def abstract(*args):
            workspace(kind)
            return jax.core.ShapedArray(result_shape, jnp.float64)

        if shell_slice is None:
            register_eager_dispatch(primitive)
            mlir.register_lowering(primitive, mlir.lower_fun(call, multiple_results=False), platform="cpu")
        else:
            register_cpu_product(primitive, call, lambda *args: result_shape)
        primitive.def_abstract_eval(abstract)
        batching.primitive_batchers[primitive] = partial(batch_product, primitive)
        return primitive

    linear = make("coefficient", "jvp", shape, "forward")
    adjoint = make("coefficient", "vjp", (ncoeff,), "adjoint")
    second = make("coefficient", "hessian_jvp", shape, "forward")
    second_adjoint = make("coefficient", "hessian_vjp", (ncoeff,), "adjoint")
    exponent = make("exponent", "jvp", shape, "exponent")
    exponent_adjoint = make("exponent", "vjp", (nexponents,), "exponent")

    def coefficient_jvp(primitive, primals, tangents):
        env, coefficients, vector = primals
        denv, dc, dv = tangents
        if not isinstance(denv, ad.Zero):
            raise NotImplementedError("Mixed ERI coefficient/exponent or coefficient/geometry products are not supported")
        primal = primitive.bind(*primals)
        terms = []
        if not isinstance(dc, ad.Zero):
            hessian = second if primitive is linear else second_adjoint
            terms.append(hessian.bind(env, coefficients, dc, vector))
        if not isinstance(dv, ad.Zero):
            terms.append(primitive.bind(env, coefficients, dv))
        return primal, sum(terms) if terms else ad.Zero(jax.typeof(primal).to_tangent_aval())

    def coefficient_transpose(other, cotangent, env, coefficients, vector):
        if ad.is_undefined_primal(env) or ad.is_undefined_primal(coefficients):
            raise NotImplementedError("ERI coefficient transpose requires primal coefficients and fixed environment")
        if not ad.is_undefined_primal(vector):
            return None, None, None
        tangent = ad.Zero(vector.aval) if isinstance(cotangent, ad.Zero) else other.bind(env, coefficients, cotangent)
        return None, None, tangent

    def second_jvp(primitive, primals, tangents):
        env, coefficients, left, right = primals
        denv, dc, dl, dr = tangents
        # A four-center coefficient Hessian is quadratic in the remaining
        # primal coefficients. Its derivative is not zero, unlike 1e/3c.
        if not isinstance(denv, ad.Zero) or not isinstance(dc, ad.Zero):
            raise NotImplementedError("Third or mixed ERI coefficient products are not supported")
        primal = primitive.bind(*primals)
        terms = []
        if not isinstance(dl, ad.Zero):
            terms.append(primitive.bind(env, coefficients, dl, right))
        if not isinstance(dr, ad.Zero):
            terms.append(primitive.bind(env, coefficients, left, dr))
        return primal, sum(terms) if terms else ad.Zero(jax.typeof(primal).to_tangent_aval())

    def second_transpose(primitive, cotangent, env, coefficients, left, right):
        if ad.is_undefined_primal(env) or ad.is_undefined_primal(coefficients):
            raise NotImplementedError("ERI coefficient Hessian transpose requires its primal coefficients/environment")
        unknown_left, unknown_right = map(ad.is_undefined_primal, (left, right))
        if unknown_left and unknown_right:
            raise NotImplementedError("An ERI Hessian transpose requires one fixed direction")
        if isinstance(cotangent, ad.Zero):
            return (None, None, ad.Zero(left.aval) if unknown_left else None,
                    ad.Zero(right.aval) if unknown_right else None)
        if primitive is second:
            return (None, None,
                    second_adjoint.bind(env, coefficients, right, cotangent) if unknown_left else None,
                    second_adjoint.bind(env, coefficients, left, cotangent) if unknown_right else None)
        return (None, None,
                second_adjoint.bind(env, coefficients, cotangent, right) if unknown_left else None,
                second.bind(env, coefficients, left, cotangent) if unknown_right else None)

    def exponent_jvp(primitive, primals, tangents):
        env, vector = primals
        denv, dv = tangents
        if not isinstance(denv, ad.Zero):
            raise NotImplementedError("Higher or mixed ERI exponent products are not supported")
        primal = primitive.bind(*primals)
        tangent = (ad.Zero(jax.typeof(primal).to_tangent_aval()) if isinstance(dv, ad.Zero)
                   else primitive.bind(env, dv))
        return primal, tangent

    def exponent_transpose(other, cotangent, env, vector):
        if ad.is_undefined_primal(env):
            raise NotImplementedError("Higher or mixed ERI exponent products are not supported")
        if not ad.is_undefined_primal(vector):
            return None, None
        return None, ad.Zero(vector.aval) if isinstance(cotangent, ad.Zero) else other.bind(env, cotangent)

    for primitive, other in ((linear, adjoint), (adjoint, linear)):
        ad.primitive_jvps[primitive] = partial(coefficient_jvp, primitive)
        ad.primitive_transposes[primitive] = partial(coefficient_transpose, other)
    for primitive in (second, second_adjoint):
        ad.primitive_jvps[primitive] = partial(second_jvp, primitive)
        ad.primitive_transposes[primitive] = partial(second_transpose, primitive)
    for primitive, other in ((exponent, exponent_adjoint), (exponent_adjoint, exponent)):
        ad.primitive_jvps[primitive] = partial(exponent_jvp, primitive)
        ad.primitive_transposes[primitive] = partial(exponent_transpose, other)
    return linear, exponent


@lru_cache(maxsize=64)
def _eri_function(atoms_key, basis_key, nao, cart, shell_slice):
    atoms, basis = np.asarray(atoms_key, np.int32), np.asarray(basis_key, np.int32)
    coordinates = atoms[:, 1, None]+np.arange(3)[None, :]
    coefficients = np.concatenate([np.arange(int(b[6]), int(b[6])+int(b[2])*int(b[3])) for b in basis])
    exponents = np.concatenate([np.arange(int(b[5]), int(b[5])+int(b[2])) for b in basis])
    groups = (coordinates.ravel(), coefficients, exponents, np.arange(1, 4))
    if any(np.unique(group).size != group.size for group in groups) or any(
            np.intersect1d(left, right).size for i, left in enumerate(groups) for right in groups[i+1:]):
        raise ValueError("Native ERI coordinate, coefficient and exponent storage must be independent")
    _, shape = _eri_shell_slice(basis, cart, shell_slice)
    shape = (nao,)*4 if shape is None else shape
    linear, exponent = _products(atoms, basis, shape, cart, len(coefficients), len(exponents), shell_slice)
    geometry = _geometry_function("eri", atoms_key, basis_key, nao, cart, shell_slice=shell_slice)

    def pack_coefficients(env, c):
        return env.at[coefficients].set(c, unique_indices=True)

    def pack_exponents(env, a):
        return env.at[exponents].set(a, unique_indices=True)

    def pack_geometry(env, coords, origin):
        return env.at[coordinates].set(coords, unique_indices=True).at[1:4].set(origin)

    @jax.custom_jvp
    def value(env, c, coords, origin, a):
        fixed = pack_coefficients(pack_exponents(env, a), c)
        if shell_slice is not None:
            # Reuse the existing bounded CPU/CUDA value dispatch; no full ERI
            # value or product is formed before selecting this shell window.
            return geometry(fixed, coords, origin)
        packed = pack_geometry(fixed, coords, origin)
        return evaluate_values("eri", atoms, basis, packed, nao, cart)

    @partial(value.defjvp, symbolic_zeros=True)
    def value_jvp(primals, tangents):
        env, c, coords, origin, a = primals
        denv, dc, dcoords, dorigin, da = tangents
        if not isinstance(denv, SymbolicZero):
            raise NotImplementedError("Native ERI basis AD requires fixed control data")
        primal = value(*primals)
        fixed = pack_exponents(env, a)
        terms = []
        if not isinstance(dc, SymbolicZero):
            terms.append(linear.bind(pack_geometry(fixed, coords, origin), c, dc))
        if not isinstance(da, SymbolicZero):
            terms.append(exponent.bind(pack_geometry(pack_coefficients(fixed, c), coords, origin), da))
        if not (isinstance(dcoords, SymbolicZero) and isinstance(dorigin, SymbolicZero)):
            dr = jnp.zeros_like(coords) if isinstance(dcoords, SymbolicZero) else dcoords
            do = jnp.zeros_like(origin) if isinstance(dorigin, SymbolicZero) else dorigin
            terms.append(jax.jvp(lambda r, o: geometry(pack_coefficients(fixed, c), r, o),
                                 (coords, origin), (dr, do))[1])
        return primal, sum(terms) if terms else jnp.zeros_like(primal)

    def apply(env, c, coords, origin, a):
        a = jax.lax.stop_gradient(env[exponents]) if a is None else a
        return value(env, c, coords, origin, a)

    return apply


def evaluate_eri_basis(atm, bas, fixed_env, coefficients, coords, origin, nao,
                       cart=True, *, exponents=None, shell_slice=None):
    """Full or shell-selected s1 ERIs with dynamic coefficients/exponents.

    Tables are immutable topology; all physical values remain explicit dynamic
    inputs. Active coefficient slots in ``fixed_env`` are placeholders, and
    ``exponents`` are packed separately when supplied. Full ERI products retain
    their existing CPU-only execution boundary; bounded windows reuse the
    bounded CPU/CUDA bridge.
    """
    if not isinstance(cart, (bool, np.bool_)):
        raise TypeError("cart must be a static bool")
    nao = operator.index(nao)
    atoms, basis = np.asarray(atm), np.asarray(bas)
    for name, table, width in (("atm", atoms, 6), ("bas", basis, 8)):
        if table.dtype != np.int32 or table.ndim != 2 or table.shape[1] != width or not len(table):
            raise ValueError(f"{name} must be a nonempty int32 array with shape (n, {width})")
    angular = basis[:, 1].astype(np.int64)
    if np.any(angular < 0) or np.any(angular > 12):
        raise ValueError("Native angular momentum must lie in 0..12")
    dims = basis[:, 3]*((angular+1)*(angular+2)//2 if cart else 2*angular+1)
    if nao < 1 or int(dims.sum()) != nao:
        raise ValueError("nao differs from the supplied basis AO count")
    shell_slice, _ = _eri_shell_slice(basis, cart, shell_slice)
    if shell_slice is None:
        _check_eri_capacity(basis, cart)
    fixed_env, coefficients, coords, origin = map(jnp.asarray, (fixed_env, coefficients, coords, origin))
    ncoeff = int(np.sum(basis[:, 2].astype(np.int64)*basis[:, 3]))
    for name, array, shape in (("coefficients", coefficients, (ncoeff,)),
                               ("coords", coords, (len(atoms), 3)), ("origin", origin, (3,))):
        if array.dtype != jnp.float64 or array.shape != shape:
            raise ValueError(f"{name} must have float64 dtype and shape {shape}")
    if fixed_env.dtype != jnp.float64 or fixed_env.ndim != 1 or fixed_env.size < 20:
        raise ValueError("fixed_env must be a float64 vector of at least 20 entries")
    if exponents is not None:
        exponents = jnp.asarray(exponents)
        if exponents.dtype != jnp.float64 or exponents.shape != (int(basis[:, 2].sum()),):
            raise ValueError("Exponents must be a float64 vector matching all primitives")
    return _eri_function(tuple(map(tuple, atoms.tolist())), tuple(map(tuple, basis.tolist())),
                         nao, bool(cart), shell_slice)(fixed_env, coefficients, coords, origin, exponents)
