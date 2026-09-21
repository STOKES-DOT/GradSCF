"""Native packed/RI integrals and direct J/K, without dense ERI buffers."""

from functools import lru_cache
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
from jax.extend import core
from jax.interpreters import ad, batching, mlir
from .._native import register_integrals


@dataclass(frozen=True)
class NativeDirectBasis:
    """Bound native J/K provider for the existing restricted direct-SCF hook."""

    plan: object
    parameters: object

    @property
    def nao(self):
        return self.plan.topology.nao

    def get_jk(self, density, *, screening_threshold=0.0):
        return self.plan.get_jk(
            self.parameters, density, screening_threshold=screening_threshold
        )

    def response_eri_pair_matrix(self):
        return self.plan.evaluate("eri", self.parameters, aosym="s4")


@dataclass(frozen=True)
class ProjectedNativeDirectBasis:
    """Exact contracted J/K from a fixed primitive shell-direct operator.

    The native call is differentiated only with respect to the primitive AO
    density.  JAX owns both contractions with ``transform``, so contraction
    coefficient derivatives do not require basis derivatives in libcint.
    """

    primitive_basis: NativeDirectBasis
    transform: object

    @property
    def nao(self):
        return self.transform.shape[1]

    def get_jk(self, density, *, screening_threshold=0.0):
        transform = jnp.asarray(self.transform)
        primitive_density = jnp.einsum(
            "pi,...ij,qj->...pq", transform, density, transform
        )
        fixed_parameters = jax.tree.map(
            jax.lax.stop_gradient, self.primitive_basis.parameters
        )
        primitive_j, primitive_k = self.primitive_basis.plan.get_jk(
            fixed_parameters, primitive_density,
            screening_threshold=screening_threshold,
        )
        project = lambda matrix: jnp.einsum(
            "pi,...pq,qj->...ij", transform, matrix, transform
        )
        return project(primitive_j), project(primitive_k)


def evaluate(atm, bas, env, shape, *, layout, cart, split=0):
    register_integrals()
    if jnp.asarray(env).dtype != jnp.float64:
        raise ValueError("Native compact integrals require float64.")
    call = jax.ffi.ffi_call(
        "gradscf_compact_cpu_v1",
        jax.ShapeDtypeStruct(shape, jnp.float64),
        vmap_method="sequential",
    )
    return call(
        jnp.asarray(atm),
        jnp.asarray(bas),
        env,
        layout=np.int32(layout),
        cart=np.int32(cart),
        split=np.int32(split),
    )


@lru_cache(maxsize=32)
def _direct_operator(atoms_key, basis_key, cart, cutoff):
    atoms = np.asarray(atoms_key, dtype=np.int32)
    basis = np.asarray(basis_key, dtype=np.int32)
    primitive = core.Primitive("gradscf_direct_jk")

    def apply(env, density):
        register_integrals()
        call = jax.ffi.ffi_call(
            "gradscf_direct_jk_cpu_v1",
            jax.ShapeDtypeStruct((2, *density.shape), jnp.float64),
            vmap_method="sequential",
        )
        return call(
            jnp.asarray(atoms),
            jnp.asarray(basis),
            env,
            density,
            cart=np.int32(cart),
            cutoff=np.float64(cutoff),
        )

    primitive.def_impl(apply)
    primitive.def_abstract_eval(
        lambda env, d: jax.core.ShapedArray((2, *d.shape), jnp.float64)
    )
    mlir.register_lowering(
        primitive, mlir.lower_fun(apply, multiple_results=False), platform="cpu"
    )

    def jvp(primals, tangents):
        env, d = primals
        de, dd = tangents
        if not isinstance(de, ad.Zero):
            raise NotImplementedError(
                "Direct J/K supports density AD only; use the RI contraction path for coefficients."
            )
        value = primitive.bind(env, d)
        return value, (
            ad.Zero(jax.core.ShapedArray(value.shape, value.dtype))
            if isinstance(dd, ad.Zero)
            else primitive.bind(env, dd)
        )

    def transpose(cot, env, d):
        if ad.is_undefined_primal(env):
            raise NotImplementedError(
                "Direct J/K basis/geometry VJP is not implemented."
            )
        if not ad.is_undefined_primal(d):
            return None, None
        if isinstance(cot, ad.Zero):
            return None, ad.Zero(d.aval)
        nset = cot.shape[1]
        # Both Coulomb and exchange are self-adjoint linear density maps.
        value = primitive.bind(env, jnp.concatenate((cot[0], cot[1]), axis=0))
        return None, value[0, :nset] + value[1, nset:]

    def batch(args, axes):
        size = next(x.shape[a] for x, a in zip(args, axes) if a is not None)
        arrays = tuple(batching.bdim_at_front(x, a, size) for x, a in zip(args, axes))
        return jax.lax.map(lambda xs: primitive.bind(*xs), arrays), 0

    ad.primitive_jvps[primitive] = jvp
    ad.primitive_transposes[primitive] = transpose
    batching.primitive_batchers[primitive] = batch
    return primitive


def direct_jk(atm, bas, env, density, *, cart, cutoff=0.0):
    d = jnp.asarray(density)
    n = d.shape[-1]
    if d.shape[-2:] != (n, n) or d.dtype not in (jnp.float64, jnp.complex128):
        raise ValueError("Direct J/K requires (...,n,n) float64/complex128 density.")
    env = jnp.asarray(env)
    if env.dtype != jnp.float64:
        raise ValueError("Direct J/K requires float64 environment.")
    operator = _direct_operator(
        tuple(map(tuple, np.asarray(atm).tolist())),
        tuple(map(tuple, np.asarray(bas).tolist())),
        bool(cart),
        float(cutoff),
    )
    flat = d.reshape(-1, n, n)
    count = len(flat)
    real = (
        jnp.concatenate((flat.real, flat.imag), axis=0) if jnp.iscomplexobj(d) else flat
    )
    out = operator.bind(env, real)
    if jnp.iscomplexobj(d):
        out = out[:, :count] + 1j * out[:, count:]
    out = out.reshape(2, *d.shape)
    return out[0], out[1]


@lru_cache(maxsize=1)
def _packed_operator():
    from ..layouts import _build_jk_from_packed_jax

    primitive = core.Primitive("gradscf_packed_jk")
    reference = lambda e, d: jnp.stack(_build_jk_from_packed_jax(e, d))

    def apply(eri, density):
        register_integrals()
        call = jax.ffi.ffi_call(
            "gradscf_packed_jk_cpu_v1",
            jax.ShapeDtypeStruct((2, *density.shape), jnp.float64),
            vmap_method="sequential",
        )
        return call(eri, density)

    primitive.def_impl(apply)
    primitive.def_abstract_eval(
        lambda e, d: jax.core.ShapedArray((2, *d.shape), jnp.float64)
    )
    mlir.register_lowering(primitive, mlir.lower_fun(reference, multiple_results=False))
    mlir.register_lowering(
        primitive, mlir.lower_fun(apply, multiple_results=False), platform="cpu"
    )

    def jvp(primals, tangents):
        e, d = primals
        de, dd = tangents
        value = primitive.bind(e, d)
        tangent = ad.Zero(jax.core.ShapedArray(value.shape, value.dtype))
        if not isinstance(de, ad.Zero):
            tangent = primitive.bind(de, d)
        if not isinstance(dd, ad.Zero):
            part = primitive.bind(e, dd)
            tangent = part if isinstance(tangent, ad.Zero) else tangent + part
        return value, tangent

    def transpose(cot, e, d):
        if ad.is_undefined_primal(e):
            if ad.is_undefined_primal(d):
                raise ValueError("A bilinear J/K transpose needs one known primal.")
            if isinstance(cot, ad.Zero):
                return ad.Zero(e.aval), None
            # The ERI cotangent uses the portable bounded scatter path;
            # density derivatives use the fast self-adjoint native operator.
            _, pullback = jax.vjp(
                lambda x: reference(x, d), jnp.zeros(e.aval.shape, e.aval.dtype)
            )
            return pullback(cot)[0], None
        if not ad.is_undefined_primal(d):
            return None, None
        if isinstance(cot, ad.Zero):
            return None, ad.Zero(d.aval)
        count = cot.shape[1]
        value = primitive.bind(e, jnp.concatenate((cot[0], cot[1]), axis=0))
        return None, value[0, :count] + value[1, count:]

    def batch(args, axes):
        size = next(x.shape[a] for x, a in zip(args, axes) if a is not None)
        arrays = tuple(batching.bdim_at_front(x, a, size) for x, a in zip(args, axes))
        return jax.lax.map(lambda xs: primitive.bind(*xs), arrays), 0

    ad.primitive_jvps[primitive] = jvp
    ad.primitive_transposes[primitive] = transpose
    batching.primitive_batchers[primitive] = batch
    return primitive


def packed_jk(eri, density):
    """Native s8 value/density AD; ERI VJP uses bounded JAX scatters."""
    if eri.ndim != 1:
        raise ValueError("The native packed J/K operator requires s8 ERIs.")
    n = density.shape[-1]
    flat = density.reshape(-1, n, n)
    count = len(flat)
    d = (
        jnp.concatenate((flat.real, flat.imag), axis=0)
        if jnp.iscomplexobj(flat)
        else flat
    )
    out = _packed_operator().bind(eri, d)
    if jnp.iscomplexobj(flat):
        out = out[:, :count] + 1j * out[:, count:]
    out = out.reshape(2, *density.shape)
    return out[0], out[1]
