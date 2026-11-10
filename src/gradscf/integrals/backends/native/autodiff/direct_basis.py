"""Streamed direct J/K basis and mixed geometry products with explicit operands.

JAX owns raw-coefficient and primitive normalization. Native products contract
one shell at a time. Pure/cross basis Hessians and third derivatives fail
explicitly; first/second geometry products retain the shared screened kernels.
"""
from functools import lru_cache, partial

import jax
from jax.custom_derivatives import SymbolicZero
from jax.extend import core
from jax.interpreters import ad, batching, mlir
import jax.numpy as jnp
import numpy as np

from ...._native import register_integrals
from .common import batch_product, register_cpu_product
from .direct_jk import geometry_products


@lru_cache(maxsize=32)
def basis_value(atoms_key, basis_key, cart, cutoff, workers, with_j, with_k,
                private_output_bytes, screened):
    from ..jk import _density_transpose, _direct_operator, _zero_basis_tangent

    atoms, basis = np.asarray(atoms_key, np.int32), np.asarray(basis_key, np.int32)
    coordinates = atoms[:, 1, None]+np.arange(3)[None, :]
    coefficients = np.concatenate([np.arange(int(b[6]), int(b[6])+int(b[2])*int(b[3])) for b in basis])
    exponents = np.concatenate([np.arange(int(b[5]), int(b[5])+int(b[2])) for b in basis])
    groups = (coordinates.ravel(), coefficients, exponents, np.arange(20))
    if any(np.unique(group).size != group.size for group in groups) or any(
            np.intersect1d(left, right).size for i, left in enumerate(groups) for right in groups[i+1:]):
        raise ValueError("Direct coordinate, coefficient, exponent and control storage must be independent")
    nops = int(with_j)+int(with_k)
    operator = _direct_operator(atoms_key, basis_key, cart, cutoff, workers,
                                with_j, with_k, private_output_bytes, screened)
    geometry = geometry_products(atoms_key, basis_key, cart, with_j, with_k,
                                 workers=workers, cutoff=cutoff)

    def pack(env, c, a, r):
        return (env.at[coefficients].set(c, unique_indices=True)
                .at[exponents].set(a, unique_indices=True)
                .at[coordinates].set(r, unique_indices=True))

    def geometry_direction(env, vector):
        return jnp.zeros_like(env).at[coordinates].set(vector, unique_indices=True)

    def guard(tangents, message):
        result = 0.
        for tangent in tangents:
            if not isinstance(tangent, (ad.Zero, SymbolicZero)):
                result = result + _zero_basis_tangent(message).bind(tangent)
        return result

    mixed = {}
    for kind, size in (("coefficient", len(coefficients)), ("exponent", len(exponents))):
        products = tuple(core.Primitive(f"gradscf_direct_jk_{kind}_geometry_{mode}")
                         for mode in ("jvp", "parameter_vjp", "coordinate_vjp"))
        mixed[kind] = products

        def mixed_call(kind, size, mode, env, c, a, r, density, reference, left, right):
            angular = basis[:, 1].astype(np.int64)
            ordinary = int(np.max(basis[:, 3]*((angular+1)*(angular+2)//2)))
            if kind == "exponent":
                if np.any(angular > 10):
                    raise NotImplementedError("Direct exponent angular raising requires l <= 10")
                largest = int(np.max(basis[:, 3]*((angular+3)*(angular+4)//2)))
            else:
                largest = int(np.max(np.maximum(basis[:, 2], basis[:, 3])
                                     *((angular+1)*(angular+2)//2)))
            if 9*largest*ordinary**3 >= np.iinfo(np.int32).max:
                raise ValueError("Direct mixed Cartesian shell cache exceeds int32 indexing")
            register_integrals()
            shape = (nops, *density.shape) if mode == 0 else ((size,) if mode == 1 else env.shape)
            function = jax.ffi.ffi_call(f"gradscf_direct_jk_{kind}_geometry_cpu_v1",
                jax.ShapeDtypeStruct(shape, jnp.float64), vmap_method="sequential")
            operands = (c,) if kind == "coefficient" else ()
            coordinate = geometry_direction(env, left) if mode < 2 else left
            result = function(jnp.asarray(atoms), jnp.asarray(basis), pack(env, c, a, r),
                              *operands, density, reference, coordinate, right,
                              cart=np.int32(cart), cutoff=np.float64(cutoff),
                              with_j=np.int32(with_j), with_k=np.int32(with_k),
                              workers=np.int32(workers), mode=np.int32(mode))
            return result[coordinates] if mode == 2 else result

        def mixed_derivative(primitive, primals, tangents):
            env, c, a, r, density, reference, left, right = primals
            denv, dc, da, dr, dd, _, dl, dv = tangents
            primal = primitive.bind(*primals)
            zero = guard((denv, dc, da, dr),
                         "Direct J/K third or higher mixed basis derivatives are not implemented")
            terms = [jnp.zeros_like(primal)+zero]
            for index, tangent in ((4, dd), (6, dl), (7, dv)):
                if not isinstance(tangent, ad.Zero):
                    changed = list(primals)
                    changed[index] = tangent
                    terms.append(primitive.bind(*changed))
            return primal, sum(terms)

        def mixed_transpose(products, mode, cotangent, env, c, a, r, density, reference, left, right):
            if any(ad.is_undefined_primal(x) for x in (env, c, a, r, reference)):
                raise NotImplementedError("Direct mixed transpose requires its primal physical inputs")
            unknown = [index for index, x in ((4, density), (6, left), (7, right))
                       if ad.is_undefined_primal(x)]
            result = [None]*8
            if not unknown:
                return tuple(result)
            if len(unknown) != 1:
                raise NotImplementedError("A direct mixed transpose needs one unknown multilinear argument")
            index = unknown[0]
            if isinstance(cotangent, ad.Zero):
                result[index] = ad.Zero({4: density, 6: left, 7: right}[index].aval)
                return tuple(result)
            base = (env, c, a, r, density, reference)
            push, parameter_pull, coordinate_pull = products
            if index == 4:
                u, v, weights = ((left, right, cotangent) if mode == 0 else
                                  (left, cotangent, right) if mode == 1 else
                                  (cotangent, left, right))
                result[4] = _density_transpose(lambda d: push.bind(
                    env, c, a, r, d, reference, u, v), weights)
            elif mode == 0:
                result[index] = (coordinate_pull.bind(*base, right, cotangent) if index == 6
                                 else parameter_pull.bind(*base, left, cotangent))
            elif mode == 1:
                result[index] = (coordinate_pull.bind(*base, cotangent, right) if index == 6
                                 else push.bind(*base, left, cotangent))
            else:
                result[index] = (parameter_pull.bind(*base, cotangent, right) if index == 6
                                 else push.bind(*base, cotangent, left))
            return tuple(result)

        for mode, primitive in enumerate(products):
            def shape(env, c, a, r, density, reference, left, right, mode=mode, size=size):
                return (nops, *density.shape) if mode == 0 else ((size,) if mode == 1 else r.shape)
            register_cpu_product(primitive, partial(mixed_call, kind, size, mode), shape)
            batching.primitive_batchers[primitive] = partial(batch_product, primitive)
            ad.primitive_jvps[primitive] = partial(mixed_derivative, primitive)
            ad.primitive_transposes[primitive] = partial(mixed_transpose, products, mode)

    geometry_push, geometry_pull = {}, {}
    for order in (1, 2):
        push = core.Primitive(f"gradscf_direct_jk_physical_geometry_order{order}_jvp")
        pull = core.Primitive(f"gradscf_direct_jk_physical_geometry_order{order}_vjp")
        geometry_push[order], geometry_pull[order] = push, pull

        def geometry_call(is_pull, order, env, c, a, r, density, reference, *vectors):
            # Reuse existing CPU/screened geometry kernels. AD of this wrapper
            # is registered below and never exposes packed ENV as its state.
            def push(*directions):
                if order == 1:
                    return geometry.bind(pack(env, c, a, r), density, reference,
                                         geometry_direction(env, directions[0]))
                return jax.jvp(lambda q: geometry.bind(pack(env, c, a, q), density, reference,
                    geometry_direction(env, directions[1])), (r,), (directions[0],))[1]
            if is_pull:
                return jax.vjp(lambda v: push(*vectors[:-1], v), jnp.zeros_like(r))[1](vectors[-1])[0]
            return push(*vectors)

        register_cpu_product(push, partial(geometry_call, False, order),
                             lambda env, c, a, r, density, reference, *vectors: (nops, *density.shape))
        register_cpu_product(pull, partial(geometry_call, True, order),
                             lambda env, c, a, r, *args: r.shape)
        for primitive in (push, pull):
            batching.primitive_batchers[primitive] = partial(batch_product, primitive)

    def geometry_derivative(order, is_pull, primals, tangents):
        env, c, a, r, density, reference, *vectors = primals
        denv, dc, da, dr, dd, _, *dvectors = tangents
        family = geometry_pull if is_pull else geometry_push
        primitive = family[order]
        primal = primitive.bind(*primals)
        terms = [jnp.zeros_like(primal)+guard((denv,),
                 "Direct geometry AD requires fixed environment control data")]
        base = (env, c, a, r, density, reference)
        for kind, tangent in (("coefficient", dc), ("exponent", da)):
            if not isinstance(tangent, ad.Zero):
                if order == 2:
                    terms.append(jnp.zeros_like(primal)+guard((tangent,),
                        "Direct J/K third mixed geometry/basis derivatives are not implemented"))
                elif is_pull:
                    terms.append(mixed[kind][2].bind(*base, tangent, vectors[-1]))
                else:
                    terms.append(mixed[kind][0].bind(*base, vectors[0], tangent))
        if not isinstance(dr, ad.Zero):
            if order == 2:
                terms.append(jnp.zeros_like(primal)+guard((dr,),
                    "Native direct J/K third geometry derivatives are not supported"))
            else:
                terms.append(family[2].bind(*base, dr, *vectors))
        if not isinstance(dd, ad.Zero):
            terms.append(primitive.bind(env, c, a, r, dd, reference, *vectors))
        for index, tangent in enumerate(dvectors):
            if not isinstance(tangent, ad.Zero):
                changed = list(vectors)
                changed[index] = tangent
                terms.append(primitive.bind(*base, *changed))
        return primal, sum(terms)

    def geometry_transpose(order, is_pull, cotangent, env, c, a, r, density, reference, *vectors):
        if any(ad.is_undefined_primal(x) for x in (env, c, a, r, reference)):
            raise NotImplementedError("Direct geometry transpose requires its primal physical inputs")
        arguments = (density, *vectors)
        unknown = [i for i, x in enumerate(arguments) if ad.is_undefined_primal(x)]
        result = [None]*(6+len(vectors))
        if not unknown:
            return tuple(result)
        if len(unknown) != 1:
            raise NotImplementedError("A direct geometry transpose needs one unknown multilinear argument")
        index = unknown[0]
        destination = 4 if index == 0 else 5+index
        base = (env, c, a, r, density, reference)
        if isinstance(cotangent, ad.Zero):
            result[destination] = ad.Zero(arguments[index].aval)
        elif not is_pull:
            if index == 0:
                result[4] = _density_transpose(lambda d: geometry_push[order].bind(
                    env, c, a, r, d, reference, *vectors), cotangent)
            else:
                fixed = [v for i, v in enumerate(vectors) if i != index-1]
                result[destination] = geometry_pull[order].bind(*base, *fixed, cotangent)
        elif index == 0:
            result[4] = _density_transpose(lambda d: geometry_push[order].bind(
                env, c, a, r, d, reference, *vectors[:-1], cotangent), vectors[-1])
        elif index == len(arguments)-1:
            result[destination] = geometry_push[order].bind(*base, *vectors[:-1], cotangent)
        else:
            result[destination] = geometry_pull[order].bind(*base, cotangent, vectors[-1])
        return tuple(result)

    for order in (1, 2):
        for primitive, is_pull in ((geometry_push[order], False), (geometry_pull[order], True)):
            ad.primitive_jvps[primitive] = partial(geometry_derivative, order, is_pull)
            ad.primitive_transposes[primitive] = partial(geometry_transpose, order, is_pull)

    families = {}
    for kind, size in (("coefficient", len(coefficients)), ("exponent", len(exponents))):
        push = core.Primitive(f"gradscf_direct_jk_{kind}_jvp")
        pull = core.Primitive(f"gradscf_direct_jk_{kind}_vjp")
        families[kind] = push, pull

        def call(transpose, kind, size, env, c, a, r, density, reference, vector):
            angular = basis[:, 1].astype(np.int64)
            ordinary = int(np.max(basis[:, 3]*((angular+1)*(angular+2)//2)))
            if kind == "exponent":
                if np.any(angular > 10):
                    raise NotImplementedError("Direct exponent angular raising requires l <= 10")
                largest = int(np.max(basis[:, 3]*((angular+3)*(angular+4)//2)))
            elif transpose:
                largest = int(np.max(basis[:, 2]*((angular+1)*(angular+2)//2)))
            else:
                largest = ordinary
            if 3*largest*ordinary**3 >= np.iinfo(np.int32).max:
                raise ValueError("Direct basis Cartesian shell cache exceeds int32 indexing")
            register_integrals()
            suffix = "vjp" if transpose else "jvp"
            shape = (size,) if transpose else (nops, *density.shape)
            function = jax.ffi.ffi_call(f"gradscf_direct_jk_{kind}_{suffix}_cpu_v1",
                jax.ShapeDtypeStruct(shape, jnp.float64), vmap_method="sequential")
            operands = (c,) if kind == "coefficient" else ()
            return function(jnp.asarray(atoms), jnp.asarray(basis), pack(env, c, a, r),
                            *operands, density, reference, vector,
                            cart=np.int32(cart), cutoff=np.float64(cutoff),
                            with_j=np.int32(with_j), with_k=np.int32(with_k),
                            workers=np.int32(workers))

        register_cpu_product(push, partial(call, False, kind, size),
                             lambda env, c, a, r, density, reference, vector: (nops, *density.shape))
        register_cpu_product(pull, partial(call, True, kind, size),
                             lambda *args, size=size: (size,))
        for primitive in (push, pull):
            batching.primitive_batchers[primitive] = partial(batch_product, primitive)

        def derivative(primitive, kind, is_pull, primals, tangents):
            env, c, a, r, density, reference, vector = primals
            denv, dc, da, dr, dd, _, dv = tangents
            primal = primitive.bind(*primals)
            zero = guard((denv, dc, da),
                         "Direct J/K pure or cross basis second derivatives are not implemented")
            terms = [jnp.zeros_like(primal)+zero]
            if not isinstance(dr, ad.Zero):
                terms.append(mixed[kind][1 if is_pull else 0].bind(
                    env, c, a, r, density, reference, dr, vector))
            if not isinstance(dd, ad.Zero):
                terms.append(primitive.bind(env, c, a, r, dd, reference, vector))
            if not isinstance(dv, ad.Zero):
                terms.append(primitive.bind(env, c, a, r, density, reference, dv))
            return primal, sum(terms)

        def transpose(is_pull, push, pull, cotangent, env, c, a, r, density, reference, vector):
            if any(ad.is_undefined_primal(x) for x in (env, c, a, r, reference)):
                raise NotImplementedError("Direct basis transposition requires its primal physical inputs")
            unknown_d, unknown_v = map(ad.is_undefined_primal, (density, vector))
            if unknown_d and unknown_v:
                raise NotImplementedError("A direct basis transpose requires one fixed linear argument")
            result = [None]*7
            if isinstance(cotangent, ad.Zero):
                if unknown_d: result[4] = ad.Zero(density.aval)
                if unknown_v: result[6] = ad.Zero(vector.aval)
            elif unknown_v:
                result[6] = (push if is_pull else pull).bind(
                    env, c, a, r, density, reference, cotangent)
            elif unknown_d:
                direction, weights = (cotangent, vector) if is_pull else (vector, cotangent)
                result[4] = _density_transpose(lambda d: push.bind(
                    env, c, a, r, d, reference, direction), weights)
            return tuple(result)

        for primitive, is_pull in ((push, False), (pull, True)):
            ad.primitive_jvps[primitive] = partial(derivative, primitive, kind, is_pull)
            ad.primitive_transposes[primitive] = partial(transpose, is_pull, push, pull)

    value = core.Primitive("gradscf_direct_jk_basis")

    def apply(env, c, a, r, density, reference):
        return operator.bind(pack(env, c, a, r), density, *((reference,) if screened else ()))

    value.def_impl(apply)
    value.def_abstract_eval(lambda env, c, a, r, density, reference:
                            jax.core.ShapedArray((nops, *density.shape), jnp.float64))
    mlir.register_lowering(value, mlir.lower_fun(apply, multiple_results=False))

    def value_jvp(primals, tangents):
        env, c, a, r, density, reference = primals
        denv, dc, da, dr, dd, _ = tangents
        primal = value.bind(*primals)
        terms = []
        if not isinstance(denv, ad.Zero):
            terms.append(jnp.zeros_like(primal)+guard((denv,),
                "Direct J/K basis AD requires fixed environment control data"))
        if not isinstance(dc, ad.Zero):
            terms.append(families["coefficient"][0].bind(*primals, dc))
        if not isinstance(da, ad.Zero):
            terms.append(families["exponent"][0].bind(*primals, da))
        if not isinstance(dr, ad.Zero):
            terms.append(geometry_push[1].bind(*primals, dr))
        if not isinstance(dd, ad.Zero):
            terms.append(value.bind(env, c, a, r, dd, reference))
        return primal, sum(terms) if terms else ad.Zero.from_primal_value(primal)

    def value_transpose(cotangent, env, c, a, r, density, reference):
        if any(ad.is_undefined_primal(x) for x in (env, c, a, r, reference)):
            raise NotImplementedError("Direct J/K density transpose requires its primal basis and coordinates")
        result = None
        if ad.is_undefined_primal(density):
            # Retain independent physical operands in the density adjoint.
            # Transposing the raw packed-ENV operator here loses coordinate
            # and basis response under an outer differentiation.
            result = (ad.Zero(density.aval) if isinstance(cotangent, ad.Zero) else
                      _density_transpose(lambda d: value.bind(env, c, a, r, d, reference), cotangent))
        return None, None, None, None, result, None

    ad.primitive_jvps[value] = value_jvp
    ad.primitive_transposes[value] = value_transpose
    batching.primitive_batchers[value] = partial(batch_product, value)
    return value.bind
