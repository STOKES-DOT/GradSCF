"""Two isolated exact rearrangements, enabled only by this probe harness.

No density symmetrization, factor symmetry assumption, or physical threshold
is introduced. The general non-symmetric raw-factor contract is retained.
"""
from __future__ import annotations
import jax.numpy as jnp
from jax.lax import Precision


def lowrank_factory(*, full):
    def build(j_factors, k_factors, orbo, orbv, hybrid_fraction, *, include_exchange, dtype):
        j_source = jnp.asarray(j_factors, dtype=dtype)
        co, cv = jnp.asarray(orbo, dtype=dtype), jnp.asarray(orbv, dtype=dtype)
        vo = jnp.einsum("Qpq,pa,qi->Qai", j_source, cv, co, precision=Precision.HIGHEST)
        ov = (jnp.einsum("Qpq,pi,qa->Qia", j_source, co, cv, precision=Precision.HIGHEST)
              if full else None)
        alpha = jnp.asarray(hybrid_fraction, dtype=dtype)
        if include_exchange:
            k_source = jnp.asarray(k_factors, dtype=dtype)
            oo = jnp.einsum("Qpq,pi,qj->Qij", k_source, co, co, precision=Precision.HIGHEST)
            vv = jnp.einsum("Qpq,pa,qb->Qab", k_source, cv, cv, precision=Precision.HIGHEST)
            if full:
                kov = ov if k_factors is j_factors else jnp.einsum("Qpq,pi,qa->Qia", k_source, co, cv, precision=Precision.HIGHEST)
                kvo = vo if k_factors is j_factors else jnp.einsum("Qpq,pa,qi->Qai", k_source, cv, co, precision=Precision.HIGHEST)

        def action(values, *, bottom_density, bottom_projection):
            shape = values.shape
            x = jnp.asarray(values, dtype=dtype).reshape(-1, co.shape[1], cv.shape[1])
            if not full and (bottom_density or bottom_projection):
                raise ValueError("The TDA probe prepares only its actual A branch")
            left = ov if bottom_projection else vo.transpose(0, 2, 1)
            right = ov if bottom_density else vo.transpose(0, 2, 1)
            rho = 2*jnp.einsum("Qia,nia->nQ", right, x, precision=Precision.HIGHEST)
            out = jnp.einsum("Qia,nQ->nia", left, rho, precision=Precision.HIGHEST)
            if include_exchange:
                if bottom_density == bottom_projection:
                    k = jnp.einsum("Qij,Qab,njb->nia", oo, vv, x, precision=Precision.HIGHEST)
                else:
                    k = jnp.einsum("Qaj,Qib,njb->nia", kvo, kov, x, precision=Precision.HIGHEST)
                out = out-alpha*k
            return out.reshape(shape)
        return action
    return build


def install(cfg):
    import gradscf.tddft.response as response
    if cfg.get("experiment") == "packed-public":
        from gradscf.integrals.molecular.jk import build_jk_from_packed
        response._jk_from_eri_pair_matrix = build_jk_from_packed
    elif cfg.get("experiment") in {"df-demand", "df-share"}:
        response.build_restricted_lowrank_mo_response_action = lowrank_factory(full=cfg["full"])
    else:
        raise ValueError(cfg.get("experiment"))
