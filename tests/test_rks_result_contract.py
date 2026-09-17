"""RKS result compatibility across host and differentiable boundaries."""

from dataclasses import fields, replace

import jax
import jax.numpy as jnp
import numpy as np

from gradscf.scf.rks import (
    RKSConfig,
    RKSResult,
    TraceableRKSResult,
    run_rks_from_integrals,
    run_rks_from_integrals_traceable,
)
from gradscf.scf.gks import GKSResult
from gradscf.scf.roks import ROKSResult


def _inputs(hcore=-1.0):
    # One doubly occupied orbital: E = 2 h + (11|11) + E_nuc, in Hartree.
    return dict(
        overlap=jnp.eye(1),
        hcore=jnp.reshape(jnp.asarray(hcore), (1, 1)),
        eri=jnp.full((1, 1, 1, 1), 0.6),
        nelectron=2,
        nuclear_repulsion=0.3,
        ao=jnp.ones((1, 1)),
        ao_deriv1=jnp.zeros((4, 1, 1)),
        grid_weights=jnp.ones(1),
        config=RKSConfig(xc_spec="hf", max_cycle=4, conv_tol=1e-12),
    )


def test_result_names_refer_to_one_canonical_pytree_dataclass():
    assert TraceableRKSResult is RKSResult
    result = run_rks_from_integrals_traceable(**_inputs())
    leaves, structure = jax.tree_util.tree_flatten(result)
    assert len(leaves) == len(fields(RKSResult)) == 14
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    restored = jax.tree_util.tree_unflatten(structure, leaves)
    assert type(restored) is RKSResult
    assert replace(restored, total_energy=jnp.array(0.0)).total_energy == 0.0


def test_eager_preserves_python_scalar_types_and_traceable_shapes():
    eager = run_rks_from_integrals(**_inputs())
    traceable = run_rks_from_integrals_traceable(**_inputs())
    assert type(eager.converged) is bool
    assert type(eager.cycles) is int
    for name in (
        "total_energy", "electronic_energy", "nuclear_repulsion", "xc_energy",
        "exact_exchange_fraction",
    ):
        assert type(getattr(eager, name)) is float
        assert getattr(traceable, name).shape == ()
    assert traceable.converged.shape == traceable.cycles.shape == ()
    for name in ("mo_energy", "mo_occ"):
        assert getattr(eager, name).shape == getattr(traceable, name).shape == (1,)
    for name in ("mo_coeff", "density_matrix", "fock_matrix", "overlap_matrix", "hcore_matrix"):
        assert getattr(eager, name).shape == getattr(traceable, name).shape == (1, 1)
    for field in fields(RKSResult):
        np.testing.assert_allclose(getattr(eager, field.name), getattr(traceable, field.name), atol=1e-13)
    np.testing.assert_allclose(eager.total_energy, -1.1, atol=1e-13)


def test_jit_returns_complete_canonical_result_and_differentiates_energy():
    run = jax.jit(lambda h: run_rks_from_integrals_traceable(**_inputs(h)))
    result = run(jnp.array(-1.0))
    assert isinstance(result, RKSResult)
    assert isinstance(result, TraceableRKSResult)
    assert bool(result.converged)
    np.testing.assert_allclose(result.total_energy, -1.1, atol=1e-13)
    np.testing.assert_allclose(jax.grad(lambda h: run(h).total_energy)(jnp.array(-1.0)), 2.0, atol=1e-12)


def test_eager_result_is_also_a_valid_pytree_input():
    result = run_rks_from_integrals(**_inputs())
    energy = jax.jit(lambda value: value.electronic_energy + value.nuclear_repulsion)(result)
    np.testing.assert_allclose(energy, result.total_energy, atol=1e-13)


def test_existing_result_subclasses_keep_dataclass_fields_and_constructors():
    result = run_rks_from_integrals(**_inputs())
    common = {field.name: getattr(result, field.name) for field in fields(RKSResult)}
    gks = GKSResult(**common, orbital_gradient_norm=0.0)
    roks = ROKSResult(
        **common,
        density_matrix_alpha=result.density_matrix * 0.5,
        density_matrix_beta=result.density_matrix * 0.5,
        fock_matrix_alpha=result.fock_matrix,
        fock_matrix_beta=result.fock_matrix,
        mo_energy_alpha=result.mo_energy,
        mo_energy_beta=result.mo_energy,
        orbital_gradient_norm=0.0,
    )
    assert len(fields(gks)) == 15
    assert len(fields(roks)) == 21
    assert isinstance(gks, RKSResult)
    assert isinstance(roks, RKSResult)
    assert replace(gks, total_energy=-2.0).orbital_gradient_norm == 0.0
    assert replace(roks, total_energy=-2.0).density_matrix_alpha.shape == (1, 1)
