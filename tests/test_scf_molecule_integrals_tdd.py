"""Regression checks for reuse at the molecular SCF input boundary."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.integrals import make_plan, prepare_basis
from gradscf.integrals.backends.native.autodiff import evaluation
from gradscf.integrals.plan import IntegralPlan
from gradscf.scf import RKSConfig, UKSConfig
from gradscf.scf.inputs import assembly


_H2 = "H 0 0 0; H 0 0 .8"


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("only_dfj", [False, True])
def test_direct_inputs_reuse_the_first_raw_basis_and_xc_kind(monkeypatch, unrestricted, only_dfj):
    prepared = []
    classified = []
    prepare = assembly.prepare_basis
    classify = assembly.xc_type

    def record_prepare(spec, basis, **kwargs):
        topology, params = prepare(spec, basis, **kwargs)
        prepared.append((basis, params))
        return topology, params

    def record_classification(spec):
        classified.append(spec)
        return classify(spec)

    monkeypatch.setattr(assembly, "prepare_basis", record_prepare)
    monkeypatch.setattr(assembly, "xc_type", record_classification)
    build, config_type = (
        (assembly.build_uks_integral_inputs, UKSConfig) if unrestricted
        else (assembly.build_rks_integral_inputs, RKSConfig)
    )
    config = config_type(xc_spec="hf", jk_backend="df" if only_dfj else "direct",
                         only_dfj=only_dfj, auxbasis="weigend" if only_dfj else None)
    inputs = build(atom=_H2, basis="sto-3g", spin=0, config=config)
    expected = ["sto-3g", "weigend"] if only_dfj else ["sto-3g"]

    assert ([basis for basis, _ in prepared], classified) == (expected, ["hf"])
    assert inputs.direct_basis.parameters is prepared[0][1]
    assert (inputs.df_factors is not None) == only_dfj


@pytest.mark.parametrize("unrestricted", [False, True])
def test_scf_input_assembly_packs_all_one_electron_operators_once(monkeypatch, unrestricted):
    operands = []
    coefficient_data = IntegralPlan.coefficient_data

    def record_packing(plan, params, **kwargs):
        operands.append(params)
        return coefficient_data(plan, params, **kwargs)

    monkeypatch.setattr(IntegralPlan, "coefficient_data", record_packing)
    build, config_type = (
        (assembly.build_uks_integral_inputs, UKSConfig) if unrestricted
        else (assembly.build_rks_integral_inputs, RKSConfig)
    )
    inputs = build(atom=_H2, basis="sto-3g", spin=0, config=config_type(xc_spec="hf"))

    assert len(operands) == 1
    assert inputs.dipole_integrals.shape == (3, 2, 2)
    assert np.isfinite(np.asarray(inputs.hcore)).all()


def test_grouped_scf_one_electron_integrals_keep_joint_parameter_ad():
    topology, params = prepare_basis("He .10 -.22 .09; H -.25 .35 .79", "sto-3g",
                                     unit="Bohr", charge=1)
    plan = make_plan(topology)

    def grouped(physical):
        return plan.evaluate_scf_one_electron(physical, include_dipole=True)

    def separate(physical):
        return tuple(plan.evaluate(name, physical) for name in (
            "overlap", "kinetic", "nuclear", "dipole",
        ))

    def signal(array):
        return .05 * jnp.sin(jnp.arange(array.size, dtype=array.dtype).reshape(array.shape) + .2)

    direction = jax.tree_util.tree_map(signal, params)

    def value_and_tangent(operation):
        return jax.jit(lambda physical, tangent: jax.jvp(operation, (physical,), (tangent,)))(
            params, direction,
        )

    reference_values, reference_tangents = value_and_tangent(separate)
    values, tangents = value_and_tangent(grouped)
    for observed, expected in zip(values + tangents, reference_values + reference_tangents, strict=True):
        np.testing.assert_allclose(observed, expected, atol=2e-10, rtol=2e-9)

    weights = jax.tree_util.tree_map(signal, reference_values)

    def pullback(operation):
        return jax.jit(lambda physical, cotangent: jax.vjp(operation, physical)[1](cotangent)[0])(
            params, weights,
        )

    for observed, expected in zip(
        jax.tree_util.tree_leaves(pullback(grouped)),
        jax.tree_util.tree_leaves(pullback(separate)),
        strict=True,
    ):
        np.testing.assert_allclose(observed, expected, atol=2e-10, rtol=2e-9)


def test_uks_input_dipole_is_default_on_but_optional(monkeypatch):
    operators = []
    evaluate = evaluation.evaluate_differentiable

    def record_operator(operator, *args, **kwargs):
        operators.append(operator)
        return evaluate(operator, *args, **kwargs)

    monkeypatch.setattr(evaluation, "evaluate_differentiable", record_operator)
    options = dict(atom=_H2, basis="sto-3g", spin=0, config=UKSConfig(xc_spec="hf"))
    default = assembly.build_uks_integral_inputs(**options)
    assert default.dipole_integrals.shape == (3, 2, 2)
    assert operators.count("dipole") == 1

    skipped = assembly.build_uks_integral_inputs(**options, include_dipole_integrals=False)
    assert skipped.dipole_integrals is None
    assert operators.count("dipole") == 1
    np.testing.assert_array_equal(skipped.overlap, default.overlap)
    np.testing.assert_array_equal(skipped.hcore, default.hcore)
