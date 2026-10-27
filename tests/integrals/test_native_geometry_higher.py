"""Native cubic/quartic coordinate products with independent integral oracles."""
from dataclasses import replace
from itertools import permutations, product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals


def problem(operator, angular=0):
    basis = {"H": [[angular, [.9, 1.]]]}
    top, parameters = integrals.prepare_basis(
        "H .07 -.11 .08; H .31 .22 1.19", basis=basis, unit="Bohr", cart=True)
    plan = integrals.make_plan(top)
    coordinates = jnp.concatenate((parameters.nuclear_coords, parameters.centers))

    def value(coords):
        moving = replace(parameters, nuclear_coords=coords[:2], centers=coords[2:])
        return plan.evaluate(operator, moving)

    reference = lambda coords: pyscf_value(top, parameters, operator, coords)
    return (value, reference), coordinates


def pyscf_value(topology, parameters, operator, coordinates):
    """Independent values with floating ghost AOs and physical potential centers."""
    gto = pytest.importorskip('pyscf.gto')
    coords = np.asarray(coordinates)
    count = parameters.nuclear_coords.shape[0]
    labels = [f'ghost-H{i}' for i in range(len(topology.angular_momenta))]
    shells = {label: [[l, *np.column_stack((np.asarray(a), np.asarray(c))).tolist()]]
              for label, l, a, c in zip(labels, topology.angular_momenta,
                                       parameters.exponents, parameters.coefficients)}
    mol = gto.M(atom=list(zip(labels, coords[count:])), basis=shells,
                cart=topology.cart, unit='Bohr', verbose=0)
    if operator == 'nuclear':
        value = np.zeros((mol.nao, mol.nao))
        for charge, center in zip(topology.nuclear_charges, coords[:count]):
            with mol.with_rinv_origin(center):
                value -= charge*mol.intor('int1e_rinv_cart')
        return value
    name = {'overlap': 'int1e_ovlp_cart', 'kinetic': 'int1e_kin_cart', 'eri': 'int2e_cart'}[operator]
    return mol.intor(name, aosym='s1') if operator == 'eri' else mol.intor(name)


def independent_mixed_difference(function, coordinates, directions):
    """Richardson-extrapolated PySCF values; h is in Bohr.

    The 3e-6 operator-a.u./Bohr**order acceptance below matches the existing
    independent d-shell quartic gate. Native AD uses no displaced integrals.
    """
    coords, ds = np.asarray(coordinates), np.asarray(directions)
    approximations = []
    for step in (.02, .01):
        value = None
        for signs in product((-1., 1.), repeat=len(ds)):
            displaced = coords + step*np.einsum('i,ijk->jk', signs, ds)
            term = np.prod(signs)*np.asarray(function(displaced))
            value = term if value is None else value+term
        approximations.append(value/(2*step)**len(ds))
    return (4*approximations[1]-approximations[0])/3


def directional(function, coordinates, directions):
    current = function
    for direction in directions:
        previous = current
        current = lambda coords, previous=previous, direction=direction: jax.jvp(
            previous, (coords,), (direction,))[1]
    return current(coordinates)


@pytest.mark.parametrize("operator", ["overlap", "kinetic", "nuclear", "eri"])
@pytest.mark.parametrize("order", [3, 4])
def test_directional_products_match_independent_pyscf_differences(operator, order):
    (native, reference), coordinates = problem(operator)
    directions = jnp.asarray(np.random.default_rng(193).normal(size=(order, *coordinates.shape))*.2)
    actual = jax.jit(lambda x, ds: directional(native, x, ds))(coordinates, directions)
    expected = independent_mixed_difference(reference, coordinates, directions)
    assert np.isfinite(actual).all() and np.max(np.abs(actual)) > 1e-8
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("order", [3, 4])
def test_nuclear_operator_and_floating_ao_mixed_permutations(order):
    (native, reference), coordinates = problem("nuclear")
    # All original nuclear coordinates and shell centers are independent.
    # Two different derivatives of the SAME potential center are needed for
    # a nonzero operator/operator/AO/AO quartic; different nuclei would give zero.
    directions = jnp.zeros((4, *coordinates.shape))
    directions = directions.at[0, 0, 0].set(1.).at[1, 2, 1].set(1.)
    directions = directions.at[2, 3, 2].set(1.).at[3, 0, 2].set(1.)
    directions = directions[:order]
    calculate = jax.jit(lambda x, ds: directional(native, x, ds))
    expected = independent_mixed_difference(reference, coordinates, directions)
    assert np.max(np.abs(expected)) > 1e-7
    for indices in permutations(range(order)):
        actual = calculate(coordinates, directions[jnp.asarray(indices)])
        np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("operator", ["nuclear", "eri"])
def test_quartic_adjoint_nonlinear_cotangent_and_direction_slots(operator):
    (native, reference), coordinates = problem(operator)
    rng = np.random.default_rng(198)
    directions = jnp.asarray(rng.normal(size=(4, *coordinates.shape))*.15)
    cotangent = jnp.asarray(rng.normal(size=native(coordinates).shape))
    product = jax.jit(lambda ds: directional(native, coordinates, ds))
    adjoint = lambda c: directional(jax.grad(lambda x: jnp.vdot(native(x), c)),
                                    coordinates, directions[1:])
    pull = jax.jit(adjoint)(cotangent)
    np.testing.assert_allclose(jnp.vdot(pull, directions[0]),
                               jnp.vdot(product(directions), cotangent), atol=2e-10)
    through_direction = jax.jit(jax.grad(lambda v: jnp.vdot(
        product(directions.at[0].set(v)), cotangent)))(directions[0])
    np.testing.assert_allclose(through_direction, pull, atol=2e-10)
    through_cotangent = jax.jit(jax.grad(lambda c: jnp.vdot(adjoint(c), directions[0])))(cotangent)
    np.testing.assert_allclose(through_cotangent, product(directions), atol=2e-10)
    def adjoint_direction_loss(vector):
        pull = directional(jax.grad(lambda x: jnp.vdot(native(x), cotangent)),
                           coordinates, (vector, directions[2], directions[3]))
        return jnp.vdot(pull, directions[0])
    through_adjoint_direction = jax.jit(jax.grad(adjoint_direction_loss))(directions[1])
    expected_pull = directional(jax.grad(lambda x: jnp.vdot(native(x), cotangent)),
                                coordinates, directions[jnp.array([0, 2, 3])])
    np.testing.assert_allclose(through_adjoint_direction, expected_pull, atol=2e-10)
    native_loss = lambda x: jnp.sum(jnp.sin(.17*native(x)))
    reference_loss = lambda x: np.sum(np.sin(.17*reference(x)))
    actual = jax.jit(lambda x: directional(jax.grad(native_loss), x, directions[1:]))(coordinates)
    expected = independent_mixed_difference(reference_loss, coordinates, directions)
    np.testing.assert_allclose(jnp.vdot(actual, directions[0]), expected, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("operator,angular", [(op, 1) for op in ("overlap", "kinetic", "nuclear", "eri")]
                         + [("overlap", 2), ("kinetic", 2)])
def test_p_d_shell_normalization_primitives_and_contractions(operator, angular):
    # Two primitive exponents; the p shell also has two contractions. This
    # exercises alpha-weighted shadow coefficients and contraction-major AO order.
    rows = [[1.3, .7, .2], [.4, .5, .8]] if angular == 1 else [[1.3, .7], [.4, .5]]
    top, parameters = integrals.prepare_basis("H .07 -.11 .08; H .31 .22 1.19",
        basis={"H": [[angular, *rows]]}, unit="Bohr", cart=True)
    native_plan = integrals.make_plan(top)
    coordinates = jnp.concatenate((parameters.nuclear_coords, parameters.centers))
    half, cart_size = top.nao//2, (angular+1)*(angular+2)//2
    indices = ((0, half, cart_size-1, top.nao-1), (cart_size-1, half, 0, top.nao-1),
               (0, 0, half, half), (half-1, half-1, top.nao-1, top.nao-1))

    def value(coords, native):
        moving = replace(parameters, nuclear_coords=coords[:2], centers=coords[2:])
        tensor = (native_plan.evaluate(operator, moving) if native else
                  pyscf_value(top, parameters, operator, coords))
        if operator == "eri":
            return jnp.stack([tensor[index] for index in indices])
        if angular == 2:
            return jnp.stack([tensor[index[:2]] for index in indices])
        return tensor

    directions = jnp.asarray(np.random.default_rng(211).normal(size=(4, *coordinates.shape))*.15)
    actual = jax.jit(lambda x, ds: directional(lambda r: value(r, True), x, ds))(coordinates, directions)
    expected = independent_mixed_difference(lambda r: value(r, False), coordinates, directions)
    assert np.isfinite(actual).all() and np.max(np.abs(actual)) > 1e-8
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("operator", ["nuclear", "eri"])
def test_d_shell_quartic_against_independent_pyscf_mixed_differences(operator):
    from pyscf import gto

    rows = [[1.3, .7], [.4, .5]]
    top, parameters = integrals.prepare_basis("H .07 -.11 .08; H .31 .22 1.19",
        basis={"H": [[2, *rows]]}, unit="Bohr", cart=True)
    plan = integrals.make_plan(top)
    coordinates = jnp.concatenate((parameters.nuclear_coords, parameters.centers))
    directions = jnp.asarray(np.random.default_rng(211).normal(size=(4, *coordinates.shape))*.15)

    def native(coords):
        moving = replace(parameters, nuclear_coords=coords[:2], centers=coords[2:])
        return plan.evaluate(operator, moving)

    def reference(coords):
        # Independently movable ghost d shells and explicit physical potential
        # centers. This checks the full tensor and every Cartesian normalization.
        atom = [("ghost-H", row) for row in coords[2:]]
        mol = gto.M(atom=atom, basis={"H": [[2, *rows]]},
                    cart=True, unit="Bohr", verbose=0)
        if operator == "eri":
            return mol.intor("int2e_cart", aosym="s1")
        value = np.zeros((mol.nao, mol.nao))
        for center in coords[:2]:
            with mol.with_rinv_origin(center):
                value -= mol.intor("int1e_rinv_cart")
        return value

    np.testing.assert_allclose(native(coordinates), reference(np.array(coordinates)), atol=2e-12, rtol=0)
    actual = np.asarray(jax.jit(lambda x, ds: directional(native, x, ds))(coordinates, directions))
    assert np.isfinite(actual).all() and np.max(np.abs(actual)) > 1e-8
    # Independent centered mixed differences are only an oracle. The AD result
    # above uses no displaced integrals. Three steps disclose truncation/roundoff.
    errors, approximations = [], []
    for step in (.03, .015, .0075):
        expected = np.zeros_like(actual)
        for signs in product((-1., 1.), repeat=4):
            displacement = np.einsum("i,ijk->jk", signs, np.array(directions))
            expected += np.prod(signs)*reference(np.array(coordinates)+step*displacement)
        expected /= (2*step)**4
        approximations.append(expected)
        errors.append(float(np.max(np.abs(expected-actual))))
    extrapolated = [(4*fine-coarse)/3 for coarse, fine in zip(approximations[:-1], approximations[1:])]
    extrapolated_errors = [float(np.max(np.abs(value-actual))) for value in extrapolated]
    print(f"d {operator} quartic full tensor PySCF mixed-FD errors: {errors}")
    print(f"Richardson mixed-FD errors: {extrapolated_errors}")
    assert max(extrapolated_errors) < 3e-6
    assert errors[1] < errors[0]/3 and errors[2] < errors[1]/3


@pytest.mark.parametrize("operator", ["overlap", "kinetic", "nuclear", "eri"])
def test_quartic_rigid_translation_and_zero_direction_derivative(operator):
    (native, reference), coordinates = problem(operator)
    directions = jnp.asarray(np.random.default_rng(231).normal(size=(4, *coordinates.shape))*.2)
    product = jax.jit(lambda ds: directional(native, coordinates, ds))
    translation = jnp.broadcast_to(jnp.array([.3, -.2, .1]), coordinates.shape)
    np.testing.assert_allclose(product(directions.at[0].set(translation)), 0., atol=2e-11)
    np.testing.assert_allclose(product(directions.at[0].set(0.)), 0., atol=2e-11)
    # A zero direction is still a live multilinear input: differentiating that
    # slot must recover the nonzero product instead of freezing a zero branch.
    slot = lambda vector: product(directions.at[0].set(vector))
    actual = jax.jit(lambda vector: jax.jvp(slot, (jnp.zeros_like(vector),), (vector,))[1])(directions[0])
    expected = product(directions)
    assert np.max(np.abs(expected)) > 1e-8
    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=2e-7)


@pytest.mark.parametrize("zero", [False, True])
def test_fifth_geometry_derivative_is_explicitly_unsupported(zero):
    (native, _), coordinates = problem("overlap")
    directions = jnp.ones((5, *coordinates.shape))
    if zero:
        directions = directions.at[-1].set(0.)
    with pytest.raises(NotImplementedError, match="(?i)fifth.*geometry"):
        directional(native, coordinates, directions)
