"""NNAO consumers share the implicit libcint production integral interface.

These are small randomly initialized MACE fixtures, not trained checkpoints.
All basis arrays and SCF calculations use float64. Geometry is in Angstrom;
the finite-difference directions below are dimensionless raw coefficients and
log(alpha) scales. Energy tolerances are in Hartree.
"""
import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.model.nnao import prepare_direct_basis
from gradscf.model.nnao.integrals.context import prepare_native_context


def test_nnao_consumers_use_implicit_libcint_interface():
    root = Path(__file__).parents[2] / 'src/gradscf/model/nnao'
    explicit = []
    for source in root.rglob('*.py'):
        if 'vendor' in source.parts:
            continue
        for node in ast.walk(ast.parse(source.read_text())):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in {'make_plan', 'make_auxiliary_plan'}):
                if any(key.arg == 'backend' for key in node.keywords):
                    explicit.append(f'{source.relative_to(root)}:{node.lineno}')
    assert not explicit, f'NNAO selects an integral engine explicitly: {explicit}'
    atom = [('H', (0., 0., 0.)), ('H', (0., 0., .74))]
    layout = prepare_direct_basis(atom, cart=False, basis_family='szp663_direct')
    context = prepare_native_context(layout, atom, 'def2-universal-jkfit')
    assert context['plan'].backend == 'libcint'


@pytest.mark.parametrize('bond', [.74, .91])
def test_two_geometry_projection_log_exponents_jvp_vjp_and_fd(bond):
    from gradscf.model.nnao.integrals.context import prepare_fixed_context
    from gradscf.model.nnao.integrals.fixed_exponents import physical_inputs
    from gradscf.model.nnao.integrals.variable_exponents import native_projection

    atom = [('H', (0., 0., 0.)), ('H', (0., 0., bond))]
    layout = prepare_direct_basis(atom, cart=False, basis_family='szp663_direct')
    keys = layout.exponent_scale_keys
    beta = jnp.array([.03, -.02])
    coefficients = layout.reference_outputs()
    coefficients = coefficients + .015*jnp.sin(jnp.arange(coefficients.size).reshape(coefficients.shape))
    scaled = layout.with_log_exponent_scales(dict(zip(keys, beta)))
    fixed = prepare_fixed_context(scaled, ('H', 'H'), np.array([p for _, p in atom]),
        basis_family='szp663_direct', jk_backend='df', auxbasis='def2-universal-jkfit')
    context = prepare_native_context(layout, atom, 'def2-universal-jkfit')
    forward, backward = native_projection(context, keys)
    expected = physical_inputs(scaled, coefficients, fixed.overlap, fixed.hcore, fixed.factors)
    actual = forward(coefficients, beta)
    for left, right in zip(expected, actual):
        np.testing.assert_allclose(left, right, atol=2e-10, rtol=2e-10)
    cotangent = tuple(.01*jnp.cos(jnp.arange(x.size).reshape(x.shape)) for x in actual)
    dc = .02*jnp.sin(jnp.arange(coefficients.size).reshape(coefficients.shape) + .4)
    db = jnp.array([.07, -.04])
    co, cb = backward(coefficients, beta, cotangent)
    _, tangent = jax.jvp(forward, (coefficients, beta), (dc, db))
    lhs = jnp.vdot(co, dc) + jnp.vdot(cb, db)
    rhs = sum(jnp.vdot(t, c) for t, c in zip(tangent, cotangent))
    np.testing.assert_allclose(lhs, rhs, atol=2e-10, rtol=2e-9)
    assert np.linalg.norm(cb) > 1e-8

    def scalar(c, b):
        return sum(jnp.vdot(a, v) for a, v in zip(forward(c, b), cotangent))

    for direction, analytic in [((dc, jnp.zeros_like(db)), jnp.vdot(co, dc)),
                                ((jnp.zeros_like(dc), db), jnp.vdot(cb, db))]:
        for step in (2e-4, 1e-4):
            dcoeff, dbeta = direction
            values = {k: scalar(coefficients+k*step*dcoeff, beta+k*step*dbeta)
                      for k in (-2, -1, 1, 2)}
            fd = (8*(values[1]-values[-1])-values[2]+values[-2])/(12*step)
            np.testing.assert_allclose(analytic, fd, atol=2e-8, rtol=2e-6)


def test_fixed_cache_rejects_changed_geometry_and_log_exponents(tmp_path):
    from gradscf.model.nnao.integrals.cache import fixed_signature, read_fixed_cache, write_fixed_cache
    from gradscf.integrals.basis.contraction import primitive_basis

    atom = [('H', (0., 0., 0.)), ('H', (0., 0., .74))]
    layout = prepare_direct_basis(atom, cart=False, basis_family='szp663_direct')
    coords = np.array([p for _, p in atom])
    def signature(basis, positions):
        topology, parameters = primitive_basis(basis.topology, basis.parameters)
        return fixed_signature(basis, ('H', 'H'), positions, 'szp663_direct',
                               'def2-universal-jkfit', topology, parameters)
    original = signature(layout, coords)
    path = write_fixed_cache(tmp_path/'integrals.npz', original, np.eye(2), np.eye(2), np.ones((1, 3)))
    changed = layout.with_log_exponent_scales({('H', 0): .01})
    moved_coords = coords + [[0., 0., 0.], [0., 0., .01]]
    moved = prepare_direct_basis(list(zip(('H', 'H'), moved_coords)),
                                 cart=False, basis_family='szp663_direct')
    for current in (signature(changed, coords), signature(moved, moved_coords)):
        assert current != original
        with pytest.raises(ValueError, match='does not match'):
            read_fixed_cache(path, current, 'host')


def test_qvszps_context_preserves_ecp_hamiltonian_and_coefficient_response():
    from gradscf import integrals
    from gradscf.model.nnao import prepare_grimme_basis
    from gradscf.model.nnao.integrals.context import prepare_fixed_context
    from gradscf.model.nnao.integrals.fixed_exponents import contracted_one_electron

    atom = [('C', (0., 0., 0.)), ('H', (0., 0., 1.09))]
    layout = prepare_grimme_basis(atom, cart=False)
    assert layout.topology.nuclear_charges == (4, 1)
    assert layout.ecps[0].ncore == 2 and layout.nelectron == 5
    context = prepare_fixed_context(layout, ('C', 'H'), np.array([p for _, p in atom]),
        basis_family='qvszps', jk_backend='direct', auxbasis='def2-universal-jkfit')
    outputs = layout.reference_outputs()
    transform, overlap, hcore = contracted_one_electron(layout, outputs, context.overlap, context.hcore)
    bound = layout.bind(outputs)
    plan = integrals.make_plan(layout.topology)
    bare = plan.evaluate('kinetic', bound) + plan.evaluate('nuclear', bound)
    ecp = plan.evaluate('ecp', bound, ecps=layout.ecps)
    assert np.linalg.norm(ecp) > 1e-3
    np.testing.assert_allclose(overlap, plan.evaluate('overlap', bound), atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(hcore, bare+ecp, atol=2e-9, rtol=2e-10)
    direction = jnp.zeros_like(outputs).at[0, 0, 0].set(.2).at[0, 1, 1].set(-.1)
    scalar = lambda c: jnp.sum(contracted_one_electron(layout, c, context.overlap, context.hcore)[2])
    analytic = jax.jvp(scalar, (outputs,), (direction,))[1]
    step = 1e-4
    fd = (scalar(outputs+step*direction)-scalar(outputs-step*direction))/(2*step)
    np.testing.assert_allclose(analytic, fd, atol=2e-6, rtol=2e-6)


def test_contracted_projection_rejects_ecp_instead_of_omitting_it():
    from gradscf.model.nnao import prepare_grimme_basis
    from gradscf.model.nnao.integrals.variable_exponents import native_projection

    atom = [('C', (0., 0., 0.)), ('H', (0., 0., 1.09))]
    layout = prepare_grimme_basis(atom, cart=False)
    context = prepare_native_context(layout, atom, 'def2-universal-jkfit')
    with pytest.raises(NotImplementedError, match='ECP'):
        native_projection(context, layout.exponent_scale_keys)


def test_small_mace_two_geometry_energy_update_and_checkpoint(tmp_path):
    pytest.importorskip('mace_jax')
    pytest.importorskip('cuequivariance')
    from flax import nnx
    from jax.flatten_util import ravel_pytree
    from gradscf import scf
    from gradscf.model.nnao import MACEBasisModel, build_graph
    from gradscf.model.nnao.integrals.variable_exponents import native_projection
    from gradscf.model.nnao.training.native import JointAdam
    from gradscf.model.nnao.training.objective import BasisRHF, _df_rhf_solver
    from gradscf.model.nnao.training.trainer import trainable_parameter_mask
    from gradscf.model.nnao.training.checkpoints import save_checkpoint, load_checkpoint

    cpu = jax.devices('cpu')[0]
    # Explicit transfer follows NativeJointEngine: libcint stays on CPU while
    # model and implicit SCF use the available model device.
    device = next((d for d in jax.devices() if d.platform == 'gpu'), cpu)
    with jax.default_device(device):
        model = MACEBasisModel(elements=(1,), channels=4, num_interactions=1,
            max_ell=1, correlation=2, basis_family='szp663_direct',
            zero_init=False, rngs=nnx.Rngs(3))
        definition, parameters, other = nnx.split(model, nnx.Param, ...)
        theta, unravel = ravel_pytree(parameters)
        mask = trainable_parameter_mask(parameters)
        beta = jnp.array([.03, -.02])
    predict = jax.jit(lambda x, graph: nnx.merge(definition, unravel(x), other)(graph))
    contexts = []
    for bond in (.74, .91):
        atom = [('H', (0., 0., 0.)), ('H', (0., 0., bond))]
        with jax.default_device(cpu):
            layout = prepare_direct_basis(atom, cart=False, basis_family='szp663_direct')
            context = prepare_native_context(layout, atom, 'def2-universal-jkfit')
            context['physical'], context['pullback'] = native_projection(context, layout.exponent_scale_keys)
            context['enuc'] = jax.device_put(scf.nuclear_repulsion_energy(
                layout.parameters.nuclear_coords, jnp.array(layout.topology.nuclear_charges)), device)
        with jax.default_device(device):
            context['graph'] = build_graph([1, 1], [p for _, p in atom], element_order=(1,))
        contexts.append(context)
    solve, solve_grad = _df_rhf_solver(2, 1e-8, 0.)

    def evaluate(vector, *, gradient=True):
        th, be = jax.device_put((vector[:theta.size], vector[theta.size:]), device)
        values, gradients = [], []
        for context in contexts:
            if gradient:
                output, nn_pullback = jax.vjp(lambda x: predict(x, context['graph']), th)
            else:
                output = predict(th, context['graph'])
            np.testing.assert_array_equal(output[:, 2], 0.)
            out_cpu, beta_cpu = jax.device_put((output, be), cpu)
            physical = context['physical'](out_cpu, beta_cpu)
            assert all(a.device.platform == 'cpu' for a in physical)
            if gradient:
                (value, info), cotangent = solve_grad(jax.device_put(physical, device), context['enuc'])
            else:
                value, info = solve(jax.device_put(physical, device), context['enuc'])
            BasisRHF._checked_info(value, info)
            values.append(float(value))
            if gradient:
                co, cb = context['pullback'](out_cpu, beta_cpu, jax.device_put(cotangent, cpu))
                gt = nn_pullback(jax.device_put(co, device))[0]
                gradients.append(np.concatenate((np.asarray(gt), np.asarray(cb))))
        loss = float(np.mean(values))
        return (loss, np.mean(gradients, axis=0)) if gradient else loss

    vector = jnp.concatenate((theta, beta))
    initial, gradient = evaluate(vector)
    assert np.isfinite(initial) and np.isfinite(gradient).all()
    assert np.linalg.norm(gradient[:theta.size]*mask) > 1e-8
    assert np.linalg.norm(gradient[theta.size:]) > 1e-8
    # Reconverge the actual two-geometry energy loss for independent model and
    # log-exponent directions; frozen MACE constants never enter FD directions.
    direction = np.sin(np.arange(theta.size)+.4)*mask
    direction /= np.linalg.norm(direction)
    directions = (np.r_[direction, np.zeros(beta.size)],
                  np.r_[np.zeros(theta.size), [.07, -.04]])
    for direction in directions:
        analytic = gradient@direction
        for step in (2e-4, 1e-4):
            values = {k: evaluate(vector+k*step*direction, gradient=False) for k in (-2, -1, 1, 2)}
            fd = (8*(values[1]-values[-1])-values[2]+values[-2])/(12*step)
            np.testing.assert_allclose(analytic, fd, atol=2e-6, rtol=2e-5)
    optimizer = JointAdam(theta.size, learning_rate=1e-5, exponent_learning_rate=1e-4,
                          log_bound=.25, trainable_mask=mask)
    state = optimizer.init(vector)
    candidate, candidate_state = optimizer.update(vector, jnp.asarray(gradient), state)
    updated, candidate_gradient = evaluate(candidate)
    assert updated < initial
    np.testing.assert_array_equal(np.asarray(candidate[:theta.size])[~mask], np.asarray(theta)[~mask])
    assert np.all(np.exp(np.asarray(candidate[theta.size:])) > 0.)
    # The initial moment tree is still the accepted pre-update state.
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(candidate_state)))
    identity = dict(fixture='small-random-mace-two-h2-geometries', dtype='float64',
                    exponent_keys=[list(k) for k in contexts[0]['layout'].exponent_scale_keys])
    checkpoint = dict(identity=identity, parameters=np.asarray(candidate),
        gradient=candidate_gradient, optimizer=candidate_state, epoch=1, loss_hartree=updated)
    path = tmp_path/'last.npz'
    save_checkpoint(path, checkpoint, optimizer)
    restored = load_checkpoint(path, identity, optimizer, vector)
    np.testing.assert_array_equal(restored['parameters'], candidate)
    np.testing.assert_array_equal(restored['gradient'], candidate_gradient)
    for a, b in zip(jax.tree.leaves(restored['optimizer']), jax.tree.leaves(candidate_state)):
        np.testing.assert_array_equal(a, b)
