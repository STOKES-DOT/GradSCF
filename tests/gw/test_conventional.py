"""Actual four-center GW: no auxiliary fit or AO-pair spectral factorization."""
import inspect
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def model():
    rng = np.random.default_rng(713)
    b = jnp.asarray(rng.normal(size=(5, 4, 4))*.05)
    b = (b+b.swapaxes(-1, -2))*.5
    energy = jnp.array([-.9, -.4, .3, .8])
    return b, dict(mo_energy=energy, mo_coeff=jnp.eye(4), nocc=2,
        fock_matrix=jnp.diag(energy), hcore_matrix=jnp.diag(energy),
        density_matrix=jnp.diag(jnp.array([2., 2., 0., 0.])),
        nw=12, orbs=(1, 2), g_orbitals=(3, 0, 1, 2), qp_tol=1e-10)


@pytest.mark.parametrize("iterative", [False, True])
def test_conventional_matches_exact_factor_oracle_and_keeps_response(monkeypatch, iterative):
    from gradscf.gw import g0w0_cd_restricted
    from gradscf.integrals.molecular import factorization
    from gradscf.solvers import LinearSolverConfig
    assert "eri" in inspect.signature(g0w0_cd_restricted).parameters
    b, kw = model()
    def forbidden(*args, **kwargs):
        raise AssertionError("Conventional GW may not create AO-pair spectral factors")
    monkeypatch.setattr(factorization, "eri_to_df_factors", forbidden)
    monkeypatch.setattr(factorization, "eri_pair_matrix_to_df_factors", forbidden)
    config = LinearSolverConfig(method="gmres", rtol=1e-12, atol=1e-14, restart=12) if iterative else None
    eri = jnp.einsum("Qij,Qkl->ijkl", b, b)
    expected = g0w0_cd_restricted(**kw, df_factors=b)
    result = g0w0_cd_restricted(**kw, eri=eri, screening_config=config)
    assert bool(result.converged)
    for name in ("mo_energy", "sigma_qp", "qp_weight", "qp_residual"):
        np.testing.assert_allclose(getattr(result, name), getattr(expected, name), atol=1e-10,
                                   rtol=1e-9, equal_nan=True)
    def score(t, conventional):
        factors = b.at[0, 0, 2].add(t).at[0, 2, 0].add(t)
        source = dict(eri=jnp.einsum("Qij,Qkl->ijkl", factors, factors)) if conventional else dict(df_factors=factors)
        return g0w0_cd_restricted(**kw, **source).mo_energy[1:3].sum()
    value, derivative = jax.jit(jax.value_and_grad(lambda t: score(t, True)))(0.)
    reference, response = jax.jit(jax.value_and_grad(lambda t: score(t, False)))(0.)
    np.testing.assert_allclose((value, derivative), (reference, response), atol=2e-10, rtol=1e-8)
    for step in (1e-4, 3e-5):
        fd = (score(step, True)-score(-step, True))/(2*step)
        np.testing.assert_allclose(derivative, fd, atol=2e-8, rtol=2e-5)


def test_conventional_budget_rejects_before_transform():
    from gradscf.gw import g0w0_cd_restricted
    assert "eri" in inspect.signature(g0w0_cd_restricted).parameters
    b, kw = model()
    with pytest.raises(ValueError, match="budget|memory"):
        g0w0_cd_restricted(**kw, eri=jnp.einsum("Qij,Qkl->ijkl", b, b), max_memory=1e-6)


@pytest.mark.parametrize("control", ["wmn_fixed", "wmn_static_fixed", "resolvent_expansion", "resolvent_data_fixed"])
def test_conventional_rejects_factor_screening_controls(control):
    from gradscf.gw import g0w0_cd_restricted
    b, kw = model()
    with pytest.raises(NotImplementedError, match="cached|resolvent"):
        g0w0_cd_restricted(**kw, eri=jnp.einsum("Qij,Qkl->ijkl", b, b),
                           **{control: True if control == "resolvent_expansion" else jnp.ones(1)})


@pytest.mark.parametrize("window", [dict(g_orbitals=()), dict(screening_occupied=()), dict(screening_virtual=())])
def test_conventional_empty_windows_keep_existing_semantics(window):
    from gradscf.gw import g0w0_cd_restricted
    b, kw = model()
    kw.update(window)
    expected = g0w0_cd_restricted(**kw, df_factors=b)
    actual = g0w0_cd_restricted(**kw, eri=jnp.einsum("Qij,Qkl->ijkl", b, b))
    np.testing.assert_allclose(actual.mo_energy, expected.mo_energy, atol=1e-10)


def test_conventional_plan_includes_the_shared_realified_block_solve():
    from gradscf.gw.conventional import prepare
    from gradscf.solvers.memory import plan_linear
    b, kw = model()
    source = dict(eri=jnp.einsum("Qij,Qkl->ijkl", b, b))
    data = prepare(kw["mo_coeff"], kw["mo_energy"], source=source,
        nocc=2, orbs=(1, 2), gi=(0, 1, 2, 3), oi=(0, 1), va=(2, 3), nw=12,
        fock=kw["fock_matrix"], hcore=kw["hcore_matrix"], density=kw["density_matrix"],
        max_memory=10, reserved_bytes=0, config=None, v_mf=None)
    plan = data["plan"]
    _, working = plan_linear(8, jnp.float64, plan.screening_config, plan.max_memory,
                            resident_bytes=plan.resident_bytes, nrhs=plan.rhs_batch)
    assert plan.working_bytes >= working


def test_native_coulomb_checks_optimizer_workspace_before_execution(monkeypatch):
    from gradscf import integrals
    from gradscf.gw.conventional import _coulomb
    from gradscf.integrals.backends.native import jk
    top, parameters = integrals.prepare_basis("H 0 0 0; H 0 0 .74", "sto-3g", cart=True)
    source = dict(direct_basis=jk.NativeDirectBasis(integrals.make_plan(top), parameters))
    monkeypatch.setattr(jk, "native_jk_workspace_bytes", lambda *args: 100_000_000)
    with pytest.raises(ValueError, match="budget|memory"):
        _coulomb(source, jnp.eye(top.nao), 1.)


def test_conventional_transforms_reserve_prepared_potential_and_coefficients(monkeypatch):
    from gradscf.gw import conventional
    from gradscf._memory import array_storage_bytes
    b, kw = model()
    seen, original = [], conventional.transform_eri_blocks
    def observe(*args, **kwargs):
        seen.append(kwargs["reserved_bytes"])
        return original(*args, **kwargs)
    monkeypatch.setattr(conventional, "transform_eri_blocks", observe)
    conventional.prepare(kw["mo_coeff"], kw["mo_energy"],
        source=dict(eri=jnp.einsum("Qij,Qkl->ijkl", b, b)), nocc=2, orbs=(1, 2),
        gi=(0, 1, 2, 3), oi=(0, 1), va=(2, 3), nw=12,
        fock=kw["fock_matrix"], hcore=kw["hcore_matrix"], density=kw["density_matrix"],
        max_memory=10, reserved_bytes=0, config=None, v_mf=None)
    # A newly formed potential is a separate owner from F/h/D. The full C
    # remains live while selected orbital copies are transformed.
    minimum = array_storage_bytes(*(kw[k] for k in
        ("mo_coeff", "mo_energy", "fock_matrix", "hcore_matrix", "density_matrix")))
    assert min(seen) >= minimum+kw["fock_matrix"].nbytes


@pytest.mark.parametrize("packed", [False, True])
def test_conventional_rejects_source_promotion_before_coulomb(monkeypatch, packed):
    from gradscf.gw import conventional
    n = 64
    pairs = n*(n+1)//2
    shape = (pairs, pairs) if packed else (n,)*4
    source = np.zeros(shape, dtype=np.float32)
    def forbidden(*args, **kwargs):
        raise AssertionError("The whole-source promotion must be checked before J is dispatched")
    monkeypatch.setattr(conventional, "_coulomb", forbidden)
    identity = jnp.eye(n)
    with pytest.raises(ValueError, match="budget|memory"):
        conventional.prepare(identity, jnp.arange(n, dtype=jnp.float64),
            source={"eri_pair_matrix" if packed else "eri": source}, nocc=1,
            orbs=(0,), gi=(1,), oi=(0,), va=(1,), nw=4,
            fock=identity, hcore=identity, density=identity,
            max_memory=50. if packed else 200., reserved_bytes=0, config=None, v_mf=None)


@pytest.mark.parametrize("update_w", [False, True])
def test_conventional_evgw_matches_fixed_orbital_oracle(update_w):
    from gradscf.gw import evgw_cd_restricted
    assert "eri" in inspect.signature(evgw_cd_restricted).parameters
    b, kw = model()
    kw.pop("qp_tol")
    controls = dict(update_w=update_w, max_iter=60, tol=1e-10, damping=.2)
    expected = evgw_cd_restricted(**kw, df_factors=b, **controls)
    actual = evgw_cd_restricted(**kw, eri=jnp.einsum("Qij,Qkl->ijkl", b, b), **controls)
    assert bool(actual.converged)
    np.testing.assert_allclose(actual.mo_energy, expected.mo_energy, atol=1e-10)


def test_public_conventional_direct_hf_never_factorizes(monkeypatch):
    from gradscf import gto, scf, gw
    from gradscf.integrals.molecular import factorization
    assert hasattr(gw.GW, "conventional"), "An explicit true four-center facade is required"
    mf = scf.RHF(gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g"), conv_tol=1e-12).direct_scf().run()
    def forbidden(*args, **kwargs):
        raise AssertionError("The direct conventional consumer may not factorize AO pairs")
    monkeypatch.setattr(factorization, "eri_to_df_factors", forbidden)
    monkeypatch.setattr(factorization, "eri_pair_matrix_to_df_factors", forbidden)
    result = gw.GW(mf, nw=12).conventional().run()
    assert bool(result.converged) and result._ao_factors is None
    assert mf._scf_inputs.eri_pair_matrix is None
    with pytest.raises(NotImplementedError, match="conventional|four-center"):
        result.get_bse_inputs()


def test_conventional_ri_source_uses_actual_ao_parameters_not_named_basis():
    from types import SimpleNamespace
    from gradscf import integrals, gw
    from gradscf.integrals.libcint import bind_cartesian
    basis = integrals.basis_from_pyscf_spec("H 0 0 0; H 0 0 .74",
        basis={"H": [[0, [1.4, .7], [.3, .4]]]}, cart=True)
    def forbidden():
        raise AssertionError("An available actual AO basis must not be rebuilt from a named spec")
    owner = object.__new__(gw.GW)
    owner._scf = SimpleNamespace(_scf_inputs=SimpleNamespace(basis=basis), _spec=forbidden)
    source = owner._conventional_source()["direct_basis"]
    actual, _ = source.get_jk(jnp.eye(2), with_k=False)
    expected = jnp.einsum("pqrr->pq", bind_cartesian(basis).evaluate("eri"))
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_native_direct_qp_complete_geometry_response():
    """H2/STO-3G: moving native basis + overlap/Pulay + relaxed HF + W/QP."""
    from gradscf import gto, scf
    from gradscf.gw import g0w0_cd_restricted
    from gradscf.scf.core import _diagonalize_fock
    from gradscf_tools.molecular_response import MolecularEnergy
    seed = scf.RHF(gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g", cart=True),
                   conv_tol=1e-12).direct_scf().run()
    reference = MolecularEnergy(seed, jk_backend="direct", forward="scf", max_memory=64.)
    def energy(coordinates):
        state, inputs = reference.solve_with_inputs(coordinates)
        fock = state.fock_matrix.mean(axis=0)
        levels, coefficients = _diagonalize_fock(fock, inputs["overlap"])
        result = g0w0_cd_restricted(mo_energy=levels, mo_coeff=coefficients, nocc=1,
            direct_basis=inputs["direct_basis"], fock_matrix=fock,
            hcore_matrix=inputs["hcore"], density_matrix=state.density_matrix.sum(axis=0),
            orbs=(0,), nw=16, eta=.03, qp_tol=1e-11, max_memory=64.)
        return result.mo_energy[0]
    coordinates = reference.coordinates
    direction = jnp.zeros_like(coordinates).at[1, 2].set(1.)
    value, tangent = jax.jit(lambda r: jax.jvp(energy, (r,), (direction,)))(coordinates)
    reverse = jax.jit(jax.grad(energy))(coordinates)
    assert bool(jnp.isfinite(value)) and bool(jnp.all(jnp.isfinite(reverse)))
    np.testing.assert_allclose(tangent, jnp.vdot(reverse, direction), atol=2e-9, rtol=1e-8)
    for step in (3e-4, 1e-4):
        fd = (energy(coordinates+step*direction)-energy(coordinates-step*direction))/(2*step)
        np.testing.assert_allclose(tangent, fd, atol=1e-5, rtol=1e-4)
