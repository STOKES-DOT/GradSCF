"""UKS occupied RI contractions and direct-density history, CPU float64."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.scf import uks
from gradscf.scf.iteration import run_unrestricted_scf


def problem():
    rng = np.random.default_rng(20261105)
    factors = rng.normal(size=(4, 5, 5)) * .035
    factors = (factors + factors.swapaxes(-1, -2)) * .5
    coefficients = np.stack([np.linalg.qr(rng.normal(size=(5, 5)))[0] for _ in range(2)])
    occupations = np.array([[1., 1., 0., 0., 0.], [1., 0., 0., 0., 0.]])
    density = (coefficients * occupations[:, None, :]) @ coefficients.swapaxes(-1, -2)
    return tuple(map(jnp.asarray, (factors, coefficients, occupations, density)))


def df_oracle(factors, density):
    eri = jnp.einsum("Qpq,Qrs->pqrs", factors, factors)
    coulomb = jnp.einsum("pqrs,sr->pq", eri, density.sum(axis=0))
    exchange = jnp.einsum("prqs,ars->apq", eri, density)
    values = jnp.concatenate([coulomb[None], exchange])
    return .5 * (values + values.swapaxes(-1, -2))


def df_action(factors, density, coefficients, occupations):
    values = uks._spin_jk_matrices(jnp.zeros((0, 0, 0, 0)), *density,
        df_factors=factors, mo_coeff_a=coefficients[0], mo_coeff_b=coefficients[1],
        mo_occ_a=occupations[0], mo_occ_b=occupations[1], nocc_a=2, nocc_b=1)
    return jnp.stack(values[1:])


def run_problem(*, factors, xc="hf", direct_basis=None, eri_lr=None, max_cycle=3):
    n = factors.shape[-1]
    eri = jnp.einsum("Qpq,Qrs->pqrs", factors, factors)
    backend = "direct" if direct_basis is not None else "full"
    return uks.run_uks_from_integrals(overlap=jnp.eye(n),
        hcore=jnp.diag(jnp.array([-1.4, -.8, .2, .8, 1.3])) + .025 * (jnp.ones((n, n)) - jnp.eye(n)),
        eri=jnp.zeros((0, 0, 0, 0)) if direct_basis is not None else eri,
        eri_lr=eri_lr, nalpha=2, nbeta=1, nuclear_repulsion=0.,
        ao=jnp.zeros((0, n)), ao_deriv1=jnp.zeros((4, 0, n)), grid_weights=jnp.zeros(0),
        direct_basis=direct_basis, config=uks.UKSConfig(xc_spec=xc, jk_backend=backend,
            max_cycle=max_cycle, conv_tol=0., direct_scf_tol=0.))


def test_df_uses_distinct_occupied_widths(monkeypatch):
    factors, coefficients, occupations, density = problem()
    widths = []
    original = uks._df_jk_with_orbital_hint

    def observed(*args, **kwargs):
        width = int(args[4] if len(args) > 4 else kwargs["nocc"])
        jax.debug.callback(lambda value: widths.append(int(value)), jnp.asarray(width), ordered=True)
        return original(*args, **kwargs)

    monkeypatch.setattr(uks, "_df_jk_with_orbital_hint", observed)
    n = factors.shape[-1]
    result = uks.run_uks_from_integrals(overlap=jnp.eye(n), hcore=jnp.diag(jnp.arange(n, dtype=jnp.float64)),
        eri=jnp.zeros((0, 0, 0, 0)), df_factors=factors, nalpha=2, nbeta=1, nuclear_repulsion=0.,
        ao=jnp.zeros((0, n)), ao_deriv1=jnp.zeros((4, 0, n)), grid_weights=jnp.zeros(0),
        init_mo_coeff_alpha=coefficients[0], init_mo_coeff_beta=coefficients[1],
        init_mo_occ_alpha=occupations[0], init_mo_occ_beta=occupations[1],
        config=uks.UKSConfig(xc_spec="hf", jk_backend="df", max_cycle=2, conv_tol=0.))
    np.asarray(result.fock_matrix_alpha)
    assert widths and set(widths) == {1, 2}, "RI exchange must use nalpha=2 and nbeta=1, not nmo=5"
    expected = df_oracle(factors, jnp.stack([result.density_matrix_alpha, result.density_matrix_beta]))
    np.testing.assert_allclose(result.fock_matrix_alpha, jnp.diag(jnp.arange(n)) + expected[0] - expected[1], atol=2e-13, rtol=2e-13)


@pytest.mark.parametrize("kind", ["fractional_tail", "nonprefix_integer", "mixed_density"])
def test_df_matches_density_oracle_for_general_states(kind):
    factors, coefficients, occupations, density = problem()
    if kind == "fractional_tail":
        occupations = jnp.array([[1., .5, .5, 0., 0.], [.5, .5, 0., 0., 0.]])
    elif kind == "nonprefix_integer":
        occupations = jnp.array([[1., 0., 0., 1., 0.], [0., 0., 0., 0., 1.]])
    if kind != "mixed_density":
        density = (coefficients * occupations[:, None, :]) @ coefficients.swapaxes(-1, -2)
    else:
        density = .7 * density + .3 * jnp.stack([jnp.diag(occupations[0]), jnp.diag(occupations[1])])
    actual = jax.jit(df_action)(factors, density, coefficients, occupations)
    np.testing.assert_allclose(actual, df_oracle(factors, density), atol=2e-13, rtol=2e-13)


def test_df_dynamic_inputs_keep_jvp_vjp_and_fd():
    factors, coefficients, occupations, density = problem()
    rng = np.random.default_rng(857)
    df = jnp.asarray(rng.normal(size=factors.shape) * .01)
    df = (df + df.swapaxes(-1, -2)) * .5
    dd = jnp.asarray(rng.normal(size=density.shape) * .03)
    dd = (dd + dd.swapaxes(-1, -2)) * .5
    action = jax.jit(lambda f, d: df_action(f, d, coefficients, occupations))
    actual, tangent = jax.jvp(action, (factors, density), (df, dd))
    expected, expected_tangent = jax.jvp(df_oracle, (factors, density), (df, dd))
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(tangent, expected_tangent, atol=2e-12, rtol=2e-12)
    weight = jnp.asarray(rng.normal(size=actual.shape))
    grad = jax.grad(lambda f, d: jnp.sum(action(f, d) * weight), argnums=(0, 1))(factors, density)
    oracle_grad = jax.grad(lambda f, d: jnp.sum(df_oracle(f, d) * weight), argnums=(0, 1))(factors, density)
    for value, reference in zip(grad, oracle_grad):
        np.testing.assert_allclose(value, reference, atol=2e-12, rtol=2e-12)
    step = 1e-5
    finite_difference = (action(factors + step * df, density + step * dd) - action(factors - step * df, density - step * dd)) / (2 * step)
    np.testing.assert_allclose(tangent, finite_difference, atol=2e-10, rtol=2e-8)
    changed = action(factors * 1.1, density + .02 * dd)
    assert not np.allclose(actual, changed), "same-shape physical updates must not reuse stale values"
    np.testing.assert_allclose(changed, df_oracle(factors * 1.1, density + .02 * dd), atol=2e-13, rtol=2e-13)


def install_direct_oracle(monkeypatch, eri):
    from gradscf.integrals.backends.native import jk
    from gradscf.scf import rsh
    calls, selections = [], []

    def selected(provider, density, *, with_j=True, with_k=True, omega=0., **controls):
        del provider, controls
        selections.append((with_j, with_k, float(omega)))
        jax.debug.callback(lambda d: calls.append(np.array(d)), density, ordered=True)
        operator = eri * (.3 if omega else 1.)
        coulomb = jnp.einsum("pqrs,...sr->...pq", operator, density) if with_j else jnp.zeros_like(density)
        exchange = jnp.einsum("prqs,...rs->...pq", operator, density) if with_k else jnp.zeros_like(density)
        return coulomb, exchange

    monkeypatch.setattr(jk, "_selected_jk", selected)
    monkeypatch.setattr(rsh, "_require_direct_rsh_provider", lambda provider: None)
    # Isolate the exchange execution; all comparisons retain zero local XC.
    monkeypatch.setattr(uks, "_unrestricted_xc_energy_and_matrices",
        lambda **kw: (jnp.asarray(0.), jnp.zeros((2,) + kw["density_a"].shape)))
    return calls, selections


@pytest.mark.parametrize("xc", ["hf", "pbe0", "cam-b3lyp"])
def test_direct_reuses_density_increment_and_matches_full(monkeypatch, xc):
    factors, _, _, _ = problem()
    eri = jnp.einsum("Qpq,Qrs->pqrs", factors, factors)
    calls, _ = install_direct_oracle(monkeypatch, eri)
    full = run_problem(factors=factors, xc=xc, eri_lr=.3 * eri if xc == "cam-b3lyp" else None)
    direct = run_problem(factors=factors, xc=xc, direct_basis=object())
    direct_density = np.stack([direct.density_matrix_alpha, direct.density_matrix_beta])
    np.testing.assert_allclose(direct_density, np.stack([full.density_matrix_alpha, full.density_matrix_beta]), atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(direct.total_energy, full.total_energy, atol=2e-12, rtol=2e-12)
    for value, reference in [(direct.fock_matrix_alpha, full.fock_matrix_alpha), (direct.fock_matrix_beta, full.fock_matrix_beta)]:
        np.testing.assert_allclose(value, reference, atol=2e-12, rtol=2e-12)
    assert any(np.linalg.norm(d) > 1e-10 and np.max(np.abs(np.trace(d, axis1=-2, axis2=-1))) < 1e-10 for d in calls), "later direct calls must consume delta density"
    assert any(np.allclose(d, direct_density, atol=2e-12) for d in calls if d.shape == direct_density.shape), "final certification must include a fresh full-density build"


def test_rsh_only_dfj_uses_ri_coulomb_and_direct_exchange(monkeypatch):
    from gradscf.scf.rsh import rsh_preset
    factors, coefficients, occupations, density = problem()
    eri = jnp.einsum("Qpq,Qrs->pqrs", factors, factors)
    _, selections = install_direct_oracle(monkeypatch, eri)
    ri_factors = 1.17 * factors  # Distinct J and K sources expose accidental RIJK.
    h = jnp.diag(jnp.linspace(-1.2, .8, 5))
    preset = rsh_preset("cam-b3lyp")
    omega, lr, sr = preset.to_range_separated_hybrid_coefficients()
    expected_j = df_oracle(ri_factors, density)[0]
    expected_k = (sr + .3 * (lr - sr)) * df_oracle(factors, density)[1:]
    expected_fock = h[None] + expected_j[None] - expected_k
    expected_energy = (jnp.sum(density * h[None]) + .5 * jnp.sum(density.sum(axis=0) * expected_j)
                       - .5 * jnp.sum(density * expected_k))
    energy, _, fa, fb = uks._raw_fock_and_energy_for_state(
        density_a=density[0], density_b=density[1], mo_coeff_a=coefficients[0], mo_coeff_b=coefficients[1],
        mo_occ_a=occupations[0], mo_occ_b=occupations[1],
        ao=jnp.zeros((0, 5)), ao_deriv1=jnp.zeros((4, 0, 5)), weights=jnp.zeros(0),
        h=h, eri=jnp.zeros((0, 0, 0, 0)), df_factors=ri_factors, enuc=jnp.asarray(0.),
        alpha=jnp.asarray(sr), overlap=jnp.eye(5), direct_basis=object(), xc_kind="GGA",
        cfg=uks.UKSConfig(xc_spec="cam-b3lyp", jk_backend="df", auxbasis="weigend", only_dfj=True))
    np.testing.assert_allclose(jnp.stack([fa, fb]), expected_fock, atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(energy, expected_energy, atol=2e-13, rtol=2e-13)
    assert selections and all(not with_j and with_k for with_j, with_k, _ in selections)
    assert {value for _, _, value in selections} == {0., omega}


@pytest.mark.parametrize("mismatch", ["energy", "fock", "nonfinite"])
def test_state_evaluator_rejects_false_incremental_convergence(mismatch):
    h, eye = jnp.diag(jnp.array([-1., 1.])), jnp.eye(2)
    occ = jnp.array([[1., 0.], [0., 0.]])
    coeff = jnp.stack([eye, eye])
    fock = jnp.stack([h, h])

    def evaluate(density, coefficients, energies, previous):
        del density, coefficients, energies
        energy, raw = jnp.asarray(0.), fock
        if previous is None:
            if mismatch == "energy":
                energy = jnp.asarray(1e-6)
            elif mismatch == "fock":
                raw = raw.at[0, 0, 1].set(1e-6).at[0, 1, 0].set(1e-6)
            else:
                energy = jnp.asarray(jnp.nan)
        return energy, raw, jnp.asarray(0.) if previous is None else previous.aux + 1

    result = run_unrestricted_scf(fock_builder=lambda *_: (fock, fock, jnp.asarray(0.)),
        state_evaluator=evaluate, initial_aux=jnp.asarray(0.), refresh_on_convergence=True,
        density_spin=jnp.stack([jnp.diag(occ[0]), jnp.diag(occ[1])]),
        mo_coeff_spin=coeff, mo_occ_spin=occ, mo_energy_spin=jnp.stack([jnp.diag(h)] * 2),
        overlap=eye, max_cycle=3, damping=0., conv_tol=1e-12, conv_tol_density=1e-11,
        conv_tol_grad=1e-10, orthogonalization_eps=1e-10)
    assert not bool(result[4]), "fresh E/F must pass before an incremental state is certified"
