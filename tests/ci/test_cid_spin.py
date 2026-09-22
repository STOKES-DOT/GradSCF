"""CID projected-FCI oracle and spin diagnostics in common/different MO frames."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _full_vector(vector, space):
    from pyscf.fci import cistring
    occ = space.nocc if isinstance(space.nocc, tuple) else (space.nocc,)*2
    strings = [list(map(int, cistring.make_strings(range(space.nmo), no))) for no in occ]
    full = np.zeros(tuple(map(len, strings)))
    for c, d in zip(vector, space.determinants):
        full[strings[0].index(d & ((1 << space.nmo)-1)), strings[1].index(d >> space.nmo)] = c
    return full


@pytest.mark.parametrize("unrestricted", [False, True])
def test_cid_space_has_only_reference_and_doubles(unrestricted):
    from gradscf import ci
    builder, nocc = (ci.make_uci_space, (2, 1)) if unrestricted else (ci.make_ci_space, 2)
    cisd = builder(4, nocc)
    size = sum(r in (0, 2) for r in cisd.ranks)
    cid = builder(4, nocc, excitation_ranks=(0, 2), max_determinants=size)
    assert cid.size == size and set(cid.ranks) == {0, 2}
    assert cid.determinants == tuple(d for d, r in zip(cisd.determinants, cisd.ranks) if r in (0, 2))
    # The budget is for retained CID determinants, not the larger CISD space.
    with pytest.raises(ValueError, match="max_determinants"):
        builder(4, nocc, max_determinants=size)
    for ranks in ((2,), (0, 0, 2), (0, -1), (0, 3), (0, .5)):
        with pytest.raises(ValueError, match="excitation_ranks"):
            builder(4, nocc, excitation_ranks=ranks)


@pytest.mark.parametrize("frozen", [None, ([0], [])])
def test_ucid_matches_projected_fci_and_response(radical, frozen):
    from gradscf import ci
    from pyscf.fci import cistring, direct_uhf
    mf, h, g = radical
    source = ci.UnrestrictedReference(h, g, (2, 1), mf.mol.energy_nuc())
    obj = ci.UCID(source, frozen=frozen, conv_tol=1e-11).run()
    strings = [cistring.make_strings(range(3), no) for no in (2, 1)]
    dets = [int(a) | (int(b) << 3) for a in strings[0] for b in strings[1]]
    effective = direct_uhf.absorb_h1e(tuple(map(np.asarray, h)), tuple(map(np.asarray, g)), 3, (2, 1), .5)
    full = np.column_stack([direct_uhf.contract_2e(effective, v.reshape(3, 3), 3, (2, 1)).ravel()
                            for v in np.eye(9)])
    indices = [dets.index(d) for d in obj.space.determinants]
    expected = np.linalg.eigvalsh(full[np.ix_(indices, indices)])[0]+mf.mol.energy_nuc()
    assert obj.converged
    np.testing.assert_allclose(obj.e_tot, expected, atol=2e-10, rtol=0)
    np.testing.assert_allclose(ci.CID(source, frozen=frozen).run().e_tot, expected, atol=2e-9, rtol=0)
    direction = jnp.diag(jnp.array([.2, -.1, .05]))
    def energy(x):
        return ci.solve_ci((h[0]+x*direction, h[1]), g, obj.space,
            config=ci.CIConfig(conv_tol=1e-11)).total_energies[0]
    np.testing.assert_allclose(jax.jit(jax.grad(energy))(0.),
                               (energy(1e-4)-energy(-1e-4))/2e-4, atol=2e-8, rtol=0)


@pytest.mark.parametrize("unrestricted,frozen", [(False, None), (False, [0, 3]),
                                                (True, None), (True, ([0], [3]))])
def test_spin_square_against_fci_arbitrary_vectors(unrestricted, frozen):
    pytest.importorskip("pyscf")
    from pyscf.fci import spin_op
    from scipy.linalg import expm
    from gradscf import ci
    n, occ = 4, (2, 1) if unrestricted else (2, 2)
    space = (ci.make_uci_space(n, occ, frozen=frozen) if unrestricted else
             ci.make_ci_space(n, 2, frozen=frozen))
    rng = np.random.default_rng(77)
    vector = rng.normal(size=space.size)
    vector /= np.linalg.norm(vector)
    x = rng.normal(size=(n, n))*.3
    overlap = expm(x-x.T) if unrestricted else np.eye(n)
    full = _full_vector(vector, space)
    expected = spin_op.spin_square(full, n, occ, mo_coeff=(np.eye(n), overlap))
    kwargs = {"overlap_ab": jnp.asarray(overlap)} if unrestricted else {}
    actual = jax.jit(lambda c: ci.spin_square(c, space, **kwargs))(jnp.asarray(vector))
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)
    # Normalization, sign and amplitude response of the diagnostic.
    np.testing.assert_allclose(ci.spin_square(-3*vector, space, **kwargs), actual, atol=2e-12)
    direction = jnp.asarray(rng.normal(size=space.size))
    def value(parameter):
        from jax.scipy.linalg import expm
        dynamic = ({"overlap_ab": jnp.asarray(overlap) @ expm(parameter*jnp.asarray(x-x.T))}
                   if unrestricted else {})
        return ci.spin_square(jnp.asarray(vector)+parameter*direction, space, **dynamic)[0]
    np.testing.assert_allclose(jax.jit(jax.grad(value))(0.),
                               (value(1e-5)-value(-1e-5))/2e-5, atol=2e-8, rtol=0)


@pytest.mark.parametrize("ss,multiplicity", [(0., 1.), (2., 3.), (6., 5.)])
def test_known_spin_eigenstates(ss, multiplicity):
    pytest.importorskip("pyscf")
    from pyscf.fci import spin_op
    from gradscf import ci
    space = ci.make_ci_space(4, 2, max_excitation=4)
    matrix = np.column_stack([spin_op.contract_ss(v.reshape(6, 6), 4, (2, 2)).ravel()
                              for v in np.eye(36)])
    values, vectors = np.linalg.eigh(matrix)
    full = vectors[:, np.argmin(abs(values-ss))].reshape(6, 6)
    mapping = _full_vector(np.arange(1, space.size+1), space).astype(int)
    vector = np.empty(space.size)
    vector[mapping.ravel()-1] = full.ravel()
    np.testing.assert_allclose(ci.spin_square(vector, space), (ss, multiplicity), atol=3e-12, rtol=0)


def test_explicit_unrestricted_spin_requires_overlap(radical):
    from gradscf import ci
    mf, h, g = radical
    obj = ci.UCID(ci.UnrestrictedReference(h, g, (2, 1))).run()
    with pytest.raises(ValueError, match="overlap_ab"):
        obj.spin_square()
    ca, cb = mf.mo_coeff
    overlap = ca.T @ mf.get_ovlp() @ cb
    actual = obj.spin_square(overlap_ab=overlap)
    from pyscf.fci import spin_op
    expected = spin_op.spin_square(_full_vector(obj.ci, obj.space), 3, (2, 1),
                                    mo_coeff=mf.mo_coeff, ovlp=mf.get_ovlp())
    np.testing.assert_allclose(actual, expected, atol=2e-11)
    with pytest.raises(ValueError, match="shape"):
        obj.spin_square(overlap_ab=np.eye(2))
    assert np.isnan(ci.spin_square(np.zeros(obj.space.size), obj.space, overlap_ab=overlap)[0])
    obj.excitation_ranks = None
    with pytest.raises(RuntimeError, match="changed"):
        obj.spin_square(overlap_ab=overlap)


@pytest.mark.parametrize("reference", ["UHF", "ROHF"])
def test_native_cid_spin_facade(reference):
    from gradscf import ci, gto, scf
    from pyscf.fci import spin_op
    mol = gto.M(atom="H 0 0 0; H 0 0 .85; H 0 0 1.9", basis="sto-3g", spin=1)
    mf = getattr(scf, reference)(mol, conv_tol=1e-12, max_cycle=150).run()
    obj = mf.CID(conv_tol=1e-11).run()
    if reference == "UHF":
        coeff, overlap = np.asarray(mf.mo_coeff), np.asarray(mf.reference.overlap_matrix)
    else:
        coeff, overlap = (np.asarray(mf.mo_coeff),)*2, np.asarray(mf.scf_result.overlap_matrix)
    expected = spin_op.spin_square(_full_vector(obj.ci, obj.space), 3, (2, 1),
                                    mo_coeff=coeff, ovlp=overlap)
    np.testing.assert_allclose(obj.spin_square(), expected, atol=2e-11)
    assert set(obj.space.ranks) == {0, 2}


def test_cis_rejects_ignored_rank_selection():
    from gradscf import ci
    h, g = jnp.diag(jnp.array([-1., .4])), jnp.zeros((2,)*4)
    with pytest.raises(ValueError, match="excitation_ranks"):
        ci.CIS(ci.CIReference(h, g, 1), excitation_ranks=(0,))
    with pytest.raises(ValueError, match="excitation_ranks"):
        ci.UCIS(ci.UnrestrictedReference((h, h), (g, g, g), (1, 0)), excitation_ranks=(0,))
    for obj in (ci.CIS(ci.CIReference(h, g, 1)),
                ci.UCIS(ci.UnrestrictedReference((h, h), (g, g, g), (1, 0)))):
        obj.excitation_ranks = (0,)
        with pytest.raises(ValueError, match="excitation_ranks"):
            obj.kernel()


def test_restricted_cid_multiroot_and_spin_against_fci():
    pyscf = pytest.importorskip("pyscf")
    from pyscf import ao2mo, fci
    from gradscf import ci
    mf = pyscf.gto.M(atom="H 0 0 0; H 0 0 .8; H 0 0 1.9; H 0 0 3.1",
                     basis="sto-3g", verbose=0).RHF().run(conv_tol=1e-13)
    c = mf.mo_coeff
    h, g = c.T @ mf.get_hcore() @ c, ao2mo.restore(1, ao2mo.kernel(mf.mol, c), 4)
    strings = fci.cistring.make_strings(range(4), 2)
    dets = [int(a) | (int(b) << 4) for a in strings for b in strings]
    ref_det = 3 | (3 << 4)
    keep = [k for k, d in enumerate(dets) if (ref_det & ~d).bit_count() in (0, 2)]
    effective = fci.direct_spin1.absorb_h1e(h, g, 4, (2, 2), .5)
    full = np.column_stack([fci.direct_spin1.contract_2e(effective, v.reshape(6, 6), 4, (2, 2)).ravel()
                            for v in np.eye(36)])
    expected = np.linalg.eigvalsh(full[np.ix_(keep, keep)])[:2]+mf.mol.energy_nuc()
    source = ci.CIReference(jnp.asarray(h), jnp.asarray(g), 2, mf.mol.energy_nuc())
    obj = ci.CID(source, nroots=2, conv_tol=1e-11).run()
    assert np.all(obj.converged)
    np.testing.assert_allclose(obj.e_tot, expected, atol=2e-10, rtol=0)
    assert float(obj.e_tot[0]-ci.CISD(source).run().e_tot) > 1e-8
    for root in range(2):
        reference_spin = fci.spin_op.spin_square0(_full_vector(obj.ci[:, root], obj.space), 4, (2, 2))
        np.testing.assert_allclose(obj.spin_square(root=root), reference_spin, atol=2e-11, rtol=0)
        np.testing.assert_allclose(np.trace(obj.make_rdm1(root=root)), 4., atol=1e-12)
    with pytest.raises(ValueError, match="root"):
        obj.spin_square(root=2)


def test_cid_spin_integral_response(radical):
    from gradscf import ci
    mf, h, g = radical
    ca, cb = mf.mo_coeff
    overlap = jnp.asarray(ca.T @ mf.get_ovlp() @ cb)
    space = ci.make_uci_space(3, (2, 1), excitation_ranks=(0, 2))
    cfg = ci.CIConfig(solver="dense", conv_tol=1e-11, gradient_mode="implicit_eigenvector")
    direction = jnp.diag(jnp.array([.2, -.1, .03]))
    def value(x):
        out = ci.solve_ci((h[0]+x*direction, h[1]), g, space, config=cfg)
        return ci.spin_square(out.coefficients[:, 0], space, overlap_ab=overlap)[0]
    np.testing.assert_allclose(jax.jit(jax.grad(value))(0.),
                               (value(1e-4)-value(-1e-4))/2e-4, atol=2e-8, rtol=0)


def test_spin_failure_and_polarized_limits():
    from gradscf import ci
    for nocc, expected in (((0, 0), (0., 1.)), ((1, 0), (.75, 2.)), ((0, 2), (2., 3.))):
        space = ci.make_uci_space(3, nocc, max_excitation=0)
        np.testing.assert_allclose(ci.spin_square(jnp.ones(1), space, overlap_ab=jnp.eye(3)), expected,
                                   atol=1e-13, rtol=0)
        with pytest.raises(NotImplementedError, match="real"):
            ci.spin_square(jnp.ones(1), space, overlap_ab=jnp.eye(3, dtype=complex))
        assert np.isnan(ci.spin_square(jnp.ones(1), space, overlap_ab=jnp.full((3, 3), jnp.nan))[0])
    frozen = ci.make_ci_space(4, 2, frozen=2, excitation_ranks=(0, 2))
    assert frozen.size == 1
    np.testing.assert_allclose(ci.spin_square(jnp.ones(1), frozen), (0., 1.), atol=1e-13)
    only_reference = ci.make_ci_space(10, 2, max_excitation=20, excitation_ranks=(0, 20), max_determinants=1)
    assert only_reference.size == 1
