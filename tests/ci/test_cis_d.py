"""CIS(D) checked against determinant-space perturbation theory.

Method: Head-Gordon et al. (1994), doi:10.1016/0009-2614(94)00070-0.
Working-expression companion: Q-Chem manual 5.1, equations 7.38–7.40:
delta omega = <CIS|V|U2 HF> + <CIS|V|T2 U1 HF> - E_MP2.
This oracle uses PySCF's independent FCI Hamiltonian, not tensor intermediates
from the GradSCF correction implementation.
"""
from itertools import combinations
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from test_ci import h4


def _operate(det, holes, particles):
    occupied = [i for i in range(64) if det & (1 << i)]
    sign = 1
    for i in holes:
        if i not in occupied:
            return None, 0
        sign *= (-1)**occupied.index(i)
        occupied.remove(i)
    for a in reversed(particles):
        if a in occupied:
            return None, 0
        sign *= (-1)**sum(p < a for p in occupied)
        occupied.append(a)
        occupied.sort()
    return sum(1 << i for i in occupied), sign


def _determinant_correction(mf, dets, hamiltonian, amplitudes, omega, frozen=()):
    n = mf.mo_coeff.shape[1]
    no = mf.mol.nelectron // 2
    occ = list(range(no)) + list(range(n, n + no))
    ref = sum(1 << i for i in occ)
    ref_idx = dets.index(ref)
    eps = np.tile(mf.mo_energy, 2)
    f_ref = eps[occ].sum()
    ranks = [sum(not d & (1 << i) for i in occ) for d in dets]
    occ = [i for i in occ if i % n not in frozen]
    vir = [i for i in list(range(no, n)) + list(range(n + no, 2 * n)) if i % n not in frozen]
    c = np.zeros(len(dets))
    for ii, i in enumerate(p for p in range(no) if p not in frozen):
        for aa, a in enumerate(p for p in range(no, n) if p not in frozen):
            for shift in (0, n):
                new, sign = _operate(ref, (i+shift,), (a+shift,))
                c[dets.index(new)] = sign * amplitudes[ii, aa] / np.sqrt(2)
    np.testing.assert_allclose(c @ c, 1., atol=1e-12)
    t2_c = np.zeros_like(c)
    emp2 = 0.
    for holes in combinations(occ, 2):
        for particles in combinations(vir, 2):
            new, phase = _operate(ref, holes, particles)
            if new not in dets:
                continue
            idx = dets.index(new)
            coupling = hamiltonian[idx, ref_idx]
            denominator = eps[list(holes)].sum() - eps[list(particles)].sum()
            t = coupling / denominator / phase
            emp2 += coupling**2 / denominator
            for j, det in enumerate(dets):
                if c[j] == 0:
                    continue
                out, sign = _operate(det, holes, particles)
                if out in dets:
                    t2_c[dets.index(out)] += t * sign * c[j]
    vc = hamiltonian @ c
    direct = 0.
    for j, d in enumerate(dets):
        frozen_ok = all(bool(d & (1 << (p + shift))) == (p < no)
                        for p in frozen for shift in (0, n))
        if ranks[j] == 2 and frozen_ok:
            f_det = sum(eps[i] for i in range(2*n) if d & (1 << i))
            direct += vc[j]**2 / (omega + f_ref - f_det)
    return direct + c @ hamiltonian @ t2_c - emp2


def test_existing_singlet_correction_against_independent_determinants(h4):
    from gradscf.tddft.cisd import restricted_cisd_second_order_correction
    from gradscf.tddft.types import TDAResult

    mf, h, g, dets, full = h4
    td = mf.TDA().set(nstates=3, conv_tol=1e-11).run()
    # PySCF restricted X has norm^2=1/2, the same convention as the old API.
    x = np.asarray([xy[0] for xy in td.xy])
    ref = SimpleNamespace(mo_coeff=jnp.eye(4), mo_occ=jnp.asarray(mf.mo_occ),
                          mo_energy=jnp.asarray(mf.mo_energy), mo_eri=jnp.asarray(g), nocc=2)
    correction = restricted_cisd_second_order_correction(ref, TDAResult(jnp.asarray(td.e), jnp.asarray(x)))
    expected = [_determinant_correction(mf, dets, full, b*np.sqrt(2), w) for b, w in zip(x, td.e)]
    np.testing.assert_allclose(correction, expected, atol=1e-10)


def test_public_cis_d_and_response(h4):
    from gradscf import ci

    mf, h, g, dets, full = h4
    ref = ci.CIReference(h, g, 2, mf.mol.energy_nuc(), mf.mo_energy)
    obj = ci.CIS_D(ref, nroots=3, solver="dense").run()
    expected = [_determinant_correction(mf, dets, full, b, w)
                for b, w in zip(np.asarray(obj.amplitudes), np.asarray(obj.e_cis))]
    np.testing.assert_allclose(obj.correction, expected, atol=1e-10)
    np.testing.assert_allclose(obj.e, obj.e_cis + np.asarray(expected), atol=1e-12)
    assert np.all(obj.result.valid)

    # Vary h/g while preserving the canonical Fock matrix and orbital energies.
    # This isolates CIS-vector response without pretending the orbitals are fixed
    # along a molecular geometry perturbation.
    from gradscf.ci.solver import restricted_fock
    potential = restricted_fock(jnp.zeros_like(h), jnp.asarray(g), 2)
    cfg = ci.CIConfig(nroots=1, solver="dense", gradient_mode="implicit_eigenvector")
    def energy(t):
        ht, gt = h - t*potential, g*(1+t)
        singles = ci.solve_cis(ht, gt, nocc=2, config=cfg)
        return ci.cis_d_correction(gt, mf.mo_energy, singles, nocc=2).excitation_energies[0]
    ad = jax.jit(jax.grad(energy))(0.)
    fd = (energy(1e-4)-energy(-1e-4))/2e-4
    np.testing.assert_allclose(ad, fd, atol=2e-7, rtol=2e-6)


def test_cis_d_rejects_triplet_and_reports_singular_denominators(h4):
    from gradscf import ci

    mf, h, g, _, _ = h4
    ref = ci.CIReference(h, g, 2, mo_energy=mf.mo_energy)
    with pytest.raises(NotImplementedError, match="singlet"):
        ci.CIS_D(ref, singlet=False).run()
    triplet = ci.solve_cis(h, g, nocc=2, singlet=False)
    with pytest.raises(NotImplementedError, match="singlet"):
        ci.cis_d_correction(g, mf.mo_energy, triplet, nocc=2)
    singles = ci.solve_cis(h, g, nocc=2)
    bad = ci.cis_d_correction(g, jnp.zeros(4), singles, nocc=2)
    assert not np.all(bad.valid)
    assert np.all(np.isnan(bad.excitation_energies))


@pytest.mark.parametrize("frozen", [[0], [3], [0, 3]])
def test_frozen_cis_d_against_determinant_oracle(h4, frozen):
    from gradscf import ci

    mf, h, g, dets, full = h4
    ref = ci.CIReference(h, g, 2, mo_energy=mf.mo_energy)
    obj = ci.CIS_D(ref, frozen=frozen, nroots=1, solver="dense").run()
    expected = _determinant_correction(mf, dets, full, np.asarray(obj.amplitudes[0]),
                                        float(obj.e_cis[0]), frozen=frozen)
    np.testing.assert_allclose(obj.correction[0], expected, atol=1e-10)


def test_cis_d_energy_only_input_does_not_give_partial_gradients(h4):
    from gradscf import ci

    mf, h, g, _, _ = h4
    def correction(t):
        result = ci.solve_cis(h, g*(1+t), nocc=2, config=ci.CIConfig(solver="dense"))
        return ci.cis_d_correction(g*(1+t), mf.mo_energy, result, nocc=2).corrections[0]
    assert np.isfinite(correction(0.))
    assert not np.isfinite(jax.grad(correction)(0.))
