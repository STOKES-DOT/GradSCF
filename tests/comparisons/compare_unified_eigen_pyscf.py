"""Forward validation against installed PySCF; no production PySCF dependency.

Run: PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 python
     tests/comparisons/compare_unified_eigen_pyscf.py
This is a fixed reproducible suite, not a CLI. Same-matrix comparisons isolate
solver changes; native HF/TDA comparisons separately test the full forward chain.
"""

from pathlib import Path
import json
import platform
import subprocess
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pyscf
from pyscf import gto as pgto, scf as pscf, dft as pdft, ao2mo
from gradscf import ci, dft, scf, gto
from gradscf.solvers import (
    EigenSolverConfig,
    EigenResponseConfig,
    LinearOperator,
    solve_hermitian,
)
from gradscf.tddft.tda import solve_tda_from_operator
from gradscf.tddft.unrestricted import solve_unrestricted_tda_from_operator
from gradscf.tools.spectra import HARTREE_TO_EV

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reproducibility/solver_validation/unified_eigen_pyscf_cpu.json"
WATER = "O 0 0 0; H 0 -.757 .587; H 0 .757 .587"
H3 = "H 0 0 0; H 0 0 .85; H 0 0 1.9"
H4 = "H 0 0 0; H 0 0 .8; H 0 0 1.9; H 0 0 3.1"
SCF_TOL, ROOT_TOL, PYSCF_ROOT_TOL, CLUSTER_TOL = 1e-12, 1e-11, 1e-9, 1e-7
records = []


def clusters(energy):
    groups = []
    for index, value in enumerate(energy):
        if not groups or abs(value - energy[groups[-1][-1]]) > CLUSTER_TOL:
            groups.append([index])
        else:
            groups[-1].append(index)
    return groups


def max_cluster_error(reference, actual, groups):
    return max(abs(np.sum(actual[g]) - np.sum(reference[g])) for g in groups)


def build_mf(atom, basis, xc="hf", spin=0):
    mol = pgto.M(
        atom=atom, basis=basis, unit="Angstrom", cart=True, spin=spin, verbose=0
    )
    if xc == "hf":
        mf = pscf.UHF(mol) if spin else pscf.RHF(mol)
    else:
        mf = pdft.UKS(mol) if spin else pdft.RKS(mol)
        mf.xc = xc
        mf.grids.level = 1
    mf.conv_tol = SCF_TOL
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("Reference SCF did not converge")
    return mf


def reference_td(mf, roots, singlet=True, *, exhaustive=False):
    td = mf.TDA()
    td.nstates = roots
    if mf.mo_coeff.ndim == 2:
        td.singlet = singlet
    td.conv_tol = ROOT_TOL if exhaustive else PYSCF_ROOT_TOL
    td.max_cycle = 200
    td.lindep = 1e-12
    if exhaustive:
        dim = len(td.gen_vind()[1])
        assert dim <= 256
        # PySCF truncates initial guesses to space_inc, itself bounded by
        # nstates. Request the full dimension to actually use the full basis.
        td.kernel(x0=np.eye(dim), nstates=dim)
    else:
        td.kernel()
    if len(td.e) < roots or not np.all(td.converged):
        raise RuntimeError("Reference TDA roots did not converge")
    return td


def dipole_probes(mf, singlet):
    dipole = mf.mol.intor_symmetric("int1e_r", comp=3)
    if mf.mo_coeff.ndim == 3:
        probes = []
        for c, occ in zip(mf.mo_coeff, mf.mo_occ):
            no = int(np.count_nonzero(occ))
            probes.append(
                np.einsum("xmn,mi,na->xia", dipole, c[:, :no], c[:, no:]).reshape(3, -1)
            )
        return np.concatenate(probes, axis=1).T
    c = mf.mo_coeff
    no = mf.mol.nelectron // 2
    d = np.einsum("xmn,mi,na->xia", dipole, c[:, :no], c[:, no:]).reshape(3, -1).T
    return np.sqrt(2) * d if singlet else np.zeros_like(d)


def same_operator(label, atom, basis, xc="hf", spin=0, singlet=True, roots=4):
    start = time.perf_counter()
    mf = build_mf(atom, basis, xc, spin)
    td = mf.TDA()
    if not spin:
        td.singlet = singlet
    vind, diagonal = td.gen_vind()
    dim = len(diagonal)
    a = np.asarray(vind(np.eye(dim))).T
    symmetry_error = float(np.max(abs(a - a.T)))
    np.testing.assert_allclose(a, a.T, atol=1e-11, rtol=0)
    positive = np.linalg.eigvalsh(a)
    positive = positive[positive > td.positive_eig_threshold]
    roots = min(roots, len(positive))
    while (
        roots < len(positive)
        and abs(positive[roots] - positive[roots - 1]) < CLUSTER_TOL
    ):
        roots += 1
    default_td = reference_td(mf, roots, singlet)
    default_energy = np.asarray(default_td.e)[:roots]
    default_error = float(np.max(abs(default_energy - positive[:roots])))
    td = reference_td(mf, roots, singlet, exhaustive=True)
    energy = np.asarray(td.e)[:roots]
    np.testing.assert_allclose(energy, positive[:roots], atol=1e-8, rtol=0)
    probes = dipole_probes(mf, singlet)
    if spin:
        vectors = np.stack(
            [np.concatenate([np.ravel(x) for x in xy[0]]) for xy in td.xy[:roots]],
            axis=1,
        )
    else:
        vectors = np.stack(
            [np.ravel(xy[0]) * np.sqrt(2) for xy in td.xy[:roots]], axis=1
        )
    strengths = np.asarray(td.oscillator_strength())[:roots]
    np.testing.assert_allclose(
        2 / 3 * energy * np.sum((vectors.T @ probes) ** 2, axis=1),
        strengths,
        atol=1e-10,
        rtol=0,
    )
    groups = clusters(energy)
    for method in ("dense", "davidson"):
        array = jnp.asarray(a)
        operator = LinearOperator(
            (dim, dim),
            array.dtype,
            lambda x: array @ x,
            diagonal=jnp.diag(array),
            matmat=lambda x: array @ x,
        )
        cfg = EigenSolverConfig(
            method=method,
            nroots=roots,
            atol=ROOT_TOL,
            maxiter=200,
            max_subspace=min(dim, 40),
            value_min=float(td.positive_eig_threshold),
        )
        out = solve_hermitian(
            operator, config=cfg, response=EigenResponseConfig(target="eigenvalues")
        )
        e, v = np.asarray(out.values), np.asarray(out.vectors)
        f = 2 / 3 * e * np.sum((v.T @ probes) ** 2, axis=1)
        subspace_error = max(
            np.linalg.norm(v[:, g] @ v[:, g].T - vectors[:, g] @ vectors[:, g].T)
            for g in groups
        )
        row = dict(
            section="same_operator",
            case=label,
            method=method,
            xc=xc,
            basis=basis,
            spin=spin,
            pyscf_reference_initialization="complete basis, nstates=dimension",
            pyscf_default_energies=default_energy.tolist(),
            pyscf_default_max_energy_error_hartree=default_error,
            pyscf_default_converged=np.asarray(default_td.converged).tolist(),
            singlet=None if spin else singlet,
            atom=atom,
            unit="Angstrom",
            cart=True,
            grid_level=None if xc == "hf" else 1,
            dimension=dim,
            nroots=roots,
            cluster_sizes=[len(g) for g in groups],
            pyscf_energy_hartree=energy.tolist(),
            gradscf_energy_hartree=e.tolist(),
            pyscf_strength=strengths.tolist(),
            gradscf_strength=f.tolist(),
            max_energy_error_hartree=float(np.max(abs(e - energy))),
            max_cluster_strength_error=float(max_cluster_error(strengths, f, groups)),
            max_cluster_projector_error=float(subspace_error),
            max_residual_hartree=float(np.max(out.residual_norms)),
            pyscf_max_residual_hartree=float(
                np.max(np.linalg.norm(a @ vectors - vectors * energy, axis=0))
            ),
            symmetry_error=symmetry_error,
            state_response_valid=np.asarray(out.response_valid).tolist(),
            converged=bool(np.all(out.converged)),
        )
        np.testing.assert_allclose(e, energy, atol=1e-8, rtol=0)
        assert row["max_cluster_strength_error"] < 5e-7
        assert subspace_error < 5e-7 and row["converged"]
        records.append(row)
    # Check method adapters and their restricted/unrestricted amplitude norms.
    if spin:
        energies = mf.mo_energy
        noa, nob = (int(np.count_nonzero(o)) for o in mf.mo_occ)
        de = [
            energies[s][no:][None, :] - energies[s][:no, None]
            for s, no in enumerate((noa, nob))
        ]
        result = solve_unrestricted_tda_from_operator(
            *map(jnp.asarray, de),
            lambda x: x @ array.T,
            jnp.diag(array),
            nstates=roots,
            davidson_tol=ROOT_TOL,
            davidson_max_iter=200,
            davidson_max_subspace=min(dim, 40),
        )
        v = np.concatenate(
            [
                np.asarray(result.amplitudes_alpha).reshape(roots, -1),
                np.asarray(result.amplitudes_beta).reshape(roots, -1),
            ],
            axis=1,
        ).T
    else:
        no = mf.mol.nelectron // 2
        de = mf.mo_energy[no:][None, :] - mf.mo_energy[:no, None]
        result = solve_tda_from_operator(
            jnp.asarray(de),
            lambda x: x @ array.T,
            jnp.diag(array),
            nstates=roots,
            davidson_tol=ROOT_TOL,
            davidson_max_iter=200,
            davidson_max_subspace=min(dim, 40),
        )
        v = np.sqrt(2) * np.asarray(result.amplitudes).reshape(roots, -1).T
    e = np.asarray(result.excitation_energies)
    f = 2 / 3 * e * np.sum((v.T @ probes) ** 2, axis=1)
    np.testing.assert_allclose(e, energy, atol=1e-8, rtol=0)
    assert max_cluster_error(strengths, f, groups) < 5e-7 and result.converged
    records.append(
        dict(
            section="method_adapter",
            case=label,
            max_energy_error_hartree=float(np.max(abs(e - energy))),
            max_cluster_strength_error=float(max_cluster_error(strengths, f, groups)),
            converged=True,
        )
    )
    if max(map(len, groups)) > 1:
        out = solve_hermitian(
            operator,
            config=cfg,
            response=EigenResponseConfig(target="subspace"),
            probes=jnp.asarray(probes),
        )
        reference_projection = vectors @ (vectors.T @ probes)
        np.testing.assert_allclose(
            out.projection, reference_projection, atol=1e-8, rtol=0
        )
        records.append(
            dict(
                section="degenerate_subspace",
                case=label,
                rank=roots,
                response_valid=bool(out.response_valid),
                projection_error=float(
                    np.max(abs(np.asarray(out.projection) - reference_projection))
                ),
                trace_error=float(abs(out.eigenvalue_sum - np.sum(energy))),
            )
        )
        assert out.response_valid
    print(label, "complete", f"{time.perf_counter()-start:.2f}s", flush=True)


def ci_comparison(label, atom, spin=0, frozen=None):
    start = time.perf_counter()
    mf = build_mf(atom, "sto-3g", spin=spin)
    n = mf.mo_coeff.shape[-1]
    if spin:
        ca, cb = mf.mo_coeff
        h = tuple(c.T @ mf.get_hcore() @ c for c in (ca, cb))
        g = tuple(
            ao2mo.general(mf.mol, cs, compact=False).reshape((n,) * 4)
            for cs in ((ca, ca, ca, ca), (ca, ca, cb, cb), (cb, cb, cb, cb))
        )
        ref = ci.UnrestrictedReference(h, g, mf.mol.nelec, mf.mol.energy_nuc())
    else:
        c = mf.mo_coeff
        h = c.T @ mf.get_hcore() @ c
        g = ao2mo.restore(1, ao2mo.kernel(mf.mol, c), n)
        ref = ci.CIReference(h, g, mf.mol.nelectron // 2, mf.mol.energy_nuc())
    oracle = mf.CISD(frozen=frozen).run(conv_tol=1e-13, max_cycle=200)
    assert oracle.converged
    for method in ("dense", "davidson"):
        calc = ci.CISD(
            ref, solver=method, frozen=frozen, conv_tol=ROOT_TOL, max_cycle=200
        ).run()
        actual_dm = np.asarray(calc.make_rdm1())
        expected_dm = np.asarray(oracle.make_rdm1())
        row = dict(
            section="ci",
            case=label,
            method=method,
            atom=atom,
            basis="sto-3g",
            unit="Angstrom",
            spin=spin,
            frozen=frozen,
            pyscf_total_energy=float(oracle.e_tot),
            gradscf_total_energy=float(calc.e_tot),
            total_energy_error=float(abs(calc.e_tot - oracle.e_tot)),
            correlation_energy_error=float(abs(calc.e_corr - oracle.e_corr)),
            dm1_max_error=float(np.max(abs(actual_dm - expected_dm))),
            converged=bool(calc.converged),
            max_residual_hartree=float(np.max(calc.result.residual_norms)),
        )
        assert (
            row["converged"]
            and row["total_energy_error"] < 2e-9
            and row["dm1_max_error"] < 1e-7
        )
        records.append(row)
    print(label, "complete", f"{time.perf_counter()-start:.2f}s", flush=True)


def native_comparison(label, atom, spin=0):
    start = time.perf_counter()
    reference = build_mf(atom, "sto-3g", spin=spin)
    roots = 3 if not spin else 2
    oracle = reference_td(reference, roots, exhaustive=True)
    reference_e = np.asarray(oracle.e)[:roots]
    reference_f = np.asarray(oracle.oscillator_strength())[:roots]
    mol = gto.M(atom=atom, basis="sto-3g", unit="Angstrom", cart=True, spin=spin)
    cls = scf.UHF if spin else dft.RKS
    kwargs = {} if spin else dict(xc="hf")
    mf = cls(
        mol,
        conv_tol=SCF_TOL,
        conv_tol_density=1e-11,
        conv_tol_grad=1e-10,
        max_cycle=200,
        **kwargs,
    ).run()
    assert mf.converged
    calc = mf.TDA(nstates=roots, davidson_tol=ROOT_TOL, davidson_max_iter=200)
    calc.kernel()
    e = np.asarray(calc.e)
    f = np.asarray(calc.oscillator_strength())
    row = dict(
        section="native_chain",
        case=label,
        atom=atom,
        basis="sto-3g",
        spin=spin,
        unit="Angstrom",
        cart=True,
        scf_energy_error=float(abs(mf.e_tot - reference.e_tot)),
        pyscf_energy_hartree=reference_e.tolist(),
        gradscf_energy_hartree=e.tolist(),
        pyscf_strength=reference_f.tolist(),
        gradscf_strength=f.tolist(),
        max_energy_error_hartree=float(np.max(abs(e - reference_e))),
        max_cluster_strength_error=float(
            max_cluster_error(reference_f, f, clusters(reference_e))
        ),
        converged=bool(np.all(calc.result.converged)),
        elapsed_seconds=time.perf_counter() - start,
    )
    assert (
        row["converged"]
        and row["scf_energy_error"] < 1e-8
        and row["max_energy_error_hartree"] < 2e-7
    )
    assert row["max_cluster_strength_error"] < 1e-6
    records.append(row)
    print(label, "complete", f"{time.perf_counter()-start:.2f}s", flush=True)


def main():
    start = time.perf_counter()
    same_operator("water HF singlet", WATER, "6-31g")
    same_operator("water HF triplet", WATER, "6-31g", singlet=False)
    same_operator("water PBE singlet", WATER, "6-31g", xc="pbe")
    same_operator("water B3LYP singlet", WATER, "6-31g", xc="b3lyp")
    same_operator("Be HF singlet", "Be 0 0 0", "sto-3g", roots=3)
    same_operator("Be HF triplet", "Be 0 0 0", "sto-3g", singlet=False, roots=3)
    same_operator("OH UHF", "O 0 0 0; H 0 0 .9697", "6-31g", spin=1)
    ci_comparison("H4 CISD", H4)
    ci_comparison("LiH frozen-core CISD", "Li 0 0 0; H 0 0 1.6", frozen=1)
    ci_comparison("H3 UCISD", H3, spin=1)
    native_comparison("native water RHF/TDA", WATER)
    native_comparison("native H3 UHF/TDA", H3, spin=1)
    payload = dict(
        platform=platform.platform(),
        python=sys.version.split()[0],
        jax=jax.__version__,
        pyscf=pyscf.__version__,
        numpy=np.__version__,
        backend=jax.default_backend(),
        dtype="float64",
        omp_threads=1,
        production_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        command="PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 "
        + sys.executable
        + " tests/comparisons/compare_unified_eigen_pyscf.py",
        scf_tolerance=SCF_TOL,
        gradscf_root_tolerance=ROOT_TOL,
        pyscf_root_tolerance=ROOT_TOL,
        pyscf_default_probe_tolerance=PYSCF_ROOT_TOL,
        pyscf_tda_lindep=1e-12,
        cluster_tolerance=CLUSTER_TOL,
        elapsed_seconds=time.perf_counter() - start,
        records=records,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print("Saved", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
