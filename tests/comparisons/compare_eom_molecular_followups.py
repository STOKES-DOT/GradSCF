"""Controlled followups for N2 SCF branch selection and CO degenerate EE pairs.

This records failed attempts as well as successful ones. No production solver
is changed and no PySCF density/orbitals seed the GradSCF calculations.
"""

import json
from time import perf_counter
import jax
import numpy as np
from pyscf import gto as py_gto, scf as py_scf
from pyscf.cc import eom_rccsd
from compare_eom_molecules import CASES, run_case
from gradscf import gto, dft, cc
from gradscf.scf.init_guess import orbital_rotation_guesses

jax.config.update("jax_enable_x64", True)


def nitrogen():
    case = next(c for c in CASES if c[0] == "N2")
    _, basis, atom = case
    mol = gto.M(atom=atom, basis=basis, unit="Angstrom")
    base = dft.RKS(mol, xc="hf", conv_tol=1e-12).run()
    no = int(np.count_nonzero(np.asarray(base.mo_occ) > 0))
    choices = [base]
    trials = [
        {
            "amplitude": 0.0,
            "energy": float(base.e_tot),
            "converged": bool(base.converged),
        }
    ]
    for amplitude, coeff in zip(
        (0.05, 0.15, 0.4),
        orbital_rotation_guesses(
            base.mo_coeff, amplitudes=(0.05, 0.15, 0.4), seed=20260923
        ),
    ):
        density = 2 * coeff[:, :no] @ coeff[:, :no].T
        trial = dft.RKS(
            mol,
            xc="hf",
            init_guess=density,
            conv_tol=1e-12,
            conv_tol_density=1e-10,
            conv_tol_grad=1e-9,
            max_cycle=150,
        ).run()
        trials.append(
            {
                "amplitude": amplitude,
                "energy": float(trial.e_tot),
                "converged": bool(trial.converged),
            }
        )
        choices.append(trial)
        print(json.dumps({"nitrogen_scf_trial": trials[-1]}), flush=True)
    valid = [(i, m) for i, m in enumerate(choices) if m.converged]
    index, selected = min(valid, key=lambda pair: float(pair[1].e_tot))
    result = run_case(case, mean_field=selected)
    result.update(
        followup="native_rotated_density",
        scf_trials=trials,
        selected_scf_trial=index,
        scf_rotation_seed=20260923,
    )
    return result


def carbon_monoxide():
    _, basis, atom = next(c for c in CASES if c[0] == "CO")
    mf = dft.RKS(
        gto.M(atom=atom, basis=basis, unit="Angstrom"), xc="hf", conv_tol=1e-12
    ).run()
    obj = cc.CCSD(mf, conv_tol=1e-12, residual_tol=1e-11, max_cycle=200).run()
    p = py_scf.RHF(py_gto.M(atom=atom, basis=basis, unit="Angstrom", verbose=0)).run(
        conv_tol=1e-13
    )
    pcc = p.CCSD().run(conv_tol=1e-13, conv_tol_normt=1e-12, max_cycle=200)
    oracle = eom_rccsd.EOMEESinglet(pcc)
    imds = oracle.make_imds()
    matrix = np.column_stack(
        [oracle.matvec(v, imds) for v in np.eye(oracle.vector_size())]
    )
    ref = np.linalg.eigvals(matrix)
    ref = ref[np.lexsort((ref.imag, ref.real))][:3]
    rows = []
    for method, space, seed, tol in (
        ("davidson", 48, 0, 1e-9),
        ("davidson", 48, 1, 1e-9),
        ("davidson", 64, 0, 1e-9),
        ("davidson", 48, 0, 1e-10),
        ("dense", 48, 0, 1e-9),
    ):
        start = perf_counter()
        e = cc.EOMEE(
            obj,
            nroots=3,
            solver=method,
            max_space=space,
            seed=seed,
            conv_tol=tol,
            max_cycle=180,
            max_dense=300,
        ).run()
        out = e.result
        raw = np.asarray(out.raw_eigenvalues)
        row = {
            "method": method,
            "max_space": space,
            "seed": seed,
            "tolerance": tol,
            "energies": np.asarray(e.e).tolist(),
            "error": float(np.max(abs(np.asarray(e.e) - ref))),
            "converged": np.asarray(out.converged).tolist(),
            "response_valid": np.asarray(out.response_valid).tolist(),
            "raw_first_real": raw[:4].real.tolist(),
            "raw_first_imag": raw[:4].imag.tolist(),
            "right_singular_values": np.linalg.svd(
                np.asarray(e.right_vectors), compute_uv=False
            ).tolist(),
            "left_singular_values": np.linalg.svd(
                np.asarray(e.left_vectors), compute_uv=False
            ).tolist(),
            "biorthogonality_error": float(out.biorthogonality_error),
            "condition_numbers": np.asarray(out.condition_numbers).tolist(),
            "right_residuals": np.asarray(out.residual_norms).tolist(),
            "left_residuals": np.asarray(out.left_residual_norms).tolist(),
            "guard_residuals": np.asarray(out.guard_residual_norms).tolist(),
            "iterations": int(out.iterations),
            "seconds": perf_counter() - start,
        }
        rows.append(row)
        print(json.dumps({"co_ee_trial": row}), flush=True)
    return {
        "molecule": "CO",
        "followup": "degenerate_left_right_pairs",
        "reference": ref.real.tolist(),
        "trials": rows,
    }


if __name__ == "__main__":
    print(json.dumps(nitrogen()), flush=True)
    jax.clear_caches()
    print(json.dumps(carbon_monoxide()), flush=True)
