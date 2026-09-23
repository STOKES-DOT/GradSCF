"""Native GradSCF/PySCF EE/IP/EA validation across eight closed-shell molecules.

No CLI. Run from the repository root on CPU. Geometries are fixed numerical
fixtures (Angstrom), not optimized structures or spectroscopic benchmarks.
One JSON record per molecule is printed so interrupted sweeps retain results.
"""

import gc
import json
import platform
from time import perf_counter
import jax
import numpy as np
import pyscf
from pyscf import gto as py_gto, scf as py_scf
from pyscf.cc import eom_rccsd

jax.config.update("jax_enable_x64", True)
from gradscf import gto, dft, cc

CASES = [
    ("LiH", "6-31g", "Li 0 0 0; H 0 0 1.6"),
    ("HF", "6-31g", "F 0 0 0; H 0 0 .917"),
    ("N2", "sto-3g", "N 0 0 0; N 0 0 1.10"),
    ("CO", "sto-3g", "C 0 0 0; O 0 0 1.128"),
    (
        "NH3",
        "sto-3g",
        "N 0 0 0; "
        + "; ".join(
            f"H {0.94*np.cos(t):.16g} {0.94*np.sin(t):.16g} -0.38"
            for t in (0.0, 2 * np.pi / 3, 4 * np.pi / 3)
        ),
    ),
    (
        "CH4",
        "sto-3g",
        "C 0 0 0; H .628 .628 .628; H .628 -.628 -.628; H -.628 .628 -.628; H -.628 -.628 .628",
    ),
    ("CH2O", "sto-3g", "C 0 0 0; O 0 0 1.21; H .935 0 -.60; H -.935 0 -.60"),
    (
        "C2H4",
        "sto-3g",
        "C -.67 0 0; C .67 0 0; H -1.232 .929 0; H -1.232 -.929 0; H 1.232 .929 0; H 1.232 -.929 0",
    ),
]


def run_case(case, *, mean_field=None):
    name, basis, atom = case
    start = perf_counter()
    row = {"molecule": name, "basis": basis, "atom_angstrom": atom, "sectors": {}}
    if mean_field is None:
        mf = dft.RKS(
            gto.M(atom=atom, basis=basis, unit="Angstrom"), xc="hf", conv_tol=1e-12
        ).run()
    else:
        mf = mean_field
    row["scf_source"] = (
        "native_hcore" if mean_field is None else "supplied_native_restart"
    )
    mycc = cc.CCSD(mf, conv_tol=1e-12, residual_tol=1e-11, max_cycle=200).run()
    pm = py_gto.M(atom=atom, basis=basis, unit="Angstrom", verbose=0)
    pref = py_scf.RHF(pm).run(conv_tol=1e-13)
    pcc = pref.CCSD().run(conv_tol=1e-13, conv_tol_normt=1e-12, max_cycle=200)
    row.update(
        hf_energy=float(mf.e_tot),
        ccsd_energy=float(mycc.e_tot),
        hf_error=float(abs(mf.e_tot - pref.e_tot)),
        ccsd_error=float(abs(mycc.e_tot - pcc.e_tot)),
        cc_converged=bool(mycc.converged),
        reference_cc_converged=bool(pcc.converged),
        cc_residual=float(mycc.result.residual_norm),
    )
    if not mycc.converged or not pcc.converged:
        row["status"] = "ground_not_converged"
        row["seconds_including_compile"] = perf_counter() - start
        return row
    for cls, oracle_cls in (
        (cc.EOMEE, eom_rccsd.EOMEESinglet),
        (cc.EOMIP, eom_rccsd.EOMIP),
        (cc.EOMEA, eom_rccsd.EOMEA),
    ):
        begin = perf_counter()
        eom = cls(
            mycc,
            nroots=3,
            solver="davidson",
            max_space=48,
            max_cycle=180,
            conv_tol=1e-9,
            seed=0,
        ).run()
        out = eom.result
        oracle = oracle_cls(pcc)
        imds = oracle.make_imds()
        size = int(oracle.vector_size())
        matrix = np.empty((size, size))
        # Full reference matrix ONLY; avoid allocating an additional identity.
        for i in range(size):
            probe = np.zeros(size)
            probe[i] = 1.0
            matrix[:, i] = oracle.matvec(probe, imds)
        spectrum = np.linalg.eigvals(matrix)
        spectrum = spectrum[np.lexsort((spectrum.imag, spectrum.real))]
        reference = spectrum[:3]
        distances = np.abs(reference[:, None] - spectrum[None, :])
        distances[np.arange(3), np.arange(3)] = np.inf
        gaps = distances.min(axis=1)
        actual = np.asarray(eom.e)
        error = float(np.max(abs(actual - reference)))
        forward_ok = bool(
            np.all(out.converged)
            and np.isfinite(error)
            and error < 1e-8
            and np.max(abs(reference.imag)) < 1e-9
        )
        info = {
            "dimension": size,
            "energies_hartree": actual.tolist(),
            "reference_real": reference.real.tolist(),
            "reference_imag": reference.imag.tolist(),
            "reference_gaps": gaps.tolist(),
            "max_error": error,
            "converged": np.asarray(out.converged).tolist(),
            "response_valid": np.asarray(out.response_valid).tolist(),
            "right_residuals": np.asarray(out.residual_norms).tolist(),
            "left_residuals": np.asarray(out.left_residual_norms).tolist(),
            "guard_residuals": np.asarray(out.guard_residual_norms).tolist(),
            "ritz_gaps": np.asarray(out.spectral_gaps).tolist(),
            "condition_numbers": np.asarray(out.condition_numbers).tolist(),
            "biorthogonality_error": float(out.biorthogonality_error),
            "iterations": int(out.iterations),
            "restarts": int(out.restarts),
            "subspace_dimension": int(out.subspace_dimension),
            "spectrum_complete": bool(out.spectrum_complete),
            "forward_ok": forward_ok,
            "seconds_including_compile": perf_counter() - begin,
        }
        row["sectors"][eom.sector] = info
        print(
            json.dumps(
                {
                    "progress": name,
                    "sector": eom.sector,
                    "forward_ok": forward_ok,
                    "error": error,
                    "response_valid": info["response_valid"],
                    "iterations": info["iterations"],
                }
            ),
            flush=True,
        )
    row["status"] = (
        "passed" if all(x["forward_ok"] for x in row["sectors"].values()) else "inspect"
    )
    row["seconds_including_compile"] = perf_counter() - start
    return row


if __name__ == "__main__":
    print(
        json.dumps(
            {
                "environment": {
                    "machine": platform.machine(),
                    "python": platform.python_version(),
                    "jax": jax.__version__,
                    "pyscf": pyscf.__version__,
                    "devices": [str(d) for d in jax.devices()],
                    "dtype": "float64",
                    "eom": {
                        "nroots": 3,
                        "guard_roots": 1,
                        "max_space": 48,
                        "max_cycle": 180,
                        "residual_tolerance": 1e-9,
                        "seed": 0,
                    },
                    "energy_tolerance": 1e-8,
                }
            }
        ),
        flush=True,
    )
    for case in CASES:
        try:
            report = run_case(case)
        except Exception as error:
            import traceback

            report = {
                "molecule": case[0],
                "basis": case[1],
                "status": "error",
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
        print(json.dumps(report), flush=True)
        jax.clear_caches()
        gc.collect()
