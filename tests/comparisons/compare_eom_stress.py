"""Extended closed-shell EOM forward fixtures; failures are retained, no CLI.

Includes atoms, weak interaction, stretched bonds, linear degeneracy and pi
systems. Fixed geometries are numerical tests, not optimized reference data.
"""

import gc
import json
import platform
import traceback
from time import perf_counter
import numpy as np
import jax
import pyscf
from pyscf import gto as py_gto, scf as py_scf
from compare_eom_molecules import run_case
from gradscf import gto, dft

jax.config.update("jax_enable_x64", True)

CASES = [
    ("H2_1.5", "6-31g", "H 0 0 0; H 0 0 1.5"),
    ("H2_2.5", "6-31g", "H 0 0 0; H 0 0 2.5"),
    ("He2", "6-31g", "He 0 0 0; He 0 0 3.0"),
    ("Ne", "6-31g", "Ne 0 0 0"),
    ("CO2", "sto-3g", "C 0 0 0; O 0 0 -1.16; O 0 0 1.16"),
    ("HCN", "sto-3g", "H 0 0 -1.06; C 0 0 0; N 0 0 1.16"),
    ("C2H2", "sto-3g", "H 0 0 -1.66; C 0 0 -.60; C 0 0 .60; H 0 0 1.66"),
    ("F2_1.42", "sto-3g", "F 0 0 0; F 0 0 1.42"),
    ("F2_2.2", "sto-3g", "F 0 0 0; F 0 0 2.2"),
    (
        "O3",
        "sto-3g",
        f"O 0 0 0; O {1.28*np.sin(np.deg2rad(58.4)):.16g} 0 {1.28*np.cos(np.deg2rad(58.4)):.16g}; O {-1.28*np.sin(np.deg2rad(58.4)):.16g} 0 {1.28*np.cos(np.deg2rad(58.4)):.16g}",
    ),
]


def run_stress_case(case, *, multistart=False):
    name, basis, atom = case
    start = perf_counter()
    mol = gto.M(atom=atom, basis=basis, unit="Angstrom")
    mf = dft.RKS(mol, xc="hf", conv_tol=1e-12)
    if multistart:
        from dataclasses import asdict

        mf.max_cycle = 150
        mf.conv_tol_density = 1e-10
        mf.conv_tol_grad = 1e-9
        selection = mf.multistart(amplitudes=(0.05, 0.15, 0.4), seed=20260923)
        if not selection.converged:
            return {
                "molecule": name,
                "basis": basis,
                "status": "scf_not_converged",
                "scf_attempts": [asdict(a) for a in selection.attempts],
            }
        mf = selection.selected
    else:
        mf.run()
    if not mf.converged:
        pref = py_scf.RHF(
            py_gto.M(atom=atom, basis=basis, unit="Angstrom", verbose=0)
        ).run(conv_tol=1e-13)
        return {
            "molecule": name,
            "basis": basis,
            "atom_angstrom": atom,
            "status": "scf_not_converged",
            "hf_energy": float(mf.e_tot),
            "reference_hf_energy": float(pref.e_tot),
            "reference_hf_converged": bool(pref.converged),
            "hf_cycles": int(mf.cycles),
            "seconds_including_compile": perf_counter() - start,
        }
    report = run_case(case, mean_field=mf)
    report["scf_source"] = (
        "explicit_native_multistart" if multistart else "native_hcore"
    )
    report["total_seconds_including_scf"] = perf_counter() - start
    if multistart:
        report["scf_attempts"] = [asdict(a) for a in selection.attempts]
        report["selected_scf_attempt"] = selection.selected_index
    return report


if __name__ == "__main__":
    print(
        json.dumps(
            {
                "environment": {
                    "python": platform.python_version(),
                    "machine": platform.machine(),
                    "jax": jax.__version__,
                    "pyscf": pyscf.__version__,
                    "dtype": "float64",
                    "devices": [str(d) for d in jax.devices()],
                },
                "settings": {
                    "nroots": 3,
                    "guard_roots": 1,
                    "max_space": 48,
                    "max_cycle": 180,
                    "eom_tolerance": 1e-9,
                    "comparison_tolerance": 1e-8,
                    "seed": 0,
                },
            }
        ),
        flush=True,
    )
    for case in CASES:
        try:
            report = run_stress_case(case)
        except Exception as error:
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
