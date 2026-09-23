"""Bounded native SCF search and independent FCI/CC diagnostics for F2 at 2.2 A."""

import json
from dataclasses import asdict
from time import perf_counter
import jax
import numpy as np
from pyscf import gto as py_gto, scf as py_scf, fci
from pyscf.cc import eom_rccsd
from pyscf.cc import rccsd
from gradscf import gto, dft, cc
from compare_eom_stress_followups import reference_diagnostics

jax.config.update("jax_enable_x64", True)

if __name__ == "__main__":
    start = perf_counter()
    atom = "F 0 0 0; F 0 0 2.2"
    mol = gto.M(atom=atom, basis="sto-3g", unit="Angstrom")
    source = dft.RKS(
        mol,
        xc="hf",
        conv_tol=1e-12,
        conv_tol_density=1e-10,
        conv_tol_grad=1e-9,
        max_cycle=250,
    )
    candidates = []
    attempts = []
    for seed in (0, 1, 20260923):
        selected = source.multistart(amplitudes=(0.25, 0.5, 1.0), seed=seed)
        row = {
            "seed": seed,
            "selected": selected.selected_index,
            "attempts": [asdict(a) for a in selected.attempts],
        }
        attempts.append(row)
        if selected.converged:
            candidates.append(selected.selected)
        print(json.dumps({"native_search": row}), flush=True)
    native = min(candidates, key=lambda m: float(m.e_tot))
    ground = cc.CCSD(native, conv_tol=1e-12, residual_tol=1e-11, max_cycle=400).run()
    result = {
        "molecule": "F2_2.2",
        "native_hf": float(native.e_tot),
        "native_ccsd": float(ground.e_tot),
        "native_cc_converged": bool(ground.converged),
        "native_cc_residual": float(ground.result.residual_norm),
        "scf_attempts": attempts,
    }
    pref = py_scf.RHF(
        py_gto.M(atom=atom, basis="sto-3g", unit="Angstrom", verbose=0)
    ).run(conv_tol=1e-13)
    result["reference_hf"] = float(pref.e_tot)
    ci = fci.FCI(pref)
    ci.conv_tol = 1e-12
    energies, vectors = ci.kernel(nroots=6)
    result["fci"] = [
        {
            "energy": float(e),
            "spin_square": float(fci.spin_op.spin_square(v, 10, (9, 9))[0]),
        }
        for e, v in zip(energies, vectors)
    ]
    print(json.dumps({"fci": result["fci"]}), flush=True)
    oracle = rccsd.RCCSD(pref).run(conv_tol=1e-13, conv_tol_normt=1e-12, max_cycle=600)
    result["reference_rccsd_initial"] = reference_diagnostics(pref, oracle)
    if not oracle.converged:
        oracle.max_cycle = 5000
        oracle.kernel(t1=oracle.t1, t2=oracle.t2)
    result["reference_rccsd"] = reference_diagnostics(pref, oracle)
    ea_energies, _ = ci.kernel(nelec=(10, 9), nroots=3)
    result["fci_ea"] = [float(e - energies[0]) for e in ea_energies]
    singlets = [x["energy"] for x in result["fci"] if abs(x["spin_square"]) < 1e-5]
    result["fci_ee"] = [e - energies[0] for e in singlets[1:4]]
    print(json.dumps({"reference_rccsd": result["reference_rccsd"]}), flush=True)
    if ground.converged:
        result["eom"] = {}
        for cls, pycls in (
            (cc.EOMEE, eom_rccsd.EOMEESinglet),
            (cc.EOMIP, eom_rccsd.EOMIP),
            (cc.EOMEA, eom_rccsd.EOMEA),
        ):
            eom = cls(
                ground, nroots=3, solver="davidson", max_space=48, max_cycle=180
            ).run()
            result["eom"][eom.sector] = {
                "energies": np.asarray(eom.e).tolist(),
                "converged": np.asarray(eom.converged).tolist(),
                "response_valid": np.asarray(eom.result.response_valid).tolist(),
                "right_residuals": np.asarray(eom.result.residual_norms).tolist(),
                "left_residuals": np.asarray(eom.result.left_residual_norms).tolist(),
            }
            if oracle.converged:
                peom = pycls(oracle)
                imds = peom.make_imds()
                matrix = np.column_stack(
                    [peom.matvec(v, imds) for v in np.eye(peom.vector_size())]
                )
                spectrum = np.linalg.eigvals(matrix)
                spectrum = spectrum[np.lexsort((spectrum.imag, spectrum.real))][:3]
                result["eom"][eom.sector][
                    "reference_spectrum_real"
                ] = spectrum.real.tolist()
                result["eom"][eom.sector][
                    "reference_spectrum_imag"
                ] = spectrum.imag.tolist()
                result["eom"][eom.sector]["reference_max_error"] = float(
                    np.max(abs(np.asarray(eom.e) - spectrum))
                )
    result["seconds_including_compile"] = perf_counter() - start
    print(json.dumps(result), flush=True)
