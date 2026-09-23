"""Controlled reference/SCF checks after the extended EOM sweep; no CLI."""

import gc
import json
from dataclasses import asdict
import jax
import numpy as np
from pyscf import gto as py_gto, scf as py_scf, ao2mo
from gradscf import gto, dft
from gradscf.cc.integrals import prepare_integrals
from gradscf.cc.rccsd import residual
from compare_eom_molecules import run_case
from compare_eom_stress import CASES

jax.config.update("jax_enable_x64", True)


def reference_diagnostics(mf, obj):
    eris = obj.ao2mo()
    t1, t2 = obj.update_amps(obj.t1, obj.t2, eris)
    n = mf.mo_coeff.shape[1]
    h = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    g = ao2mo.restore(1, ao2mo.kernel(mf.mol, mf.mo_coeff), n)
    no = int(mf.mol.nelectron // 2)
    r1, r2 = residual(obj.t1, obj.t2, prepare_integrals(h, g, nocc=no), model="ccsd")
    return {
        "converged": bool(obj.converged),
        "energy": float(obj.e_tot),
        "update_norm": float(np.linalg.norm(t1 - obj.t1) + np.linalg.norm(t2 - obj.t2)),
        "physical_residual_max": max(float(np.max(abs(r1))), float(np.max(abs(r2)))),
        "conv_tol": obj.conv_tol,
        "conv_tol_normt": obj.conv_tol_normt,
        "max_cycle": obj.max_cycle,
    }


def difficult_reference(name):
    case = next(c for c in CASES if c[0] == name)
    _, basis, atom = case
    pref = py_scf.RHF(py_gto.M(atom=atom, basis=basis, unit="Angstrom", verbose=0)).run(
        conv_tol=1e-13
    )
    pcc = pref.CCSD().run(conv_tol=1e-13, conv_tol_normt=1e-12, max_cycle=200)
    history = [reference_diagnostics(pref, pcc)]
    print(json.dumps({"reference_probe": name, "attempt": history[-1]}), flush=True)
    pcc.max_cycle = 600
    pcc.diis_space = 8
    pcc.kernel(t1=pcc.t1, t2=pcc.t2)
    history.append(reference_diagnostics(pref, pcc))
    print(json.dumps({"reference_probe": name, "attempt": history[-1]}), flush=True)
    if not pcc.converged:
        pcc.conv_tol = 1e-12
        pcc.conv_tol_normt = 1e-10
        pcc.kernel(t1=pcc.t1, t2=pcc.t2)
        history.append(reference_diagnostics(pref, pcc))
        print(json.dumps({"reference_probe": name, "attempt": history[-1]}), flush=True)
    mf = dft.RKS(
        gto.M(atom=atom, basis=basis, unit="Angstrom"),
        xc="hf",
        conv_tol=1e-12,
        conv_tol_density=1e-10,
        conv_tol_grad=1e-9,
        max_cycle=150,
    )
    if name.startswith("F2"):
        selection = mf.multistart(amplitudes=(0.05, 0.15, 0.4), seed=20260923)
        if not selection.converged:
            raise RuntimeError("No native F2 candidate converged")
        mf = selection.selected
    else:
        selection = None
        mf.run()
    out = run_case(case, mean_field=mf, reference_mean_field=pref, reference_cc=pcc)
    out["followup"] = "reference_restart_and_native_scf_controls"
    out["reference_attempts"] = history
    if selection is not None:
        out["scf_attempts"] = [asdict(a) for a in selection.attempts]
        out["selected_scf_attempt"] = selection.selected_index
    return out


def helium():
    case = next(c for c in CASES if c[0] == "He2")
    mf = dft.RKS(
        gto.M(atom=case[2], basis=case[1], unit="Angstrom"),
        xc="hf",
        conv_tol=1e-14,
        conv_tol_density=1e-12,
        conv_tol_grad=1e-11,
        max_cycle=150,
    ).run()
    out = run_case(case, mean_field=mf)
    out["followup"] = "tight_native_scf"
    return out


def butadiene():
    y = 1.34 * np.sqrt(3) / 2
    h = 1.09 * np.sqrt(3) / 2
    case = (
        "C4H6_frozen",
        "sto-3g",
        f"C -1.39 {y} 0; C -.72 0 0; C .72 0 0; C 1.39 {-y} 0; H -2.48 {y} 0; H -.845 {y+h} 0; H -1.265 {-h} 0; H 1.265 {h} 0; H .845 {-y-h} 0; H 2.48 {-y} 0",
    )
    # 26 MOs, 15 occupied: freeze four core occupied and seven highest virtuals.
    # This is an explicitly truncated numerical fixture, not full-space CCSD.
    return run_case(case, frozen=[0, 1, 2, 3, 19, 20, 21, 22, 23, 24, 25])


if __name__ == "__main__":
    for function in (
        lambda: difficult_reference("F2_2.2"),
        lambda: difficult_reference("O3"),
        helium,
        butadiene,
    ):
        print(json.dumps(function()), flush=True)
        jax.clear_caches()
        gc.collect()
