"""Eight-system post-fix EOM sweep; explicit native multistart only for N2.

Single-run SCF defaults are unchanged. All attempts, including the higher N2
branch, are retained. PySCF is used only inside the independent comparison.
"""

import gc
import json
from dataclasses import asdict
import jax
from compare_eom_molecules import CASES, run_case
from gradscf import gto, dft

jax.config.update("jax_enable_x64", True)

if __name__ == "__main__":
    for case in CASES:
        selected = None
        selection = None
        if case[0] == "N2":
            mf = dft.RKS(
                gto.M(atom=case[2], basis=case[1], unit="Angstrom"),
                xc="hf",
                conv_tol=1e-12,
                conv_tol_density=1e-10,
                conv_tol_grad=1e-9,
                max_cycle=150,
            )
            selection = mf.multistart(amplitudes=(0.05, 0.15, 0.4), seed=20260923)
            if not selection.converged:
                raise RuntimeError("No converged N2 restart")
            selected = selection.selected
        report = run_case(case, mean_field=selected)
        if selection is not None:
            report["scf_attempts"] = [asdict(a) for a in selection.attempts]
            report["selected_scf_attempt"] = selection.selected_index
        print(json.dumps(report), flush=True)
        jax.clear_caches()
        gc.collect()
