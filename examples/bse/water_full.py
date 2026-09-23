"""Native RHF/G0W0 water: compare TDA and stable full BSE optical excitations.

STO-3G, coordinates in Angstrom, float64. Full BSE compares the bounded dense
reference solver with matrix-free Davidson and reports stability diagnostics. No CLI.
"""

import jax
from gradscf import bse, dft, gto, gw
from gradscf.tools.spectra import HARTREE_TO_EV

jax.config.update("jax_enable_x64", True)
mol = gto.M(atom="O 0 0 0; H 0 -.757 .587; H 0 .757 .587", basis="sto-3g")
mf = dft.RKS(mol, xc="hf", conv_tol=1e-12).run()
mygw = gw.GW(mf, nw=100).run()
print("G0W0 converged:", bool(mygw.converged))

for singlet in (True, False):
    print("Singlet" if singlet else "Triplet")
    for tda, method in ((True, "davidson"), (False, "dense"), (False, "davidson")):
        response = bse.BSE(
            mygw,
            tda=tda,
            solver=method,
            singlet=singlet,
            nroots=3,
            conv_tol=1e-10,
        ).run()
        print("TDA" if tda else "Full BSE", method)
        print("Energies / eV:", response.e * HARTREE_TO_EV)
        print("Oscillator strengths:", response.oscillator_strength())
        print("Residuals / Hartree:", response.result.residual_norms)
        if not tda:
            print(
                "Stability margins / Hartree:",
                response.result.stability_margins,
            )

            print("Global stability certified:", response.result.stability_certified)
