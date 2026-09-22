"""OH/STO-3G UHF and ROHF triples with GradSCF's tensor resolvent.

The semicanonical option is invariant to separate active occupied/virtual
rotations. Distances are Angstrom, energies Hartree. No PySCF or CLI.
"""
import jax
import numpy as np
from gradscf import cc, gto, scf

jax.config.update("jax_enable_x64", True)
mol = gto.M(atom="O 0 0 0; H 0 0 .97", basis="sto-3g", spin=1)
for reference in (scf.UHF, scf.ROHF):
    mf = reference(mol, conv_tol=1e-12, max_cycle=150).run()
    mycc = cc.CCSD(mf, conv_tol=1e-12, residual_tol=1e-10, max_cycle=150).run()
    triples = mycc.triples(orbital_basis="semicanonical")
    print(reference.__name__, "CCSD:", float(mycc.e_tot))
    print("(T):", float(triples.energy))
    print("CCSD(T):", float(mycc.e_tot+triples.energy))
    print("Input Fock off-diagonal maximum:", float(triples.canonical_error))
    print("Minimum denominator:", float(triples.min_abs_denominator))
    if reference is scf.UHF:
        np.testing.assert_allclose(triples.energy, mycc.ccsd_t(), atol=1e-11, rtol=0)
