"""OH/STO-3G: GradSCF UHF -> CI/CC densities and canonical UCCSD(T).

Fixed geometry in Angstrom; energies in Hartree. No PySCF or CLI.
These MO densities include amplitude/Lambda response, not orbital relaxation.
"""
import jax
import numpy as np

from gradscf import cc, ci, gto, scf

jax.config.update("jax_enable_x64", True)
mol = gto.M(atom="O 0 0 0; H 0 0 0.97", basis="sto-3g", spin=1)
mf = scf.UHF(mol, conv_tol=1e-12, max_cycle=150).run()
myci = ci.CISD(mf, conv_tol=1e-10).run()
mycc = cc.CCSD(mf, conv_tol=1e-12, residual_tol=1e-10).run()

print("UHF energy:", float(mf.e_tot))
print("UCISD energy:", float(myci.e_tot))
print("UCCSD energy:", float(mycc.e_tot))
print("UCCSD(T) energy:", float(mycc.e_tot + mycc.ccsd_t()))

for name, method in (("UCISD", myci), ("UCCSD", mycc)):
    dm1 = method.make_rdm1()
    dm2 = method.make_rdm2()
    reference = method.reference
    energy = float(reference.nuclear_repulsion)
    energy += sum(np.einsum("pq,qp", h, d) for h, d in zip(reference.h1, dm1))
    energy += sum(weight*np.einsum("pqrs,pqrs", g, d)
                  for weight, g, d in zip((.5, 1., .5), reference.eri, dm2))
    np.testing.assert_allclose([np.trace(d) for d in dm1], (5, 4), atol=1e-9)
    np.testing.assert_allclose(energy, method.e_tot, atol=1e-8, rtol=0)
    print(name, "density traces:", [float(np.trace(d)) for d in dm1])
    print(name, "density energy error:", energy-float(method.e_tot))
print("Lambda converged:", mycc.converged_lambda)
