"""Independent PySCF checks for native experiment outputs (test-only)."""
import numpy as np


def validate_h2_contraction_endpoints(summary):
    """Check both endpoint energies with independent PySCF SCF/integrals."""
    from pyscf import gto, scf
    energies = []
    for basis in ("3-21g", summary["optimized_basis"]):
        mol = gto.M(atom=f"H 0 0 0; H 0 0 {summary['bond_angstrom']}", basis=basis,
                    unit="Angstrom", cart=True, verbose=0)
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-9
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("Independent PySCF validation did not converge")
        energies.append(float(mf.e_tot))
    errors = np.asarray(energies)-[summary["initial_energy_hartree"], summary["final_energy_hartree"]]
    np.testing.assert_allclose(errors, 0., atol=1e-8, rtol=0)
    return dict(initial_energy_hartree=energies[0], final_energy_hartree=energies[1],
                max_energy_error_hartree=float(np.max(np.abs(errors))))


def validate_geometry(summary):
    """Compare total energy and nuclear derivatives with a converged PySCF RHF."""
    from pyscf import gto, scf

    mol = gto.M(atom=summary["atom_angstrom"], basis=summary["basis"],
                unit="Angstrom", cart=True, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-10
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("PySCF geometry reference did not converge")
    gradient = mf.nuc_grad_method().kernel()
    np.testing.assert_allclose(summary["energy_hartree"], mf.e_tot, atol=1e-8, rtol=0)
    np.testing.assert_allclose(summary["gradient"], gradient, atol=2e-6, rtol=2e-5)
    return dict(pyscf_energy_hartree=float(mf.e_tot), pyscf_gradient=gradient.tolist(),
                max_ad_pyscf_error=float(np.max(np.abs(np.asarray(summary["gradient"])-gradient))))


def validate_contraction_endpoints(summary):
    """Check native contraction experiment endpoints with independent integrals."""
    from pyscf import gto, scf

    energies = []
    for basis in (summary["initial_basis"], summary["optimized_basis"]):
        mol = gto.M(atom=summary["atom_angstrom"], basis=basis,
                    unit="Angstrom", cart=True, verbose=0)
        mf = scf.RHF(mol)
        mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-9
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("PySCF endpoint reference did not converge")
        energies.append(float(mf.e_tot))
    errors = np.asarray(energies)-[summary["initial"]["energy_hartree"],
                                  summary["final"]["energy_hartree"]]
    np.testing.assert_allclose(errors, 0., atol=1e-8, rtol=0)
    return dict(pyscf_endpoint_energies=energies,
                max_pyscf_energy_error=float(np.max(np.abs(errors))))


def main():
    """Validate a saved native summary without adding a production dependency."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("geometry", "h2-contractions", "contractions"))
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    validators = {"geometry": validate_geometry,
                  "h2-contractions": validate_h2_contraction_endpoints,
                  "contractions": validate_contraction_endpoints}
    print(json.dumps(validators[args.kind](json.loads(args.summary.read_text())), indent=2))


if __name__ == "__main__":
    main()
