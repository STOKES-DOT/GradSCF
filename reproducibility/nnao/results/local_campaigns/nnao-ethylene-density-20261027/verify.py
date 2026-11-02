"""Verify the two-plane frozen-basis density comparison without rerunning SCF."""
import json
import math
from pathlib import Path
import numpy as np
from pyscf import gto, lib
from density import evaluate_ao, evaluate_density

OUT = Path(__file__).resolve().parent


def main():
    lib.num_threads(1)
    result = json.loads((OUT / "results.json").read_text())
    assert result["status"] == "complete" and len(result["records"]) == 6
    row = json.loads((OUT / "inputs/structure.json").read_text())
    angstrom_to_bohr = json.loads((OUT / "configuration.json").read_text())["angstrom_to_bohr"]
    bases = json.loads((OUT / "inputs/frozen-bases.json").read_text())
    assert len(bases) == 6
    centers = np.asarray(row["coords_angstrom"]) * angstrom_to_bohr
    rng = np.random.default_rng(0)
    probes = np.vstack((np.zeros((1, 3)), rng.normal(size=(200, 3))))
    polynomial_errors = []
    for l in range(4):
        shell = [l, [.7, .4], [.2, .9]]
        mol = gto.M(atom="C 0 0 0", basis={"C": [shell]}, spin=0, cart=False, verbose=0)
        delta = evaluate_ao(probes, np.zeros((1, 3)), [[shell]]) - mol.eval_gto("GTOval_sph", probes)
        polynomial_errors.append(float(np.max(abs(delta))))
    assert max(polynomial_errors) < 1e-12
    checks = []
    with np.load(OUT / "arrays/plane-densities.npz") as arrays:
        x, y = arrays["x_angstrom"], arrays["vertical_angstrom"]
        assert float(arrays["angstrom_to_bohr"]) == angstrom_to_bohr
        assert x.shape == (641,) and y.shape == (561,)
        xx, yy = np.meshgrid(x, y)
        for entry, record in zip(bases, result["records"]):
            key = entry["key"]
            assert key == record["key"]
            state = dict(np.load(OUT / f"states/{key}.npz"))
            s, c, occ, d = (state[k] for k in ("overlap", "mo", "occupations", "density"))
            np.testing.assert_allclose(c.T @ s @ c, np.eye(len(s)), atol=1e-9, rtol=0)
            np.testing.assert_allclose((c * occ) @ c.T, d, atol=1e-8, rtol=0)
            assert abs(np.einsum("ij,ji", d, s) - 16) < 1e-8
            assert record["converged"] and record["internal_stable"]
            assert record["pyscf_energy_error_hartree"] < 1e-8
            labels = entry["atom_labels"]
            mol = gto.M(atom=list(zip(labels, row["coords_angstrom"])),
                basis=dict(zip(labels, entry["atom_shells"])), unit="Angstrom", cart=False, verbose=0)
            maximum_error = 0.
            for plane in ("xy", "xz"):
                rho = arrays[f"{plane}_{key}"]
                assert rho.shape == (561, 641) and np.isfinite(rho).all() and rho.min() >= 0
                selected = rng.choice(rho.size, 2048, replace=False)
                horizontal, vertical = xx.ravel()[selected], yy.ravel()[selected]
                points = np.stack([horizontal, vertical, np.zeros_like(horizontal)] if plane == "xy" else
                                  [horizontal, np.zeros_like(horizontal), vertical], axis=1) * angstrom_to_bohr
                ao = mol.eval_gto("GTOval_sph", points)
                reference = np.einsum("pi,ij,pj->p", ao, d, ao)
                actual = rho.ravel()[selected]
                error = float(np.max(abs(reference - actual)))
                assert error < 1e-8, (key, plane, error)
                maximum_error = max(maximum_error, error)
            checks.append(dict(key=key, grid_points_checked=4096, max_density_difference=maximum_error))
    # Fixed exponent pools and fixed carbon core are identical between both NNAO modes.
    for atom, (a, b) in enumerate(zip(bases[0]["atom_shells"], bases[1]["atom_shells"])):
        assert len(a) == len(b)
        for sa, sb in zip(a, b):
            assert sa[0] == sb[0]
            np.testing.assert_array_equal(np.asarray(sa[1:])[:, 0], np.asarray(sb[1:])[:, 0])
        if row["symbols"][atom] == "C":
            assert a[0] == b[0]
    optimizations = [json.loads((OUT / f"optimization/{name}-result.json").read_text())
                     for name in ("mace", "reference")]
    assert all(r["accepted"] and r["gradient_norm"] <= 1e-5 for r in optimizations)
    energies = {r["key"]: r["energy_hartree"] for r in result["records"]}
    assert energies["free"] <= energies["mace"] + 1e-9
    report = dict(status="passed", passed_scf=6, planes=2, density_maps=12,
        electron_number=16, ao_polynomial_max_errors_spdf=polynomial_errors,
        independent_sampled_grid_points=6 * 4096, density_checks=checks,
        max_pyscf_energy_error_hartree=max(r["pyscf_energy_error_hartree"] for r in result["records"]),
        free_gradient_norms=[r["gradient_norm"] for r in optimizations],
        free_endpoint_spread_hartree=abs(optimizations[0]["energy_hartree"] - optimizations[1]["energy_hartree"]),
        validation="sampled same-density-matrix pointwise density; full frozen-basis SCF energy; analytic Tr(DS)",
        finite_differences=False)
    (OUT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
