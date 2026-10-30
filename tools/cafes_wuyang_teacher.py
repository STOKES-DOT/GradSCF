#!/usr/bin/env python3
"""Prepare one CAFES CCSD-density Wu--Yang teacher on the remote worker.

This runner is intentionally thin: integral construction and inversion remain
in GradSCF. CAFES QH9 spherical source matrices are mapped by their declared
index/phase metadata, then embedded into the current Cartesian AO space by the
parent-owned normalized-sp transform. HF orbitals/Fock are source checks
only. The Wu--Yang base is kinetic plus local nuclear hcore; no exact
exchange operator enters this local-potential assessment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gradscf import integrals, dft, gto
from gradscf.data.molecule import parse_molecule_spec
from gradscf.dft import iks
from gradscf.integrals.basis import cartesian_to_spherical
from gradscf.integrals.basis.contraction import contraction_matrix
from gradscf.integrals.basis.data import load_basis_from_snapshot
from gradscf.integrals.grids import build_molecular_grid_from_spec
from gradscf.integrals.grids.blocks import AOGrid
from gradscf_tools.cafes_qm9 import ao_square_to_canonical
from gradscf_tools.wuyang_pbs import atom_centered_gaussian_pbs


QUALITY_SETTINGS = {
    "diagnostic_tolerance": 1e-3,
    "density_error_kind": "absolute_grid_L2",
    "moment_error_kind": "normalized_unit_AO_operator_basis_residual",
    "regularization": 1e-6,
    "gradient_tolerance": 1e-8,
    "maxiter": 500,
    "gap_tolerance": 1e-7,
    "interpretation": "workflow-demonstration gate; not exact-density or chemical-accuracy evidence",
}


def _atom_string(atomic_numbers, positions):
    symbols = {1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B",
               6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne"}
    return "; ".join(
        f"{symbols[int(z)]} {float(x):.16g} {float(y):.16g} {float(zcoord):.16g}"
        for z, (x, y, zcoord) in zip(atomic_numbers, positions)
    )


def _metric_natural_occupations(density, overlap):
    values, vectors = np.linalg.eigh(overlap)
    root = (vectors * np.sqrt(values)) @ vectors.T
    return np.linalg.eigvalsh(root @ density @ root)


def prepare(sample_npz: Path, sample_meta: Path, *, max_memory_mb: float,
            grid_level: int, block_size: int, reference_output: Path | None = None,
            guide: str = 'hcore', uncontract: bool = False,
            density_metric: str = 'absolute_l2', inverse_method: str = 'wu_yang',
            regularization: float | None = None, initial_teacher: Path | None = None) -> dict:
    arrays = np.load(sample_npz, allow_pickle=False)
    metadata = json.loads(sample_meta.read_text(encoding="utf-8"))
    atom = _atom_string(arrays["atomic_numbers"], arrays["positions_angstrom"])
    spec = parse_molecule_spec(atom, unit="Angstrom", charge=0, spin=0)

    # prepare_basis(..., cart=True) supplies the topology for the parent-tested
    # map. basis_from_molecule_spec supplies the matching Cartesian AO object.
    topology, parameters = integrals.prepare_basis(
        atom, "def2-svp", unit="Angstrom", charge=0, spin=0, cart=True,
    )
    orbital_basis = 'def2-svp'
    transform = np.asarray(cartesian_to_spherical(topology), dtype=float)
    if uncontract:
        # Enlarge the orbital space while preserving the SAME source density.
        # Source contractions are embedded exactly using the shared normalized
        # primitive contraction map; no density fit or exponent change occurs.
        orbital_basis = {}
        for symbol in dict.fromkeys(spec.symbols):
            shells = load_basis_from_snapshot('def2-svp', symbol)
            orbital_basis[symbol] = [
                [int(shell[0]), *[[float(row[0]), *np.eye(len(shell)-1)[i].tolist()]
                                  for i, row in enumerate(shell[1:])]] for shell in shells]
        transform = np.asarray(contraction_matrix(topology, parameters))@transform
    basis = integrals.basis_from_molecule_spec(spec, basis=orbital_basis, max_l=3)
    source_indices = arrays["ao_source_indices"]
    phases = arrays["ao_phase_signs"]
    source_overlap = ao_square_to_canonical(arrays["S"], source_indices, phases)
    source_density = ao_square_to_canonical(arrays["D_CCSD"], source_indices, phases)
    if source_overlap.shape != (transform.shape[1], transform.shape[1]):
        raise ValueError(
            "CAFES source spherical AO dimension does not match the documented "
            "GradSCF topology transform"
        )
    if basis.nao != transform.shape[0]:
        raise ValueError("GradSCF Cartesian AO dimension disagrees with transform")

    overlap_cart = np.asarray(integrals.overlap_matrix(basis), dtype=float)
    transformed_overlap = transform.T @ overlap_cart @ transform
    overlap_max_abs = float(np.max(np.abs(transformed_overlap - source_overlap)))
    if not np.allclose(transformed_overlap, source_overlap, atol=1e-8, rtol=1e-8):
        raise ValueError(
            "documented QH9 spherical-to-Cartesian overlap check failed; "
            "no density map or normalization adjustment is permitted"
        )
    target_density = np.asarray(transform @ source_density @ transform.T, dtype=float)
    hcore = np.asarray(integrals.build_hcore(basis), dtype=float)
    coords, weights = build_molecular_grid_from_spec(spec, level=grid_level)
    grid = AOGrid.from_basis(basis, coords, weights, block_size=block_size)
    pbs_basis, pbs_settings = atom_centered_gaussian_pbs(np.asarray(spec.coords_bohr))
    potential = iks.Potential(
        grid, overlap_cart, pbs_basis, pbs_settings["nparam"],
        layout="rks", max_memory=max_memory_mb,
    )
    nelec = int(round(float(np.sum(arrays["mo_occ"]))))
    target_energy = float(arrays["E_CCSD"])
    target = iks.Target(target_density, overlap_cart, nelec, energy=target_energy)
    base = hcore
    guide_settings = {'kind': 'kinetic plus local nuclear hcore'}
    if guide == 'pbe':
        guide_mol = gto.M(atom=atom, basis=orbital_basis, cart=True, unit='Angstrom')
        source = dft.RKS(guide_mol, xc='pbe', grids_level=grid_level,
            execution_device='cpu', max_memory=max_memory_mb,
            conv_tol=1e-11, conv_tol_grad=1e-8, max_cycle=100).run()
        if not source.converged:
            raise RuntimeError('Converge the fixed PBE local guide before inversion')
        base = np.asarray(source.scf_result.fock_matrix)
        guide_settings = dict(kind='fixed converged PBE local KS guide', scf_converged=True,
            total_energy_hartree=float(source.e_tot), conv_tol=1e-11, conv_tol_grad=1e-8,
            grid_level=grid_level, candidate_scf_during_inversion=False)
    elif guide != 'hcore':
        raise ValueError('Use a local hcore or pbe guide')
    selected_regularization = (1e-6 if inverse_method == 'wu_yang' else 1e-10) if regularization is None else regularization
    quality_settings = dict(QUALITY_SETTINGS, density_error_kind=density_metric,
        inverse_method=inverse_method, regularization=selected_regularization,
        moment_error_role='acceptance' if inverse_method == 'wu_yang' else 'diagnostic_only')
    if density_metric == 'relative_l2':
        quality_settings['density_error_definition'] = 'sqrt(integral (n-n_target)^2 / integral n_target^2)'
        quality_settings['interpretation'] = 'approximate local iKS reference for loop verification; not an exact-potential benchmark'
    initial = None
    if initial_teacher is not None:
        previous = json.loads(Path(initial_teacher).read_text())
        raw_hash = hashlib.sha256(Path(sample_npz).read_bytes()).hexdigest()
        if (previous['sample_id'] != str(arrays['sample_id'].item())
                or previous.get('raw_npz_sha256') != raw_hash
                or previous.get('uncontracted_orbital_space', False) != uncontract
                or previous['smooth_pbs'] != pbs_settings
                or previous['gap'] <= QUALITY_SETTINGS['gap_tolerance']):
            raise ValueError('Initial local potential must match the same input, PBS, orbital space and resolved gap')
        initial = np.asarray(previous['fitted_potential_coefficients'])
    result = iks.invert(
        target, potential, base, method=inverse_method, density_metric=density_metric,
        regularization=selected_regularization, initial=initial,
        tolerance=QUALITY_SETTINGS["diagnostic_tolerance"],
        gradient_tolerance=QUALITY_SETTINGS["gradient_tolerance"],
        maxiter=QUALITY_SETTINGS["maxiter"],
        rank_tolerance=1e-10,
        gap_tolerance=QUALITY_SETTINGS["gap_tolerance"],
    )
    reference_artifact = None
    if result.valid and reference_output is not None:
        reference_output.parent.mkdir(parents=True, exist_ok=True)
        ref = result.reference
        np.savez_compressed(reference_output, coefficients=np.asarray(ref.coefficients),
            occupations=np.asarray(ref.occupations), energies=np.asarray(ref.energies),
            density_target=target_density, density_ks=np.asarray(ref.density), overlap=overlap_cart,
            fock=np.asarray(ref.fock), hcore=hcore, grid_coords=np.asarray(coords),
            grid_weights=np.asarray(weights), energy_total=np.asarray(target_energy),
            atomic_numbers=arrays['atomic_numbers'], positions_angstrom=arrays['positions_angstrom'],
            orbital_basis_json=np.asarray(json.dumps(orbital_basis)))
        reference_artifact = {'path': str(reference_output),
            'sha256': hashlib.sha256(reference_output.read_bytes()).hexdigest()}
    natural = _metric_natural_occupations(target_density, overlap_cart)
    return {
        "sample_id": str(arrays["sample_id"].item()),
        "molecule_id": str(arrays["molecule_id"].item()),
        "reference_family": "CAFES-QM9_CCSD_local_"+result.method,
        "reference_method": result.method+"_from_unrelaxed_CCSD_Lambda_1RDM",
        "inverse_method": result.method,
        "ccsd_unrelaxed_lambda_1rdm": True,
        "ccsd_density_used_as_target": True,
        "hf_orbitals_used_as_target": False,
        "hf_fock_used_as_local_base": False,
        "base_fock_kind": guide_settings["kind"],
        "guide_settings": guide_settings,
        "exact_exchange_used": False,
        "target_kind": "total_energy",
        "target_label": "E_CCSD",
        "target_energy": target_energy,
        "target_includes_nuclear_repulsion": True,
        "source_basis": "CAFES QH9 spherical AO order after declared source index/phase map",
        "inversion_basis": "uncontracted Cartesian def2-SVP" if uncontract else "Cartesian def2-SVP",
        "orbital_basis_definitions": orbital_basis,
        "uncontracted_orbital_space": uncontract,
        "inversion_nao": basis.nao,
        "source_to_cartesian_transform": "D_cart=T D_sph T.T; S_sph=T.T S_cart T",
        "canonical_overlap_max_abs": overlap_max_abs,
        "canonical_overlap_atol_rtol": [1e-8, 1e-8],
        "ccsd_metric_natural_occupation_min": float(natural[0]),
        "ccsd_metric_natural_occupation_max": float(natural[-1]),
        "smooth_pbs": pbs_settings,
        "grid_level": grid_level,
        "quality_settings": quality_settings,
        "reference_artifact": reference_artifact,
        "optimizer_method": result.optimizer_method,
        "optimizer_evaluations": result.optimizer_evaluations,
        "refinement_evaluations": result.refinement_evaluations,
        "spectral_converged": result.spectral_converged,
        "eigen_residual_max": result.eigen_residual_max,
        "preprojection_symmetry_error": result.preprojection_symmetry_error,
        "valid": bool(result.valid),
        "converged": bool(result.converged),
        "density_error": float(result.density_error),
        "relative_density_error": float(result.relative_density_error),
        "density_metric": result.density_metric,
        "moment_error": float(result.moment_error),
        "raw_moment_error": float(result.raw_moment_error),
        "gap": float(result.gap),
        "projection_rank": int(result.projection_rank),
        "iterations": int(result.iterations),
        "gradient_norm": float(result.gradient_norm),
        "objective": float(result.objective),
        "objective_units": result.objective_units,
        "initial_teacher": None if initial_teacher is None else str(initial_teacher),
        "fitted_potential_coefficients": np.asarray(result.potential_coefficients).tolist(),
        "fitted_potential_constants": np.asarray(result.constant).tolist(),
        "failure_is_reference_or_representability": not bool(result.valid),
        "optimizer_convergence_is_not_teacher_validity": True,
        "no_density_rescaling_or_occupation_clipping": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-npz", required=True)
    parser.add_argument("--sample-meta", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-memory-mb", type=float, default=16000.)
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--guide", choices=("hcore", "pbe"), default="hcore")
    parser.add_argument("--uncontract", action="store_true")
    parser.add_argument("--density-metric", choices=("absolute_l2", "relative_l2"), default="absolute_l2")
    parser.add_argument("--method", choices=("wu_yang", "density_match"), default="wu_yang")
    parser.add_argument("--regularization", type=float)
    parser.add_argument("--initial-teacher")
    args = parser.parse_args()
    result = prepare(Path(args.sample_npz), Path(args.sample_meta),
                     max_memory_mb=args.max_memory_mb, grid_level=args.grid_level,
                     block_size=args.block_size,
                     reference_output=Path(args.output).with_suffix(".reference.npz"), guide=args.guide,
                     uncontract=args.uncontract, density_metric=args.density_metric,
                     inverse_method=args.method, regularization=args.regularization,
                     initial_teacher=None if args.initial_teacher is None else Path(args.initial_teacher))
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"valid": result["valid"], "density_error": result["density_error"],
                      "moment_error": result["moment_error"], "gap": result["gap"]}))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
