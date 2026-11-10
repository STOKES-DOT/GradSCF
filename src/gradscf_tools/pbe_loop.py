"""Small fixed-reference harness for the PBE/CCSD-density loop.

The module stores only one reference and its fixed-density affine feature
tables at a time. Integral/grid construction remains in GradSCF; no SCF or
inverse solver is reimplemented here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gradscf import dft, integrals
from gradscf.data.molecule import parse_molecule_spec
from gradscf.dft import iks
from gradscf.dft.derivatives import xc_energy_and_potential_from_density
from gradscf.solvers.nonlinear import minimize_eager
from .cafes_qm9 import deterministic_split


REFERENCE_KEYS = frozenset({
    "coefficients", "occupations", "energies", "density_target", "density_ks",
    "overlap", "fock", "hcore", "grid_coords", "grid_weights", "energy_total",
    "atomic_numbers", "positions_angstrom",
})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class TeacherArtifact:
    target: object
    reference: object
    density_ks: object
    overlap: object
    fock: object
    hcore: object
    grid_coords: object
    grid_weights: object
    atomic_numbers: object
    positions_angstrom: object
    reference_artifact: Path
    provenance: dict


def load_teacher_reference(teacher_json: str | Path,
                           reference_npz: str | Path | None = None) -> TeacherArtifact:
    """Load a valid persisted teacher and refuse partial/failed artifacts."""
    teacher_path = Path(teacher_json)
    payload = json.loads(teacher_path.read_text(encoding="utf-8"))
    if payload.get("valid") is not True:
        raise ValueError("teacher must have valid=true; failed inversion is not a reference")
    artifact = payload.get("reference_artifact")
    if not isinstance(artifact, dict) or not artifact.get("path"):
        raise ValueError("valid teacher is missing reference_artifact")
    ref_path = Path(reference_npz) if reference_npz is not None else Path(artifact["path"])
    if not ref_path.is_absolute():
        ref_path = (teacher_path.parent / ref_path).resolve()
    if not ref_path.is_file():
        raise FileNotFoundError(f"teacher reference NPZ is missing: {ref_path}")
    expected_hash = artifact.get("sha256")
    actual_hash = _sha256(ref_path)
    if expected_hash is not None and actual_hash != expected_hash:
        raise ValueError("teacher reference NPZ SHA256 does not match teacher JSON")
    with np.load(ref_path, allow_pickle=False) as stored:
        missing = REFERENCE_KEYS.difference(stored.files)
        if missing:
            raise ValueError(f"teacher reference NPZ missing keys: {sorted(missing)}")
        arrays = {key: np.asarray(stored[key]) for key in REFERENCE_KEYS}

    overlap = np.asarray(arrays["overlap"], dtype=float)
    target_density = np.asarray(arrays["density_target"], dtype=float)
    coefficients = np.asarray(arrays["coefficients"], dtype=float)
    occupations = np.asarray(arrays["occupations"], dtype=float)
    energies = np.asarray(arrays["energies"], dtype=float)
    density_ks = np.asarray(arrays["density_ks"], dtype=float)
    fock = np.asarray(arrays["fock"], dtype=float)
    hcore = np.asarray(arrays["hcore"], dtype=float)
    n = overlap.shape[0]
    if overlap.shape != (n, n) or target_density.shape != (n, n):
        raise ValueError("teacher target and overlap must be matching square AO matrices")
    if coefficients.shape != (n, n) or occupations.shape != (n,) or energies.shape != (n,):
        raise ValueError("teacher orbitals must contain a complete square AO/MO space")
    for name, value in (("density_ks", density_ks), ("fock", fock), ("hcore", hcore)):
        if value.shape != (n, n):
            raise ValueError(f"teacher {name} must match the AO dimension")
        if not np.allclose(value, value.T, atol=1e-10, rtol=0.):
            raise ValueError(f"teacher {name} must be symmetric")
    if not np.all(np.isfinite(np.concatenate([value.ravel() for value in
                                               (overlap, target_density, coefficients,
                                                occupations, energies, density_ks, fock, hcore)]))):
        raise ValueError("teacher reference contains non-finite numerical fields")
    if not np.all(np.isclose(occupations, 0.) | np.isclose(occupations, 2.)):
        raise ValueError("stationarity reference requires integer RKS occupations")
    nelec = int(round(float(np.sum(occupations))))
    if not np.isclose(np.trace(target_density @ overlap), nelec, atol=1e-7, rtol=0.):
        raise ValueError("teacher target density does not preserve its electron count")
    energy = float(np.asarray(arrays["energy_total"]))
    if not np.isfinite(energy):
        raise ValueError("teacher total energy is not finite")
    grid_coords = np.asarray(arrays["grid_coords"], dtype=float)
    grid_weights = np.asarray(arrays["grid_weights"], dtype=float)
    if grid_coords.ndim != 2 or grid_coords.shape[1] != 3 or grid_weights.shape != (grid_coords.shape[0],):
        raise ValueError("teacher grid coordinates and weights have incompatible shapes")
    if (not np.isfinite(grid_coords).all() or not np.isfinite(grid_weights).all()
            or np.any(grid_weights < 0) or not np.sum(grid_weights) > 0):
        raise ValueError('Teacher grid coordinates and positive quadrature must be finite')
    atomic_numbers = np.asarray(arrays["atomic_numbers"])
    positions = np.asarray(arrays["positions_angstrom"], dtype=float)
    if atomic_numbers.ndim != 1 or positions.shape != (atomic_numbers.size, 3):
        raise ValueError("teacher geometry fields have incompatible shapes")
    target = iks.Target(target_density, overlap, nelec, energy=energy)
    reference = iks.OrbitalReference(
        target, coefficients, occupations, energies,
        method=str(payload.get("reference_method", "wu_yang")), reference_valid=True,
    )
    if not np.allclose(np.asarray(reference.density), density_ks, atol=1e-8, rtol=1e-8):
        raise ValueError("persisted density_ks does not match persisted orbitals")
    if not np.allclose(np.asarray(reference.fock), fock, atol=1e-9, rtol=1e-12):
        raise ValueError('Persisted Fock disagrees with the complete supplied spectrum')
    if "target_energy" in payload and not np.isclose(payload["target_energy"], energy, atol=1e-10, rtol=0.):
        raise ValueError("teacher JSON and reference NPZ total energies disagree")
    required_provenance = ("source_basis", "inversion_basis", "guide_settings",
                           "smooth_pbs", "grid_level", "quality_settings")
    missing_provenance = [key for key in required_provenance if key not in payload]
    if missing_provenance:
        raise ValueError(f"teacher JSON missing provenance: {missing_provenance}")
    provenance = dict(payload)
    provenance["teacher_json_sha256"] = _sha256(teacher_path)
    provenance["reference_npz_sha256"] = actual_hash
    provenance["reference_artifact_path"] = str(ref_path)
    return TeacherArtifact(target, reference, density_ks, overlap, fock, hcore,
                           grid_coords, grid_weights, atomic_numbers, positions,
                           ref_path, provenance)


def make_pbe_energy_function(molecule, *, nuclear_repulsion):
    """Bind the shared PBE total-energy callback for a fixed density."""
    return iks.energy_function(molecule, "pbe", layout="rks",
                               nuclear_repulsion=nuclear_repulsion)


def build_single_pbe_input(teacher: TeacherArtifact, *, grid_level=None,
                           max_memory_mb: float = 6000., with_coulomb: bool = True):
    """Return (fixed-density input, nuclear repulsion) without another SCF.

    Packed AO integrals and a streamed grid are prepared once per molecule.
    Trial coefficient fits retain only small scalar/AO derivative tables.
    """
    from gradscf._memory import tree_storage_bytes, workspace_bytes
    from gradscf.integrals.grids.blocks import AOGrid
    from gradscf.tools.features import MoleculeLikeState, MoleculeGridView

    declared_level = teacher.provenance['grid_level']
    if grid_level is not None and grid_level != declared_level:
        raise ValueError('Reuse the teacher quadrature; do not change its grid_level')
    n = teacher.overlap.shape[0]
    pairs = n*(n+1)//2
    workspace_bytes(max_memory_mb, resident_bytes=tree_storage_bytes(
        teacher.overlap, teacher.fock, teacher.hcore, teacher.reference,
        teacher.grid_coords, teacher.grid_weights)+(3*pairs*pairs*8 if with_coulomb else 0))
    symbols = {1:'H', 2:'He', 3:'Li', 4:'Be', 5:'B', 6:'C', 7:'N', 8:'O', 9:'F', 10:'Ne'}
    atom = '; '.join(f'{symbols[int(z)]} {x:.16g} {y:.16g} {v:.16g}'
        for z,(x,y,v) in zip(teacher.atomic_numbers, teacher.positions_angstrom))
    spec = parse_molecule_spec(atom, unit='Angstrom')
    orbital_basis = teacher.provenance.get('orbital_basis_definitions', 'def2-svp')
    basis = integrals.basis_from_molecule_spec(spec, basis=orbital_basis)
    if basis.nao != n or not np.allclose(integrals.overlap_matrix(basis), teacher.overlap,
                                         atol=1e-9, rtol=1e-12):
        raise ValueError('Cached teacher overlap disagrees with the declared orbital basis')
    rep = integrals.eri_pair_matrix_packed(basis) if with_coulomb else None
    grid = AOGrid.from_basis(basis, teacher.grid_coords, teacher.grid_weights, block_size=1024)
    state = MoleculeLikeState(
        ao=jnp.zeros((0,n)), ao_deriv1=jnp.zeros((4,0,n)),
        grid=MoleculeGridView(jnp.asarray(teacher.grid_weights), jnp.asarray(teacher.grid_coords)),
        rdm1=teacher.reference.density, mo_coeff=teacher.reference.coefficients,
        mo_occ=teacher.reference.occupations, mo_energy=teacher.reference.energies,
        rep_tensor=rep, h1e=jnp.asarray(teacher.hcore), overlap_matrix=jnp.asarray(teacher.overlap),
        atom_coords=spec.coords_bohr, atom_charges=jnp.asarray(spec.charges), grid_ao=grid)
    return state, float(spec.nuclear_repulsion)


@dataclass(frozen=True)
class AffineTerm:
    name: str
    functional: Callable
    params: object = None


class AffineEvaluation(NamedTuple):
    total_energy: object
    energy_error: object
    energy_mse: object
    energy_mae: object
    rotation_norm2: object
    fock: object
    valid: object


@dataclass(frozen=True)
class AffineCache:
    teacher: TeacherArtifact
    term_names: tuple[str, ...]
    base_energy: object
    base_fock: object
    term_energies: object
    term_focks: object
    energy_scale: float = 1.0
    stationarity_weight: float = 0.01
    gap_floor: float = 0.05

    def energy(self, theta):
        """Cached scalar at this fixed reference; not a density-energy callback."""
        theta = jnp.asarray(theta)
        return self.base_energy + jnp.dot(theta, self.term_energies)

    def fock(self, theta):
        return self.base_fock + jnp.einsum("k,kij->ij", jnp.asarray(theta), self.term_focks)

    def evaluate(self, theta) -> AffineEvaluation:
        theta = jnp.asarray(theta)
        if theta.shape != (len(self.term_names),):
            raise ValueError("affine parameters must match the declared term count")
        fock = self.fock(theta)
        total = self.energy(theta)
        error = total - self.teacher.target.energy
        _, rotation, response_valid = iks.response.orbital_response(
            self.teacher.reference, fock, self.gap_floor, with_density=False,
        )
        finite = (jnp.isfinite(total) & jnp.isfinite(error) & jnp.isfinite(rotation)
                  & jnp.all(jnp.isfinite(theta)) & jnp.all(jnp.isfinite(fock)))
        valid = self.teacher.reference.valid & response_valid & finite
        return AffineEvaluation(total, error, (error / self.energy_scale) ** 2,
                                jnp.abs(error / self.energy_scale), rotation,
                                fock, valid)

    def loss(self, theta):
        score = self.evaluate(theta)
        value = score.energy_mse + score.energy_mae + self.stationarity_weight * score.rotation_norm2
        return jnp.where(score.valid, value, jnp.nan)


def _derivative(energy_fn, params, density):
    return xc_energy_and_potential_from_density(
        params, molecule=None, density=density,
        xc_energy_fn=lambda p, _molecule, d: energy_fn(p, d),
        check_finite=False,
    )


def build_affine_cache(teacher: TeacherArtifact, *, baseline_energy_fn: Callable,
                       terms: Iterable[AffineTerm], energy_scale: float = 1.0,
                       stationarity_weight: float = 0.01, gap_floor: float = 0.05,
                       baseline_cache: AffineCache | None = None) -> AffineCache:
    """Cache exact fixed-density energy/AO-derivative tables for an affine family."""
    if (not np.isfinite([energy_scale, stationarity_weight, gap_floor]).all()
            or energy_scale <= 0 or stationarity_weight < 0 or gap_floor <= 0):
        raise ValueError("energy scale must be positive; weights and gap floor must be nonnegative/positive")
    terms = tuple(terms)
    if len({term.name for term in terms}) != len(terms):
        raise ValueError("affine term names must be unique")
    if baseline_cache is None:
        base = _derivative(baseline_energy_fn, None, teacher.reference.density)
        base_energy, base_fock = base.xc_energy, base.vxc_matrix
    else:
        if baseline_cache.teacher.provenance['teacher_json_sha256'] != teacher.provenance['teacher_json_sha256']:
            raise ValueError('Reuse a baseline only for the identical immutable teacher')
        base_energy, base_fock = baseline_cache.base_energy, baseline_cache.base_fock
    term_energies, term_focks = [], []
    for term in terms:
        if not callable(term.functional):
            raise TypeError("each affine term must supply an external scalar Functional callback")
        value = _derivative(term.functional, term.params, teacher.reference.density)
        term_energies.append(value.xc_energy)
        term_focks.append(value.vxc_matrix)
    return AffineCache(
        teacher, tuple(term.name for term in terms), base_energy, base_fock,
        jnp.stack(term_energies) if term_energies else jnp.zeros((0,), dtype=jnp.asarray(base_energy).dtype),
        jnp.stack(term_focks) if term_focks else jnp.zeros((0,) + teacher.overlap.shape, dtype=jnp.asarray(base_fock).dtype),
        float(energy_scale), float(stationarity_weight), float(gap_floor),
    )


def build_molecular_affine_cache(teacher: TeacherArtifact, molecule, terms: Iterable[AffineTerm], *,
                                 nuclear_repulsion, energy_scale: float = 1.0,
                                 stationarity_weight: float = 0.01, gap_floor: float = 0.05,
                                 baseline_cache: AffineCache | None = None) -> AffineCache:
    """Bind PBE plus external ``dft.Functional`` terms for one molecule."""
    baseline = (make_pbe_energy_function(molecule, nuclear_repulsion=nuclear_repulsion)
                if baseline_cache is None else None)
    bound_terms = []
    for term in terms:
        if not isinstance(term.functional, dft.Functional):
            raise TypeError("molecular affine terms must be GradSCF Functional objects")
        callback = iks.energy_function(molecule, term.functional, layout="rks",
                                       component="xc", nuclear_repulsion=0.)
        bound_terms.append(AffineTerm(term.name, callback, term.params))
    return build_affine_cache(teacher, baseline_energy_fn=baseline, terms=bound_terms,
                              energy_scale=energy_scale,
                              stationarity_weight=stationarity_weight, gap_floor=gap_floor,
                              baseline_cache=baseline_cache)


def summarize_affine(caches: Iterable[AffineCache], theta) -> dict:
    scores = [cache.evaluate(theta) for cache in caches]
    if not scores:
        raise ValueError("at least one cache is required")
    errors = jnp.stack([score.energy_error for score in scores])
    rotations = jnp.stack([score.rotation_norm2 for score in scores])
    valid = bool(jnp.all(jnp.asarray([score.valid for score in scores])))
    return {
        "valid": valid,
        "energy_mse_hartree2": float(jnp.mean(errors ** 2)),
        "energy_mae_hartree": float(jnp.mean(jnp.abs(errors))),
        "energy_wrmsd_hartree": float(jnp.sqrt(jnp.mean(errors ** 2))),
        "rotation_norm2_mean": float(jnp.mean(rotations)),
        "n_molecules": len(scores),
    }


def fit_affine(train_caches: Iterable[AffineCache], initial, *, bounds=None,
               maxiter: int = 300, gradient_tolerance: float = 1e-8):
    """Fit only on supplied train caches using the shared eager minimizer."""
    caches = tuple(train_caches)
    if not caches:
        raise ValueError("fit_affine needs at least one training cache")
    initial = np.asarray(initial, dtype=float)
    if initial.ndim != 1 or initial.size != len(caches[0].term_names):
        raise ValueError("initial parameters do not match the affine term count")
    if any(cache.term_names != caches[0].term_names for cache in caches[1:]):
        raise ValueError("all training caches must share affine term names")
    if bounds is None:
        decode = lambda z: jnp.asarray(z)
        start = initial
    else:
        limits = np.asarray(bounds, dtype=float)
        if limits.shape != (initial.size, 2) or not np.all(np.isfinite(limits)):
            raise ValueError("bounds must be finite with shape (nterms,2)")
        if not np.all(limits[:, 0] < limits[:, 1]) or np.any(initial <= limits[:, 0]) or np.any(initial >= limits[:, 1]):
            raise ValueError("bounded initial parameters must be strictly inside each bound")
        mid = (limits[:, 0] + limits[:, 1]) / 2
        half = (limits[:, 1] - limits[:, 0]) / 2
        decode = lambda z: jnp.asarray(mid) + jnp.asarray(half) * jnp.tanh(z)
        start = np.arctanh((initial - mid) / half)
    def objective(z):
        theta = decode(z)
        return jnp.mean(jnp.stack([cache.loss(theta) for cache in caches]))
    result = minimize_eager(jax.jit(jax.value_and_grad(objective)), start,
                            maxiter=maxiter, gradient_tolerance=gradient_tolerance)
    return {"initial_parameters": initial.tolist(),
            "fitted_parameters": np.asarray(decode(result.x)).tolist(),
            "optimizer": result}


__all__ = [
    "AffineCache", "AffineEvaluation", "AffineTerm", "TeacherArtifact",
    "build_affine_cache", "build_molecular_affine_cache", "build_single_pbe_input",
    "fit_affine", "load_teacher_reference", "make_pbe_energy_function", "summarize_affine",
]


def save_affine_cache(cache, path, *, teacher_json, metadata=None):
    """Persist small E/F tables, retaining the teacher and source identity."""
    path, teacher_json = Path(path), Path(teacher_json).resolve()
    teacher_hash = _sha256(teacher_json)
    if teacher_hash != cache.teacher.provenance['teacher_json_sha256']:
        raise ValueError('The teacher changed after cache construction')
    meta = dict(version=1, teacher_json=str(teacher_json), teacher_sha256=teacher_hash,
                metadata={} if metadata is None else dict(metadata))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        np.savez_compressed(handle, base_energy=np.asarray(cache.base_energy),
            base_fock=np.asarray(cache.base_fock), term_energies=np.asarray(cache.term_energies),
            term_focks=np.asarray(cache.term_focks), term_names=np.asarray(cache.term_names),
            energy_scale=cache.energy_scale, stationarity_weight=cache.stationarity_weight,
            gap_floor=cache.gap_floor, metadata_json=np.asarray(json.dumps(meta)))
    return dict(path=str(path), sha256=_sha256(path))


def load_affine_cache(path, *, teacher_json=None, reference_npz=None, expected_sha256=None):
    """Load a verified cache without integral, grid or SCF reconstruction."""
    path = Path(path)
    if expected_sha256 is not None and _sha256(path) != expected_sha256:
        raise ValueError('Affine-cache SHA256 mismatch')
    with np.load(path, allow_pickle=False) as stored:
        meta = json.loads(str(stored['metadata_json'].item()))
        if meta['version'] != 1:
            raise ValueError('Unsupported affine-cache version')
        source = Path(teacher_json) if teacher_json is not None else Path(meta['teacher_json'])
        if _sha256(source) != meta['teacher_sha256']:
            raise ValueError('Cached teacher identity changed')
        teacher = load_teacher_reference(source, reference_npz=reference_npz)
        names = tuple(str(v) for v in stored['term_names'])
        arrays = {key:np.asarray(stored[key]) for key in
                  ('base_energy','base_fock','term_energies','term_focks')}
        settings = [float(stored[key]) for key in ('energy_scale','stationarity_weight','gap_floor')]
    n, k = teacher.overlap.shape[0], len(names)
    if (arrays['base_energy'].shape != () or arrays['base_fock'].shape != (n,n)
            or arrays['term_energies'].shape != (k,) or arrays['term_focks'].shape != (k,n,n)):
        raise ValueError('Affine-cache table shapes disagree with its teacher and term names')
    if (any(np.iscomplexobj(v) or not np.isfinite(v).all() for v in arrays.values())
            or not np.isfinite(settings).all() or settings[0] <= 0 or settings[1] < 0 or settings[2] <= 0):
        raise ValueError('Affine-cache values and loss settings must be finite and physical')
    return AffineCache(teacher, names, *(jnp.asarray(arrays[key]) for key in
        ('base_energy','base_fock','term_energies','term_focks')), *settings)


__all__ += ['save_affine_cache', 'load_affine_cache']
