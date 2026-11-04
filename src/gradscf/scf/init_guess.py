"""Initial densities; optional ``atom`` sums native isolated-UHF atom densities.

The atom guess initially supports H--Ar with STO-3G, 3-21G, 6-31G*, cc-pVDZ,
def2-SVP and def2-TZVP named all-electron bases (including the 6-31Gs alias).
It is an unspherical UHF superposition, not PySCF's spherically averaged SAD.
"""
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from gradscf._memory import DEFAULT_MAX_MEMORY_MB, workspace_bytes

_ATOMIC_SPINS = dict(zip("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar".split(),
                        (1, 0, 1, 0, 1, 2, 3, 2, 1, 0, 1, 0, 1, 2, 3, 2, 1, 0)))
_ATOMIC_BASES = frozenset({"sto3g", "321g", "631g*", "631gs", "ccpvdz", "def2svp", "def2tzvp"})

@dataclass(frozen=True)
class RestrictedInitGuess:
    density: Array | None = None

@dataclass(frozen=True)
class UnrestrictedInitGuess:
    density_alpha: Array | None = None
    density_beta: Array | None = None


def _native_fingerprint():
    from gradscf.integrals._native import register_integrals
    return sha256(Path(register_integrals()._name).read_bytes()).hexdigest()


@lru_cache(maxsize=128)
def _atomic_density(symbol, shells, native_fingerprint):
    """Cache only immutable atom constants, keyed by raw basis and native SHA256."""
    from gradscf.data.molecule import parse_molecule_spec
    from gradscf.integrals.basis import prepare_basis
    from gradscf.integrals.plan import make_plan
    from gradscf.scf.uhf import UHFConfig, run_uhf_from_integrals

    # These fixed atom coefficients are translation independent. Evaluate them
    # concretely even on a cold cache inside an outer molecular geometry trace.
    with jax.ensure_compile_time_eval(), jax.default_device(jax.devices("cpu")[0]):
        spec = parse_molecule_spec([(symbol, (0., 0., 0.))], unit="Bohr", spin=_ATOMIC_SPINS[symbol])
        top, parameters = prepare_basis(spec, shells, cart=True)
        plan = make_plan(top)
        s = plan.evaluate("overlap", parameters)
        h = plan.evaluate("kinetic", parameters) + plan.evaluate("nuclear", parameters)
        z, spin = top.nuclear_charges[0], _ATOMIC_SPINS[symbol]
        result = run_uhf_from_integrals(overlap=s, hcore=h, eri=plan.evaluate("eri", parameters, aosym="s4"),
            nalpha=(z + spin) // 2, nbeta=(z - spin) // 2, nuclear_repulsion=0.,
            config=UHFConfig(max_cycle=200, conv_tol=1e-9, conv_tol_grad=1e-6, conv_tol_density=1e-8))
        if not result.converged:
            raise RuntimeError(f"Native atomic UHF SCF did not converge for {symbol}; supply another initial guess")
        if not np.isfinite(result.total_energy):
            raise RuntimeError(f"Native atomic UHF SCF returned a nonfinite energy for {symbol}")
        density = result.density_matrix_alpha + result.density_matrix_beta
        if any(isinstance(value, jax.core.Tracer) for value in jax.tree_util.tree_leaves(density)):
            raise RuntimeError("Native atomic SCF produced a tracer; atom constants cannot be cached")
        dm = np.array(density, dtype=np.float64, copy=True)
        count = np.trace(dm @ np.asarray(s))
        if dm.shape != (top.nao, top.nao) or not np.isfinite(dm).all() or not np.isfinite(count) or count <= 0:
            raise RuntimeError(f"Native atomic UHF SCF returned an invalid density for {symbol}")
        dm *= z / count
        dm.setflags(write=False)
        return dm


def _atomic_superposition(spec, basis, dtype, max_memory, resident_bytes):
    from gradscf.data.molecule import atomic_number
    from gradscf.integrals.basis.data import load_basis_from_snapshot
    from gradscf.scf.core import _unrestricted_spin_electron_counts

    if spec is None or basis is None:
        raise ValueError("The atom guess requires a molecule specification and a fixed named Gaussian basis")
    names = [basis[symbol] if isinstance(basis, dict) else basis for symbol in spec.symbols]
    if not all(isinstance(name, str) for name in names):
        raise ValueError("The atom guess requires a fixed named Gaussian basis; raw/trainable bases are unsupported")
    if any(name.lower().replace("-", "").replace("_", "").replace(" ", "") not in _ATOMIC_BASES for name in names):
        raise ValueError("The atom guess supports only validated all-electron basis families "
                         "STO-3G, 3-21G, 6-31G*, cc-pVDZ, def2-SVP and def2-TZVP; ECP and other families are unsupported")
    if any(symbol not in _ATOMIC_SPINS for symbol in spec.symbols):
        raise ValueError("Native atom guesses currently support only all-electron H--Ar")
    charges = np.asarray(spec.charges)
    if not np.array_equal(charges, [atomic_number(symbol) for symbol in spec.symbols]):
        raise ValueError("The atom guess requires all-electron atoms; ghost/ECP/modified nuclear charges are unsupported")
    total = int(charges.sum()) - int(spec.charge)
    _unrestricted_spin_electron_counts(total, int(spec.spin))
    atom_shells = []
    for symbol, name in zip(spec.symbols, names):
        raw = load_basis_from_snapshot(name, symbol)
        try:
            shells = tuple((int(block[0]), *(tuple(float(v) for v in row) for row in block[1:])) for block in raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("The atom guess requires a named nonrelativistic Gaussian basis") from exc
        atom_shells.append(shells)
    dimensions = [sum((l + 1) * (l + 2) // 2 * (len(rows[0]) - 1) for l, *rows in shells) for shells in atom_shells]
    nao = sum(dimensions)
    available = workspace_bytes(max_memory, resident_bytes=resident_bytes + 4 * 8 * nao**2, device="cpu")
    blocks, fingerprint = [], _native_fingerprint()
    for symbol, shells, n in zip(spec.symbols, atom_shells, dimensions):
        pair = n * (n + 1) // 2
        # Packed ERIs, conservative contraction tiles, DIIS and SCF matrices.
        required = 8 * (2 * pair**2 + 8 * n**3 + 128 * n**2)
        if required > available:
            raise ValueError(f"Native atomic SCF for {symbol} needs at least {required / 1e6:.1f} MB; increase max_memory")
        blocks.append(_atomic_density(symbol, shells, fingerprint))
    dm = np.zeros((nao, nao), dtype=np.float64)
    start = 0
    # prepare_basis appends each atom's shells in this same static AO order.
    for block in blocks:
        end = start + len(block)
        dm[start:end, start:end] = block
        start = end
    return jnp.asarray(dm * (total / int(charges.sum())), dtype=dtype)


def _hcore_requested(value):
    if value is None:
        return True
    if isinstance(value, str):
        if value.lower() in {"hcore", "1e"}:
            return True
        raise ValueError(f"Unsupported initial guess {value!r}; use 'hcore'/'1e'/'atom' or supply a density matrix")
    return False


def _density(value, dtype):
    dm = jnp.asarray(value, dtype=dtype)
    if dm.ndim != 2 or dm.shape[0] != dm.shape[1]:
        raise ValueError("Initial density must be a square matrix")
    return .5*(dm+dm.T.conj())


def restricted_initial_guess(*, init_guess: Any, dtype: Any, spec=None, basis=None,
                             max_memory=DEFAULT_MAX_MEMORY_MB, resident_bytes=0):
    if isinstance(init_guess, str) and init_guess.lower() == "atom":
        return RestrictedInitGuess(_atomic_superposition(spec, basis, dtype, max_memory, resident_bytes))
    if _hcore_requested(init_guess):
        return RestrictedInitGuess()
    return RestrictedInitGuess(_density(init_guess, dtype))


def unrestricted_initial_guess(*, init_guess: Any, dtype: Any, spec=None, basis=None,
                               max_memory=DEFAULT_MAX_MEMORY_MB, resident_bytes=0):
    if isinstance(init_guess, str) and init_guess.lower() == "atom":
        from gradscf.scf.core import _unrestricted_spin_electron_counts
        density = _atomic_superposition(spec, basis, dtype, max_memory, resident_bytes)
        total = int(np.asarray(spec.charges).sum()) - int(spec.charge)
        alpha, beta = _unrestricted_spin_electron_counts(total, int(spec.spin))
        return UnrestrictedInitGuess(density * (alpha / total), density * (beta / total))
    if _hcore_requested(init_guess):
        return UnrestrictedInitGuess()
    dm = jnp.asarray(init_guess, dtype=dtype)
    if dm.ndim != 3 or dm.shape[0] != 2:
        raise ValueError("Unrestricted initial density must have shape (2, nao, nao)")
    return UnrestrictedInitGuess(_density(dm[0], dtype), _density(dm[1], dtype))

def orbital_rotation_guesses(mo_coeff, *, amplitudes=(0., .1), seed=20260913):
    """Return reproducible, metric-preserving orbital guesses for explicit restarts.

    This host-side helper rotates a complete S-orthonormal orbital matrix;
    refilling the requested occupations preserves electron counts/idempotency.
    It does not select a root or assert that a converged state is the ground
    state. The caller must inspect convergence and compare candidate energies.
    """
    coeff = np.asarray(mo_coeff)
    scales = tuple(float(a) for a in amplitudes)
    if coeff.ndim != 2 or coeff.shape[0] != coeff.shape[1] or not np.all(np.isfinite(coeff)):
        raise ValueError("Expected a finite, complete square orbital matrix")
    if not scales or any(not np.isfinite(a) or a < 0 for a in scales):
        raise ValueError("Rotation amplitudes must be finite and nonnegative")
    random = np.random.default_rng(seed).normal(size=coeff.shape)
    generator = random-random.T
    # exp(A) from the Hermitian eigendecomposition of i*A, avoiding another
    # runtime dependency and retaining an exactly orthogonal rotation in theory.
    eigenvalues, vectors = np.linalg.eigh(1j*generator)
    guesses = []
    for scale in scales:
        rotation = ((vectors*np.exp(-1j*scale*eigenvalues))@vectors.conj().T).real
        guesses.append(coeff.copy() if scale == 0 else coeff@rotation)
    return tuple(guesses)


__all__ = ["RestrictedInitGuess", "UnrestrictedInitGuess", "restricted_initial_guess", "unrestricted_initial_guess", "orbital_rotation_guesses"]
