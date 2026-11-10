"""Native isolated-UHF superposition guesses for fixed named Gaussian bases."""
from dataclasses import replace
import ast
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.data.molecule import parse_molecule_spec
from gradscf.scf.inputs.assembly import build_rks_integral_inputs, build_uks_integral_inputs


@pytest.mark.parametrize("method", ["RHF", "UHF", "GHF", "RKS", "UKS", "GKS"])
def test_public_six_methods_atom_guess_agrees_with_hcore(method):
    from gradscf import gto, scf

    mol = gto.M(atom="H 0 0 0; H 0 0 .74", basis="sto-3g")
    kwargs = dict(init_guess="atom", grid_block_size=-1, max_cycle=100)
    if method.endswith("KS"):
        kwargs["xc"] = "pbe"
    atomic = getattr(scf, method)(mol, **kwargs).run()
    core = getattr(scf, method)(mol, **{**kwargs, "init_guess": "hcore"}).run()
    assert atomic.converged and core.converged
    np.testing.assert_allclose(atomic.e_tot, core.e_tot, atol=2e-8, rtol=0)
    density = np.asarray(atomic.make_rdm1())
    assert np.isfinite(density).all()
    assert np.isfinite(np.asarray(atomic.mo_coeff)).all()


@pytest.mark.parametrize("distance", [.8, 1.15])
@pytest.mark.parametrize("unrestricted", [False, True])
def test_charged_atom_guess_has_metric_charge_and_spin_at_each_geometry(distance, unrestricted):
    atom = [("O", (0., 0., 0.)), ("H", (0., 0., distance)), ("H", (distance, 0., 0.))]
    builder = build_uks_integral_inputs if unrestricted else build_rks_integral_inputs
    inputs = builder(atom=atom, basis={"O": "sto-3g", "H": "sto-3g"},
        charge=1 if unrestricted else 2, spin=1 if unrestricted else 0,
        xc_spec="hf", init_guess="atom", grids_level=0)
    s = np.asarray(inputs.overlap)
    if unrestricted:
        a, b = np.asarray(inputs.init_density_alpha), np.asarray(inputs.init_density_beta)
        np.testing.assert_allclose(np.trace(a @ s), inputs.nalpha, atol=2e-10)
        np.testing.assert_allclose(np.trace(b @ s), inputs.nbeta, atol=2e-10)
        density = a + b
    else:
        density = np.asarray(inputs.init_density)
        np.testing.assert_allclose(np.trace(density @ s), inputs.nelectron, atol=2e-10)
    np.testing.assert_allclose(density, density.T, atol=1e-13)
    # Atom-major AO layout: O occupies five STO-3G AOs, each H one.
    np.testing.assert_array_equal(density[:5, 5:], 0.)
    np.testing.assert_array_equal(density[5:6, 6:], 0.)
    assert np.linalg.eigvalsh(density).min() >= -1e-12


def test_hcore_and_explicit_density_initial_guess_compatibility():
    from gradscf.scf.init_guess import restricted_initial_guess, unrestricted_initial_guess

    for name in (None, "hcore", "1e"):
        assert restricted_initial_guess(init_guess=name, dtype=jnp.float64).density is None
        guess = unrestricted_initial_guess(init_guess=name, dtype=jnp.float64)
        assert guess.density_alpha is None and guess.density_beta is None
    density = np.array([[1., .3], [.1, 0.]])
    expected = (density + density.T) / 2
    np.testing.assert_array_equal(restricted_initial_guess(init_guess=density,
        dtype=jnp.float64).density, expected)
    guess = unrestricted_initial_guess(init_guess=np.stack([density, density / 2]), dtype=jnp.float64)
    np.testing.assert_array_equal(guess.density_alpha, expected)
    np.testing.assert_array_equal(guess.density_beta, expected / 2)


def test_atom_guess_rejects_raw_basis_and_unsupported_elements():
    from gradscf.scf.init_guess import restricted_initial_guess
    from gradscf.integrals.basis.data import load_basis_from_snapshot

    spec = parse_molecule_spec("H 0 0 0; H 0 0 .74")
    raw = {"H": load_basis_from_snapshot("sto-3g", "H")}
    with pytest.raises(ValueError, match="named.*basis"):
        restricted_initial_guess(init_guess="atom", dtype=jnp.float64, spec=spec, basis=raw)
    unsupported = parse_molecule_spec("K 0 0 0; H 0 0 2")
    with pytest.raises(ValueError, match="H.*Ar"):
        restricted_initial_guess(init_guess="atom", dtype=jnp.float64,
                                 spec=unsupported, basis="sto-3g")


@pytest.mark.parametrize("symbol,spin,basis", [("Na", 1, "LANL2DZ"), ("C", 2, "bfd-vdz")])
def test_ecp_basis_families_are_rejected_before_atomic_cache(monkeypatch, symbol, spin, basis):
    from gradscf.scf import init_guess as guesses

    def unexpected_native_call(*args):
        raise AssertionError("Unsupported ECP basis reached the atomic native cache")

    monkeypatch.setattr(guesses, "_native_fingerprint", unexpected_native_call)
    with pytest.raises(ValueError, match="ECP.*unsupported"):
        guesses.unrestricted_initial_guess(init_guess="atom", dtype=jnp.float64,
            spec=parse_molecule_spec([(symbol, (0., 0., 0.))], spin=spin), basis=basis)


def test_atom_memory_preflight_counts_molecular_resident_storage(monkeypatch):
    from gradscf.scf import init_guess as guesses

    guesses._atomic_density.cache_clear()
    monkeypatch.setattr(guesses, "_native_fingerprint", lambda: "unused-budget-test")
    with pytest.raises(ValueError, match="resident|budget|memory"):
        guesses.restricted_initial_guess(init_guess="atom", dtype=jnp.float64,
            spec=parse_molecule_spec("H 0 0 0; H 0 0 .74"), basis="sto-3g",
            max_memory=.1, resident_bytes=100_000)
    assert guesses._atomic_density.cache_info().currsize == 0


def test_cold_atom_cache_inside_geometry_jit_does_not_capture_tracers():
    from gradscf.scf import init_guess as guesses
    from gradscf.scf.rks import RKSConfig, run_rks_from_integrals_traceable

    guesses._atomic_density.cache_clear()
    spec = parse_molecule_spec("H 0 0 0; H 0 0 1.4", unit="Bohr")
    config = RKSConfig(xc_spec="hf", conv_tol=1e-11, conv_tol_grad=1e-9)

    def energy(distance):
        coords = jnp.asarray(spec.coords_bohr).at[1, 2].set(distance)
        inputs = build_rks_integral_inputs(atom=replace(spec, coords_bohr=coords),
            basis="sto-3g", config=config, init_guess="atom", grids_level=0)
        return run_rks_from_integrals_traceable(**inputs.as_rks_kwargs(), config=config).total_energy

    compiled = jax.jit(energy)
    first, second = compiled(1.4), compiled(1.7)
    assert np.isfinite([first, second]).all()
    assert abs(float(first - second)) > 1e-4
    step = 1e-4
    finite_difference = (compiled(1.4 + step) - compiled(1.4 - step)) / (2 * step)
    np.testing.assert_allclose(jax.jit(jax.grad(energy))(1.4), finite_difference,
                               atol=2e-7, rtol=0)
    info = guesses._atomic_density.cache_info()
    assert info.maxsize <= 128 and info.currsize == 1 and info.hits >= 1
    # A fresh eager assembly must also reuse a numerical constant, never a tracer.
    inputs = build_rks_integral_inputs(atom=spec, basis="sto-3g", xc_spec="hf", init_guess="atom")
    assert not isinstance(inputs.init_density, jax.core.Tracer)
    assert np.isfinite(np.asarray(inputs.init_density)).all()


def test_atom_guess_failure_is_explicit_and_does_not_fall_back(monkeypatch):
    from types import SimpleNamespace
    from gradscf.scf import init_guess as guesses
    from gradscf.scf import uhf

    guesses._atomic_density.cache_clear()
    monkeypatch.setattr(uhf, "run_uhf_from_integrals", lambda **kwargs: SimpleNamespace(converged=False))
    with pytest.raises(RuntimeError, match="atomic.*SCF.*converge"):
        guesses.restricted_initial_guess(init_guess="atom", dtype=jnp.float64,
            spec=parse_molecule_spec("H 0 0 0; H 0 0 .74"), basis="sto-3g")
    assert guesses._atomic_density.cache_info().currsize == 0


def test_nonfinite_atomic_energy_is_rejected_before_caching(monkeypatch):
    from types import SimpleNamespace
    from gradscf.scf import init_guess as guesses
    from gradscf.scf import uhf

    guesses._atomic_density.cache_clear()
    monkeypatch.setattr(uhf, "run_uhf_from_integrals",
        lambda **kwargs: SimpleNamespace(converged=True, total_energy=float("nan")))
    with pytest.raises(RuntimeError, match="atomic.*SCF.*nonfinite"):
        guesses.restricted_initial_guess(init_guess="atom", dtype=jnp.float64,
            spec=parse_molecule_spec("H 0 0 0; H 0 0 .74"), basis="sto-3g")
    assert guesses._atomic_density.cache_info().currsize == 0


def test_atomic_cache_tracks_raw_basis_and_native_fingerprint_and_is_immutable(monkeypatch):
    from gradscf.scf import init_guess as guesses
    from gradscf.integrals.basis import data

    guesses._atomic_density.cache_clear()
    spec = parse_molecule_spec("H 0 0 0; H 0 0 .74")
    kwargs = dict(init_guess="atom", dtype=jnp.float64, spec=spec, basis="sto-3g")
    first = np.asarray(guesses.restricted_initial_guess(**kwargs).density).copy()
    loader = data.load_basis_from_snapshot
    raw = loader("sto-3g", "H")
    key = tuple((block[0], *(tuple(row) for row in block[1:])) for block in raw)
    cached = guesses._atomic_density("H", key, guesses._native_fingerprint())
    assert isinstance(cached, np.ndarray) and not cached.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        cached[0, 0] = 0.
    assert guesses._atomic_density.cache_info().misses == 1
    changed = loader("sto-3g", "H")
    changed[0][1][0] *= 1.01
    monkeypatch.setattr(data, "load_basis_from_snapshot", lambda *args: changed)
    guesses.restricted_initial_guess(**kwargs)
    assert guesses._atomic_density.cache_info().misses == 2
    fingerprint = guesses._native_fingerprint()
    monkeypatch.setattr(guesses, "_native_fingerprint", lambda: fingerprint + "-changed")
    guesses.restricted_initial_guess(**kwargs)
    assert guesses._atomic_density.cache_info().misses == 3
    # Copies used for molecular scaling must not mutate a cached atomic block.
    np.testing.assert_array_equal(cached, first[:1, :1])


def test_atom_guess_has_no_pyscf_runtime_import():
    from gradscf.scf import init_guess as guesses

    imports = [node for node in ast.walk(ast.parse(inspect.getsource(guesses)))
               if isinstance(node, (ast.Import, ast.ImportFrom))]
    modules = [node.module if isinstance(node, ast.ImportFrom) else alias.name
               for node in imports for alias in node.names]
    assert not any(module and module.split(".")[0] == "pyscf" for module in modules)
