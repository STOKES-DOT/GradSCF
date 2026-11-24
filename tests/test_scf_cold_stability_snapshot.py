"""Unrestricted stability reuses the validation of its cold reference."""

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from gradscf import gto, scf
from gradscf.scf import builders, diagnostics, reference


@pytest.fixture(params=["hf", "pbe"])
def cold_source(request, monkeypatch):
    mol = gto.M(atom="H 0 0 0", basis="sto-3g", spin=1)
    if request.param == "hf":
        return scf.UHF(mol).run()
    # Isolate the deferred DFT reference path from the optional XC backend.
    result = scf.UHF(mol).run().scf_result
    monkeypatch.setattr(builders, "run_uks_from_integrals", lambda **kwargs: result)
    return scf.UKS(mol, xc="pbe").run()


def test_cold_unrestricted_stability_checks_source_once(cold_source, monkeypatch):
    source = cold_source
    assert vars(source).get("reference") is None
    calls = Counter()
    names = ("_spec", "_config", "_scf_signature", "_orbital_signature")

    def counted(name, function):
        def observe(*args, **kwargs):
            calls[name] += 1
            return function(*args, **kwargs)
        return observe

    for name in names:
        monkeypatch.setattr(source, name, counted(name, getattr(source, name)))
    fingerprint = reference._array_signature
    orbital_arrays = source.mo_coeff, source.mo_occ, source.mo_energy

    def observed_fingerprint(value):
        if any(value is array for array in orbital_arrays):
            calls["orbital_array"] += 1
        return fingerprint(value)

    monkeypatch.setattr(reference, "_array_signature", observed_fingerprint)
    monkeypatch.setattr(diagnostics, "_array_signature", observed_fingerprint)

    state = diagnostics._unrestricted_scf_state(source)
    assert state is source.reference
    assert state.mo_coeff is source.mo_coeff and state.mo_occ is source.mo_occ
    assert calls == Counter(dict.fromkeys(names, 1), orbital_array=3)


def test_cold_unrestricted_stability_rejects_stale_energy(cold_source):
    source = cold_source
    source.e_tot += .01
    with pytest.raises(RuntimeError, match="energy changed"):
        diagnostics._unrestricted_scf_state(source)
    assert vars(source).get("reference") is None


def test_cold_unrestricted_stability_checks_returned_reference_orbitals(cold_source, monkeypatch):
    source = cold_source

    def corrupt(state):
        coeff = np.array(state.mo_coeff, copy=True)
        coeff.flat[0] += .01
        return replace(state, mo_coeff=coeff)

    monkeypatch.setattr(source, "_put_on_requested_device", corrupt)
    with pytest.raises(RuntimeError, match="orbital arrays changed"):
        diagnostics._unrestricted_scf_state(source)
    assert vars(source).get("reference") is None
