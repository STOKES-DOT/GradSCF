"""Public response/tool imports and canonical numerical defaults."""

import importlib
import inspect

import pytest


@pytest.mark.parametrize("name", ["gradscf.tddft", "gradscf.tdscf", "gradscf_tools"])
def test_public_response_and_tool_namespaces_import(name):
    assert importlib.import_module(name) is not None


def test_public_tddft_defaults_match_the_canonical_constants():
    from gradscf.tddft import defaults
    from gradscf.tddft.casida import RestrictedCasidaTDDFT
    from gradscf.tddft.unrestricted import UnrestrictedTDA
    from gradscf.tdscf import TDA

    for constructor in (RestrictedCasidaTDDFT, UnrestrictedTDA, TDA):
        parameters = inspect.signature(constructor).parameters
        assert parameters["davidson_tol"].default == defaults.TD_DAVIDSON_TOL
        assert parameters["excitation_threshold"].default == defaults.TD_POSITIVE_EIG_THRESHOLD
    assert inspect.signature(TDA).parameters["davidson_max_iter"].default == defaults.TD_DAVIDSON_MAX_CYCLE
