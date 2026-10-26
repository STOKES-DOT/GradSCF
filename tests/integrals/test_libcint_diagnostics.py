"""Private Cint call counts are opt-in instrumentation, never timing evidence."""
from dataclasses import replace
import os

import jax
import jax.numpy as jnp
import pytest


def test_diagnostic_build_mode_is_explicit():
    from gradscf.integrals._native.diagnostics import get_counts
    counts = get_counts()
    expected = os.environ.get('GRADSCF_EXPECT_DIAGNOSTICS') == '1'
    assert counts['enabled'] is expected
    assert counts['schema'] == 'libcint-call-counts-v1'


@pytest.mark.skipif(os.environ.get('GRADSCF_EXPECT_DIAGNOSTICS') != '1',
                    reason='requires the separate diagnostic library build')
def test_full_value_counts_real_libcint_evaluations_and_reset():
    from gradscf import integrals
    from gradscf.integrals._native.diagnostics import get_counts, reset_counts
    topology, parameters = integrals.prepare_basis('H 0 0 0; H 0 0 1.4',
        {'H': [[0, [1.2, .7], [.3, .4]]]}, unit='Bohr')
    plan = integrals.make_plan(topology)
    numerical = jax.jit(lambda p: plan.evaluate('eri', p))
    jax.block_until_ready(numerical(parameters))
    reset_counts()
    assert get_counts()['families']['eri4c']['evaluations'] == 0
    jax.block_until_ready(numerical(parameters))
    counts = get_counts()
    # The full value driver is deliberately not the S9 derivative target.
    assert counts['families']['eri4c']['evaluations'] == len(topology.angular_momenta)**4
    assert counts['families']['eri4c']['cache_queries'] > 0
    reset_counts()
    assert get_counts()['families']['eri4c']['evaluations'] == 0
