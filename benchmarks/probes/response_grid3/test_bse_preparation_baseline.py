"""Remote-only baseline: live GW export vs caller-owned BSE snapshot reuse.

This is a preparation/ownership diagnostic, not an end-to-end speed test.
It uses actual H2/PBE/grid3 G0W0 output; no QP or screening poles are supplied.
Run in the project's allocated Slurm step, never on the local workstation.
"""
import os

import pytest


pytestmark = pytest.mark.skipif(
    not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'),
    reason='Numerical diagnostics require the allocated remote Slurm step',
)


@pytest.fixture(scope='module')
def actual_gw():
    import jax
    jax.config.update('jax_enable_x64', True)
    from gradscf import dft, gto, gw

    mol = gto.M(atom='H 0 0 0; H 0 0 .74', basis='sto-3g', cart=True)
    mf = dft.RKS(mol, xc='pbe', grids_level=3, execution_device='cpu',
                 conv_tol=1e-12, conv_tol_density=1e-11,
                 conv_tol_grad=1e-10, max_cycle=200).run()
    assert mf.converged
    calculation = gw.GW(mf, nw=32).density_fit('weigend').run()
    jax.block_until_ready(calculation.result)
    assert calculation.converged
    return calculation


def test_actual_snapshot_reuse_avoids_second_transform_and_preserves_stale_guard(actual_gw, monkeypatch):
    import jax
    import numpy as np
    from gradscf import bse
    import gradscf.gw.rgw as rgw

    original = rgw._mo_factors
    calls = []

    def counted(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(rgw, '_mo_factors', counted)
    first = bse.BSE(actual_gw, nroots=1)
    assert len(calls) == 0  # Construction is lazy; preparation happens in run().
    first.run()
    jax.block_until_ready(first.result)
    assert len(calls) == 1
    snapshot = first.reference
    reused = bse.BSE(snapshot, nroots=1).run()
    jax.block_until_ready(reused.result)
    assert len(calls) == 1
    assert reused.reference is snapshot
    np.testing.assert_allclose(reused.e, first.e, atol=1e-12, rtol=0)
    np.testing.assert_allclose(reused.oscillator_strength(), first.oscillator_strength(),
                               atol=1e-12, rtol=0)
    # Consumer-owned snapshot reuse must not add a full MO-factor owner to GW.
    assert all(value is not snapshot.mo_factors for value in vars(actual_gw).values())

    baseline = bse.BSE(actual_gw, nroots=1).run()
    jax.block_until_ready(baseline.result)
    calls_after_live_export = len(calls)
    np.testing.assert_allclose(baseline.e, first.e, atol=1e-12, rtol=0)
    print({'live_source_exports': 2, 'mo_transform_calls': calls_after_live_export,
           'extra_snapshot_transform_calls': 0,
           'factor_bytes': snapshot.mo_factors.nbytes,
           'note': 'Counts only; no performance or process-memory claim'})

    # Changing an actual GW control invalidates its live source before another
    # transform. An explicit snapshot is an independent fixed reference, not a
    # live cache and must not be reused to represent a subsequently changed GW.
    eta = actual_gw.eta
    try:
        actual_gw.eta *= 1.1
        with pytest.raises(RuntimeError, match='changed'):
            bse.BSE(actual_gw, nroots=1).run()
        with pytest.raises(RuntimeError, match='changed'):
            first.oscillator_strength()
        assert len(calls) == calls_after_live_export
    finally:
        actual_gw.eta = eta
