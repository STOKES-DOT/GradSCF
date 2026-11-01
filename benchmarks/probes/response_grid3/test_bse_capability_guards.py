"""Remote public capability guards after actual tiny DFT references.

These tests execute refusals, not scGW continuation or spinor GW/BSE numerics.
No QP poles, GWResult, spin conversion or stand-in mean-field object is made.
"""
import os

import pytest


pytestmark = pytest.mark.skipif(
    not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'),
    reason='Numerical capability checks require the allocated remote Slurm step',
)


@pytest.fixture(scope='module', params=('rks', 'gks'))
def tiny_dft(request):
    import jax
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_device', jax.devices('cpu')[0])
    from gradscf import dft, gto

    mol = gto.M(atom='H 0 0 0; H 0 0 .74', basis='sto-3g', cart=True)
    mf = getattr(dft, request.param.upper())(
        mol, xc='pbe', grids_level=3, execution_device='cpu',
        conv_tol=1e-12, conv_tol_density=1e-11, conv_tol_grad=1e-10,
        max_cycle=200).run()
    jax.block_until_ready(mf.scf_result)
    assert mf.converged
    return request.param, mf


def test_public_bse_capability_guards_after_actual_dft(tiny_dft):
    from gradscf import bse, gw

    ks, mf = tiny_dft
    if ks == 'rks':
        calculation = gw.GW(mf, method='scgw', nw=8)
        # The public exporter rejects this method before a GW solve. There is
        # no reason to run a costly Matsubara fixed point for this known guard.
        for call in (calculation.get_bse_inputs,
                     lambda: bse.BSE(calculation, nroots=1).run()):
            with pytest.raises(NotImplementedError, match='analytic continuation and QP'):
                call()
        assert calculation.result is None and calculation.mo_energy is None
        print({'case': 'RKS/PBE -> scGW -> BSE', 'status': 'runtime_verified_unsupported',
               'scope': 'public export guard after converged DFT; no scGW solve or poles'})
        return

    # GKS has an actual spinor SCF result but no restricted response payload.
    restricted = gw.GW(mf, method='g0w0', nw=8)
    with pytest.raises(RuntimeError, match='restricted SCF before GW'):
        restricted.run()
    assert restricted.result is None
    unrestricted = gw.UGW(mf, method='g0w0', nw=8)
    with pytest.raises(NotImplementedError, match='Generalized spinor response'):
        unrestricted.run()
    assert unrestricted.result is None
    with pytest.raises(NotImplementedError, match='BSE accepts a real GW/UGW facade or explicit BSEReference'):
        bse.BSE(mf, nroots=1).run()
    print({'case': 'GKS/PBE -> GW/BSE', 'status': 'runtime_verified_unsupported',
           'scope': 'actual GKS state rejected by GW, UGW and direct BSE source adapters',
           'restricted_gw_error_kind': 'missing restricted SCF payload RuntimeError'})
