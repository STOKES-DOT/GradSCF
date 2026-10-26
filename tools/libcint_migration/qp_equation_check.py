"""Shared-input CD equation check at completed QP energies; no root shortcut.

The input is the saved GradSCF HF/screening snapshot from a completed workflow.
This checks the original auxiliary equation against its exact RPA resolvent at
fixed frequencies, including the frequency derivative. It is not another
independent end-to-end GW calculation or a new QP solution for the eta sweep.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import numpy as np
    import gradscf
    from gradscf.gw.freq import scaled_legendre_grid
    from gradscf.gw.g0w0 import _exchange_mo
    from gradscf.gw.polarizability import rho_response_iw
    from gradscf.gw.screened import screened_w_imag_axis
    from gradscf.gw.poles import rpa_resolvent, screened_w_imag_resolvent
    from gradscf.gw.qp import sigma_cd_batch, _df_dw_batch
    from gradscf.integrals.molecular.jk import build_j_from_df

    started = time.perf_counter()
    screening = np.load(args.input/'screening.npz', allow_pickle=False)
    reference = np.load(args.input/'reference.npz', allow_pickle=False)
    energy = jnp.asarray(reference['mo_energy'])
    screen = jnp.asarray(screening['screening_energy'])
    qp = jnp.asarray(screening['qp_energy'])
    factors = jnp.asarray(screening['mo_factors'])
    nocc = int(screening['nocc'])
    nmo = energy.size
    coeff, overlap = jnp.asarray(reference['mo_coeff']), jnp.asarray(reference['overlap'])
    inverse = coeff.T@overlap
    # Complete S-orthonormal orbitals: C^-1=C.T S. This is a conversion of
    # saved data, not an external integral/SCF provider.
    ao_factors = jnp.einsum('pi,Qpq,qj->Qij', inverse, factors, inverse)
    coulomb = build_j_from_df(ao_factors, jnp.asarray(reference['density']))
    v_mf = jnp.asarray(reference['fock'])-jnp.asarray(reference['hcore'])-coulomb
    delta = jnp.diag(_exchange_mo(factors, nocc)-coeff.T@v_mf@coeff)
    frequencies, weights = scaled_legendre_grid(100)
    ov = factors[:, :nocc, nocc:]
    w_imag = screened_w_imag_axis(factors,
        lambda w: rho_response_iw(w, (screen[:nocc], screen[nocc:]), ov, spin_factor=4.),
        frequencies)
    w_static = screened_w_imag_axis(factors,
        lambda w: rho_response_iw(w, (screen[:nocc], screen[nocc:]), ov, spin_factor=4.),
        jnp.zeros(1))[0]
    jax.block_until_ready((w_imag, w_static, delta))
    report = dict(status='running', scope='shared-input fixed-frequency CD equation diagnostic',
        input_origin='completed independent GradSCF HF to G0W0 to BSE workflow',
        conversion_included_in_preparation=True, source=gradscf.__file__,
        input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (args.input/'screening.npz', args.input/'reference.npz')},
        nao=int(overlap.shape[0]), nmo=int(nmo), nocc=nocc, naux=int(factors.shape[0]),
        nw=100, dtype='float64', device='cpu', affinity=sorted(os.sched_getaffinity(0)),
        gates=dict(sigma_atol_hartree=1e-8, sigma_rtol=2e-8,
                   slope_atol=2e-6, slope_rtol=2e-6,
                   qp_residual_at_base_eta_hartree=1e-6), rows=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    report['preparation_seconds'] = time.perf_counter()-started
    save()
    evaluate = jax.jit(lambda w, e, dv, sh, st:
        (sigma_cd_batch(w, sh, st), _df_dw_batch(w, e, dv, sh, st)))
    for eta, selected in ((1e-5, tuple(range(nmo))), (1e-6, (11, 17, 21)), (1e-4, (11, 17, 21))):
        indices = jnp.asarray(selected)
        shared = dict(mo_energy=energy, channels=(((screen[:nocc], screen[nocc:]), ov, 2.),),
            ef=.5*(energy[nocc-1]+energy[nocc]), eta=jnp.asarray(eta), freqs=frequencies,
            wts=weights, occupation_sign=jnp.where(jnp.arange(nmo)<nocc, -1., 1.))
        stacked = dict(wmn_p=w_imag[:, :, indices].transpose(2, 0, 1),
            wmn_static=w_static[:, indices].T, b_pm=factors[:, indices, :].transpose(1, 0, 2),
            b_mp=factors[:, :, indices].transpose(2, 0, 1), p_index=indices)
        target = qp[indices]
        tick = time.perf_counter()
        old, old_slope = jax.block_until_ready(evaluate(target, energy[indices], delta[indices], shared, stacked))
        direct_seconds = time.perf_counter()-tick
        model = rpa_resolvent(screen, ov, factors, eta=jnp.asarray(eta))
        if eta == 1e-5:
            equivalent = screened_w_imag_resolvent(model, frequencies)
            np.testing.assert_allclose(equivalent, w_imag, atol=1e-9, rtol=1e-9)
        tick = time.perf_counter()
        new, new_slope = jax.block_until_ready(evaluate(target, energy[indices], delta[indices],
            dict(shared, resolvent_data=model), stacked))
        residual = target-energy[indices]-jnp.real(old)-delta[indices]
        row = dict(eta_hartree=eta, orbitals=list(selected), fixed_qp_frequency_hartree=np.asarray(target).tolist(),
            direct_sigma_real=np.asarray(old.real).tolist(), direct_sigma_imag=np.asarray(old.imag).tolist(),
            resolvent_sigma_real=np.asarray(new.real).tolist(), resolvent_sigma_imag=np.asarray(new.imag).tolist(),
            direct_slope=np.asarray(old_slope).tolist(), resolvent_slope=np.asarray(new_slope).tolist(),
            direct_dyson_residual_hartree=np.asarray(residual).tolist(),
            max_sigma_difference_hartree=float(jnp.max(jnp.abs(old-new))),
            max_slope_difference=float(jnp.max(jnp.abs(old_slope-new_slope))),
            direct_diagnostic_seconds=direct_seconds, resolvent_diagnostic_seconds=time.perf_counter()-tick)
        report['rows'].append(row)
        save()
        np.testing.assert_allclose(new, old, atol=1e-8, rtol=2e-8)
        np.testing.assert_allclose(new_slope, old_slope, atol=2e-6, rtol=2e-6)
        if eta == 1e-5:
            assert np.max(np.abs(residual)) < 1e-6
    report.update(status='ok', elapsed_seconds=time.perf_counter()-started,
                  peak_rss_bytes=1024*resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    save()
    print(json.dumps(dict(status=report['status'], rows=len(report['rows']), output=str(args.output))))


if __name__ == '__main__':
    main()
