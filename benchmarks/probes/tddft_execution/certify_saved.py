"""Check saved restricted states using a separately imported parent operator.

This is a fixed-input diagnostic, not an independent end-to-end speed test.
SCF/integrals/grid are independently rebuilt; saved orbitals, energies and
density are then bound to that reference. No eigensolve or dense A is used.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from worker import Recorder, ready, reference, solver

jax.config.update('jax_enable_x64', True)


def state_columns(saved):
    c = saved['mo_coeff']
    if c.ndim == 3:
        c = c[0]
    occ = saved['mo_occ']
    if occ.ndim == 2:
        occ = occ[0]
    nocc = np.count_nonzero(occ)
    names = ('x_amplitudes', 'y_amplitudes') if 'x_amplitudes' in saved else ('amplitudes',)
    # Whiten AO transition matrices with the AO metric, then compare thin
    # subspaces. This tolerates both orbital and degenerate-state rotations.
    l = np.linalg.cholesky(saved['overlap'])
    columns = []
    for name in names:
        p = np.einsum('pi,nia,qa->npq', c[:, :nocc], saved[name], c[:, nocc:])
        columns.append(np.einsum('pa,npq,qb->nab', l, p, l).reshape(len(p), -1))
    return np.concatenate(columns, axis=1).T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('states', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text())
    cfg.update(device='cpu', variant='parent-operator-certificate')
    log = Recorder(cfg, args.output)
    started = time.perf_counter()
    mf, ref = reference(cfg, log)
    saved = np.load(args.states)
    np.testing.assert_allclose(ref.overlap_matrix, saved['overlap'], atol=2e-12, rtol=2e-12)
    fields = {name: jnp.asarray(saved[name]) for name in
              ('mo_coeff', 'mo_occ', 'mo_energy', 'rdm1', 'fock_matrix') if name in saved}
    ref = replace(ref, **fields)
    low = replace(solver(cfg, mf)._build_solver(), molecule=ref)
    vind = low.gen_tdhf_vind() if cfg['full'] else low.gen_tda_vind()
    names = ('x_amplitudes', 'y_amplitudes') if cfg['full'] else ('amplitudes',)
    vectors = np.concatenate([saved[name].reshape(len(saved['energies']), -1) for name in names], axis=1)
    acted = np.asarray(ready(vind(jnp.asarray(vectors))))
    residuals = np.linalg.norm(acted-saved['energies'][:, None]*vectors, axis=1)/np.linalg.norm(vectors, axis=1)
    result = dict(source='independently imported parent operator on saved numerical inputs',
                  residual_per_root=residuals.tolist(), tolerance=cfg['td_tol'],
                  scientific_pass=bool(np.all(residuals <= cfg['td_tol'])),
                  elapsed_seconds=time.perf_counter()-started)
    if args.compare:
        other = np.load(args.compare)
        result['energy_error_ev'] = float(np.max(np.abs(saved['energies']-other['energies']))*27.211386245988)
        left, right = state_columns(saved), state_columns(other)
        cuts = [0]+(np.flatnonzero(np.diff(other['energies']) > 1e-5)+1).tolist()+[len(other['energies'])]
        clusters = []
        for start, end in zip(cuts[:-1], cuts[1:]):
            a = np.linalg.qr(left[:, start:end], mode='reduced')[0]
            b = np.linalg.qr(right[:, start:end], mode='reduced')[0]
            distance = np.sqrt(max(0., 2*(end-start)-2*np.linalg.norm(a.T@b)**2))
            clusters.append(dict(roots=list(range(start, end)), projector_frobenius=float(distance)))
        result.update(clusters=clusters, cluster_threshold_ha=1e-5, projector_tolerance=1e-4)
        result['scientific_pass'] &= result['energy_error_ev'] <= 1e-4 and all(x['projector_frobenius'] < 1e-4 for x in clusters)
    (args.output/'certificate.json').write_text(json.dumps(result, indent=2)+'\n')
    if not result['scientific_pass']:
        raise RuntimeError('Saved states failed parent-operator or subspace certification')


if __name__ == '__main__':
    main()
