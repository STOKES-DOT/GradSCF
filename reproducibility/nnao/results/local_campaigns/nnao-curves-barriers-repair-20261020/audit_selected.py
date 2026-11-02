"""Same-shell diagnostics only: PySCF is not the native solver's fallback."""
from pathlib import Path
import argparse
import json
import sys
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto, scf

sys.path.insert(0, 'tests/comparisons')
from nnao_core.native import _solver, make_inputs
from gradscf.integrals.molecular.density_fitting import unpack_factors
from gradscf.integrals.molecular.jk import build_jk_from_df
from gradscf.model.nnao.runtime import atomic_json, file_hash

ROOT = Path(__file__).resolve().parent
PARENT = Path('artifacts/nnao-curves-barriers-20261020/representative')
DEFAULT_KEYS = ['fb81f9654e02577e0508', 'f8ce7859d72f242c10e1', 'ea90966f585e090d1442',
                '20c7cb8f37f742ca8e54', 'a51aa9ee02b4857e5efb', '791331ccd7a3b1428add']


def physical(data, density_spin):
    factors = unpack_factors(jnp.asarray(data['factors']), len(data['overlap']))
    ds = jnp.asarray(density_spin)
    jt, _ = build_jk_from_df(factors, ds.sum(axis=0))
    ks = jnp.stack([build_jk_from_df(factors, dm)[1] for dm in ds])
    fock = data['hcore'][None]+jt[None]-ks
    energy = (data['enuc']+jnp.sum(ds.sum(axis=0)*data['hcore'])
              +.5*jnp.sum(ds.sum(axis=0)*jt)-.5*jnp.sum(ds*ks))
    return np.asarray(fock), float(energy), np.asarray(jt), np.asarray(ks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keys', nargs='+', default=DEFAULT_KEYS)
    parser.add_argument('--skip-native-iteration', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'audit-local.json')
    args = parser.parse_args()
    original = json.loads((PARENT/'results.json').read_text())
    records = {r['key']: r for r in original['results']}
    report = dict(status='running', backend=jax.default_backend(),
                  parent_sha256=file_hash(PARENT/'results.json'), results=[])
    for key in args.keys:
        old = records[key]
        row = dict(key=key, identifier=old['identifier'], basis=old['basis'],
                   previous_status=old['status'], previous_energy=old.get('energy_hartree'))
        tick = time.perf_counter()
        print('BEGIN', row, flush=True)
        try:
            exported = json.loads((PARENT/f'{key}-basis.json').read_text())
            molecule, shells = exported['row'], exported['atom_shells']
            data = make_inputs(molecule, shells)
            labels = [s+str(i) for i, s in enumerate(molecule['symbols'])]
            mol = gto.M(atom=list(zip(labels, molecule['coords_angstrom'])),
                basis=dict(zip(labels, shells)), charge=molecule['charge'],
                spin=molecule['spin'], cart=False, verbose=0)
            mf = (scf.UHF(mol) if molecule['spin'] else scf.RHF(mol)).density_fit(auxbasis=data['auxbasis'])
            mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-8, 150
            reference = mf.kernel()
            if not mf.converged:
                mf = mf.newton()
                reference = mf.kernel(dm0=mf.make_rdm1())
            row.update(reference_converged=bool(mf.converged), reference_energy=float(reference),
                nao=mol.nao, nelectron=mol.nelectron, spin=mol.spin,
                overlap_max_difference=float(np.max(abs(data['overlap']-mf.get_ovlp()))),
                hcore_max_difference=float(np.max(abs(data['hcore']-mf.get_hcore()))))
            dm = mf.make_rdm1()
            ds = np.asarray(dm) if mol.spin else np.stack([dm/2, dm/2])
            fock, energy, native_j, native_k = physical(data, ds)
            ref_j, ref_k = mf.get_jk(dm=dm)
            row.update(native_energy_at_reference_density=energy,
                fixed_density_energy_difference=energy-reference,
                coulomb_max_difference=float(np.max(abs(native_j-(ref_j.sum(axis=0) if mol.spin else ref_j)))),
                exchange_max_difference=float(np.max(abs(native_k-(ref_k if mol.spin else ref_k[None]/2)))),
                reference_orbital_residual=float(np.linalg.norm(fock@ds@data['overlap']-data['overlap']@ds@fock)),
                reference_spin_squared=float(mf.spin_square()[0]) if mol.spin else 0.,
                pyscf_four_index_eri_retained=mf._eri is not None)
            np.savez_compressed(args.output.with_name(key+'-reference-density.npz'), density_spin=ds,
                                mo_coeff=np.asarray(mf.mo_coeff), mo_occ=np.asarray(mf.mo_occ))
            if not args.skip_native_iteration:
                inputs = tuple(map(jnp.asarray, (data['overlap'], data['hcore'], data['factors'], data['enuc'])))
                state = _solver(mol.nelectron, mol.spin, 0.)(*inputs, None)
                if not bool(state['converged']):
                    shift = _solver(mol.nelectron, mol.spin, .2)(*inputs, state['density_spin'])
                    state = _solver(mol.nelectron, mol.spin, 0.)(*inputs, shift['density_spin'])
                sf, se, _, _ = physical(data, state['density_spin'])
                ndm = np.asarray(state['density_spin'])
                row.update(legacy_converged=bool(state['converged']), legacy_energy=float(state['energy']),
                    legacy_reconstructed_energy=se,
                    legacy_fock_max_difference=float(np.max(abs(sf-np.asarray(state['fock'])))),
                    legacy_physical_residual=float(np.linalg.norm(sf@ndm@data['overlap']-data['overlap']@ndm@sf)))
                # Diagnostic equality check only: this reference seed is not
                # used by the independent native retry workflow.
                seeded = _solver(mol.nelectron, mol.spin, 0.)(*inputs, jnp.asarray(ds))
                row.update(reference_seeded_native_converged=bool(seeded['converged']),
                           reference_seeded_native_energy=float(seeded['energy']))
            row['status'] = 'audited'
        except Exception as error:
            row.update(status='audit_failed', error=str(error), traceback=traceback.format_exc())
        row['seconds'] = time.perf_counter()-tick
        report['results'].append(row)
        atomic_json(args.output, report)
        print('END', json.dumps(row), flush=True)
        jax.clear_caches()
    report['status'] = 'complete'
    atomic_json(args.output, report)


if __name__ == '__main__':
    main()
