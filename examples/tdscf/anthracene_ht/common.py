"""Shared inputs and bookkeeping for the gas-phase anthracene calculation."""
import json
from pathlib import Path
from time import perf_counter

import jax
import numpy as np
jax.config.update("jax_enable_x64", True)

from gradscf import dft, gto, tdscf

ROOT = Path(__file__).resolve().parents[3]
SETTINGS = json.loads(Path(__file__).with_name("input.json").read_text())
OUTPUT = ROOT / "reports/anthracene_ht/20261110-b3lyp-def2svp/raw"
OUTPUT.mkdir(parents=True, exist_ok=True)
SYMBOLS = [row.split()[0] for row in SETTINGS['atom'].split(';')]


def save(name, value):
    path = OUTPUT / name
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def event(stage, **values):
    record = dict(stage=stage, **values)
    with (OUTPUT/'progress.jsonl').open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False)+'\n')
    print(json.dumps(record), flush=True)


def mean_field(coordinates=None, density=None):
    atom = SETTINGS['atom'] if coordinates is None else list(zip(SYMBOLS, np.asarray(coordinates)))
    mol = gto.M(atom=atom, basis=SETTINGS['basis'], unit='Angstrom' if coordinates is None else 'Bohr', cart=True)
    mf = dft.RKS(mol, xc=SETTINGS['xc'], grids_level=SETTINGS['grid_level'],
        jk_backend='direct', direct_scf_workers=SETTINGS['direct_workers'],
        direct_scf_tol=SETTINGS['direct_tol'], max_memory=SETTINGS['max_memory_mb'],
        conv_tol=SETTINGS['scf_energy_tol'], conv_tol_density=SETTINGS['scf_density_tol'],
        conv_tol_grad=SETTINGS['scf_gradient_tol'], max_cycle=200,
        init_guess='atom' if density is None else density, execution_device='gpu',
        grid_screen_tol=0.)
    start = perf_counter()
    event('scf_start', natom=len(SYMBOLS), basis=SETTINGS['basis'], xc=SETTINGS['xc'])
    mf.run()
    jax.block_until_ready((mf.e_tot,mf.mo_coeff,mf.mo_energy))
    event('scf', seconds=perf_counter()-start, energy_ha=float(mf.e_tot),
          converged=bool(mf.converged), cycles=int(mf.cycles))
    if not mf.converged:
        raise RuntimeError('The ground-state SCF did not converge.')
    return mf


def response(mf, *, nstates=None):
    nstates = SETTINGS['roots'] if nstates is None else nstates
    td = tdscf.TDDFT(mf, nstates=nstates, davidson_tol=SETTINGS['td_tol'],
        davidson_max_iter=200, response_kernel_options=dict(max_memory=SETTINGS['max_memory_mb'],
        direct_workers=SETTINGS['direct_workers'], direct_tol=0., integral_cache='none', grid_block_size=2048))
    start = perf_counter()
    event('full_tddft_start', roots=nstates, integral_cache='none', direct_tol=0.)
    with jax.default_device(jax.devices('gpu')[0]):
        td.kernel()
        moment = td.transition_dipole()
        strength = td.oscillator_strength()
        jax.block_until_ready((td.e, moment, strength))
    event('full_tddft', seconds=perf_counter()-start, converged=bool(np.all(td.converged)),
          energies_ha=np.asarray(td.e).tolist(), oscillator_strength=np.asarray(strength).tolist())
    if not np.all(td.converged):
        raise RuntimeError('Full TDDFT did not converge.')
    return td, np.asarray(moment), np.asarray(strength)
