"""Check the actual anthracene SCF/full-TDDFT cost before nuclear response.

The PubChem structure is an initial conformer, not a stationary DFT geometry.
All electronic results here are explicitly preliminary.
"""
import hashlib
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import jax
import numpy as np
from common import OUTPUT, SETTINGS, event, mean_field, response, save
from gradscf.integrals._native import register_integrals

start = perf_counter()
jax.config.update('jax_default_device', jax.devices('cpu')[0])
native = Path(register_integrals()._name).resolve()
save('environment.json',dict(python=sys.executable, python_version=platform.python_version(),
    source_revision=Path('SOURCE_REVISION').read_text().strip(), native_path=str(native),
    native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
    jax_version=jax.__version__, hostname=platform.node(), job=os.environ.get('SLURM_JOB_ID'),
    affinity=sorted(os.sched_getaffinity(0)), devices=[str(d) for d in jax.devices()],
    settings=SETTINGS, numerical_scope='Preliminary unoptimized geometry; no FC/HT result yet'))
mf = mean_field()
coordinates = np.asarray(mf.mol.to_spec().coords_bohr)
np.savez(OUTPUT/'initial_scf.npz', coordinates=coordinates, mo_coeff=np.asarray(mf.mo_coeff),
    mo_energy=np.asarray(mf.mo_energy), mo_occ=np.asarray(mf.mo_occ), density=np.asarray(mf.make_rdm1()), energy=float(mf.e_tot))
nocc = int(np.count_nonzero(np.asarray(mf.mo_occ)))
event('dimensions', natom=len(coordinates), nao=int(mf.mo_coeff.shape[0]), nocc=nocc,
    nvir=int(mf.mo_coeff.shape[1])-nocc)
td, dipole, strengths = response(mf)
np.savez(OUTPUT/'initial_td.npz', energy=np.asarray(td.e), dipole=dipole, oscillator_strength=strengths)
save('initial_status.json',dict(complete=True, optimized_geometry=False, elapsed_seconds=perf_counter()-start))
