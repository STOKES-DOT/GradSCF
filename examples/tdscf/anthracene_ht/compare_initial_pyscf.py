"""Independent PySCF check of the preliminary anthracene vertical spectrum.

The level-3 grids are independently generated, so this is not a bitwise aligned
operator test. No PySCF quantity is supplied to a GradSCF production calculation.
"""
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from pyscf import dft, gto, lib

case=json.loads(Path(__file__).with_name('input.json').read_text())
root=Path(__file__).resolve().parents[3]
out=root/'reports/anthracene_ht/20261110-b3lyp-def2svp/raw'
lib.num_threads(64)
start=perf_counter()
mol=gto.M(atom=case['atom'],basis=case['basis'],unit=case['unit'],cart=True,verbose=3)
mf=dft.RKS(mol)
# Match GradSCF's complete B3LYP recipe, including VWN-RPA correlation.
mf.xc='.2*HF+.08*LDA_X+.72*GGA_X_B88,.19*LDA_C_VWN_RPA+.81*GGA_C_LYP'
mf.grids.level=case['grid_level'];mf.small_rho_cutoff=0.
mf.max_memory=60000;mf.conv_tol=1e-11;mf.conv_tol_grad=1e-8;mf.max_cycle=200
mf.kernel()
scf_seconds=perf_counter()-start
if not mf.converged:raise RuntimeError('Independent reference SCF failed.')
td=mf.TDDFT();td.nstates=4;td.max_memory=128;td.lindep=1e-20;td.conv_tol=case['td_tol'];td.max_cycle=200
td.kernel()
# Preserve every result, including a failure, before applying acceptance gates.
vind,_=td.gen_vind(mf)
xy=np.array([np.r_[x.ravel(),y.ravel()] for x,y in td.xy])
residuals=np.linalg.norm(vind(xy)-np.asarray(td.e)[:,None]*xy,axis=1)/np.linalg.norm(xy,axis=1)
gradscf=np.load(out/'initial_td.npz');ground=np.load(out/'initial_scf.npz')
error=np.asarray(td.e)-gradscf['energy'][:td.nstates]
result=dict(independent_reference='PySCF',grid_scope='independent level-3 grids, pruning not point-aligned',
    xc=mf.xc,ground_energy_hartree=float(mf.e_tot),ground_error_hartree=float(mf.e_tot-ground['energy']),
    energies_hartree=np.asarray(td.e).tolist(),energy_difference_hartree=error.tolist(),
    oscillator_strength=np.asarray(td.oscillator_strength()).tolist(),
    roots=td.nstates,solver_max_memory_mb=td.max_memory,lindep=td.lindep,converged_per_root=np.asarray(td.converged).tolist(),
    original_operator_residual_hartree=residuals.tolist(),
    scf_seconds=scf_seconds,total_seconds=perf_counter()-start,threads=64,
    job=os.environ.get('SLURM_JOB_ID'),production_input_uses_reference=False,
    warning='Differences include independent grid discretization; no speed ratio is calculated.')
(out/'pyscf_initial_final.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
if not np.all(td.converged) or np.max(residuals)>1e-8:
    raise RuntimeError('Independent reference root residual/convergence gate failed.')
if np.max(abs(error))>1e-4:
    raise RuntimeError('Vertical differences exceed the coarse-grid diagnostic gate; align grids before proceeding.')
