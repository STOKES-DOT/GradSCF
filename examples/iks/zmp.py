"""GradSCF ZMP on two small molecular targets, with a finite-lambda audit.

No CLI. PBE SCF supplies each target once; replace its AO density to invert an
external correlated density. KSPies is not imported by this calculation.
"""
from pathlib import Path
import json
import os
import platform
from time import perf_counter

import jax
import numpy as np

jax.config.update('jax_enable_x64', True)

from gradscf import dft, gto, integrals
from gradscf.dft import iks
from gradscf.integrals.grids.blocks import AOGrid


output = Path(os.environ.get('GRADSCF_IKS_VALIDATION', 'artifacts/iks-kspies-20261110'))
output.mkdir(parents=True, exist_ok=True)
cases = [
    dict(name='heh_plus', atom='He 0 0 0; H 0 0 1.5', charge=1, spin=0, layout='rks'),
    dict(name='lithium', atom='Li 0 0 0', charge=0, spin=1, layout='uks'),
]
penalties = (1., 5., 10., 20., 50., 100.)
for case in cases:
    started = perf_counter()
    mol = gto.M(atom=case['atom'], basis='sto-3g', unit='Bohr', cart=True,
                charge=case['charge'], spin=case['spin'])
    method = dft.RKS if case['layout'] == 'rks' else dft.UKS
    mf = method(mol, xc='pbe', grids_level=1, execution_device='cpu',
                max_memory=4096., conv_tol=1e-11, conv_tol_grad=1e-8, max_cycle=150).run()
    if not mf.converged:
        raise RuntimeError('Converge the target-generating PBE calculation first')
    state = mf.to_reference()
    target = iks.Target(mf.make_rdm1(), state.overlap_matrix,
                        mol.nelectron if case['layout']=='rks' else
                        ((mol.nelectron+case['spin'])//2, (mol.nelectron-case['spin'])//2),
                        layout=case['layout'], energy=mf.e_tot)
    basis = integrals.basis_from_molecule_spec(mol.to_spec(), basis='sto-3g')
    packed = integrals.eri_pair_matrix_packed(basis)
    # Only J is requested. The same ZMP interface accepts direct or DF actions.
    coulomb = lambda density: integrals.build_jk_from_packed(
        packed, density, with_k=False, max_memory=4096.)[0]
    grid = AOGrid.from_basis(basis, state.grid.coords, state.grid.weights, block_size=512)
    # The fixed Fermi-Amaldi guide is built from the target total density.
    base = iks.guide_fock(target, state.h1e, coulomb)
    folder = output / case['name']
    folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/'target.npz', density=np.asarray(target.density),
        overlap=np.asarray(target.overlap), hcore=np.asarray(state.h1e))
    settings = dict(case, basis='sto-3g', unit='Bohr', cart=True, lambdas=penalties,
                    guide='fermi_amaldi', target_method='GradSCF PBE, grid1', target_energy_hartree=float(mf.e_tot))
    (folder/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    rows = []
    for index, penalty in enumerate(penalties):
        result = iks.invert(target, base_fock=base, method='zmp', coulomb=coulomb,
                            grid=grid, hcore=state.h1e, lambdas=penalties[:index+1],
                            scf_tolerance=1e-9, coulomb_tolerance=1e-6,
                            density_tolerance=1e-3, maxiter=400, max_memory=4096.)
        np.savez_compressed(folder/f'gradscf_lambda_{penalty:g}.npz',
                            density=np.asarray(result.density), fock=np.asarray(result.fock))
        row = dict(penalty=penalty, scf_converged=result.converged, inverse_valid=result.valid,
            scf_residual_hartree=result.scf_residual, coulomb_error_hartree=result.coulomb_energy,
            relative_density_error=result.relative_density_error, gap_hartree=result.gap,
            stages=[dict(penalty=s.penalty, iterations=s.iterations, residual=s.scf_residual) for s in result.stages])
        rows.append(row)
        print(case['name'], json.dumps(row), flush=True)
        if not result.converged:
            raise RuntimeError('Finite-lambda SCF did not converge; preserve its diagnostics')
    (folder/'gradscf.json').write_text(json.dumps(dict(rows=rows, seconds=perf_counter()-started,
        node=platform.node(), backend=jax.default_backend(), jax=jax.__version__, dtype='float64'), indent=2)+'\n')

# A converged finite-lambda equation can still fail the requested density gate.
# inverse_valid and scf_converged are intentionally different diagnostics.
