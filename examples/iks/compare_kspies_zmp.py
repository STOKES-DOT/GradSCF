"""Independent KSPies/PySCF audit of saved GradSCF ZMP results. No GradSCF import.

Set GRADSCF_IKS_VALIDATION to the output of zmp.py; put a pinned KSPies source
checkout on PYTHONPATH. This comparison does not generate or modify targets.
"""
from pathlib import Path
import contextlib
import json
import os

import numpy as np
from pyscf import gto
from scipy import special

# KSPies' unused angular plotting helper imports the legacy SciPy name.
# Adapt only this reference process; leave the upstream source and solver intact.
if not hasattr(special, 'sph_harm'):
    special.sph_harm = lambda m, n, theta, phi: special.sph_harm_y(n, m, phi, theta)
    np.testing.assert_allclose(special.sph_harm(0, 0, .3, .7), 1/np.sqrt(4*np.pi), atol=1e-14)
    np.testing.assert_allclose(special.sph_harm(0, 1, .3, .7),
                                np.sqrt(3/(4*np.pi))*np.cos(.7), atol=1e-14)

from kspies import zmp


root = Path(os.environ.get('GRADSCF_IKS_VALIDATION', 'artifacts/iks-kspies-20261110'))
rows = []
for name in ('heh_plus', 'lithium'):
    folder = root/name
    settings = json.loads((folder/'settings.json').read_text())
    with np.load(folder/'target.npz', allow_pickle=False) as saved:
        target = saved['density']; overlap = saved['overlap']; hcore = saved['hcore']
    mol = gto.M(atom=settings['atom'], basis=settings['basis'], unit='Bohr',
                cart=True, charge=settings['charge'], spin=settings['spin'], verbose=0)
    np.testing.assert_allclose(mol.intor_symmetric('int1e_ovlp'), overlap, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(mol.intor_symmetric('int1e_kin')+mol.intor_symmetric('int1e_nuc'),
                                hcore, atol=1e-10, rtol=1e-10)
    method = zmp.RZMP if settings['layout']=='rks' else zmp.UZMP
    reference = method(mol, target)
    reference.guide = 'faxc'  # KSPies name for the same Fermi-Amaldi guide
    reference.diis_space = 2  # Avoid rank-deficient histories in these tiny AO spaces.
    reference.conv_tol_dm = 1e-11
    reference.conv_tol_diis = 1e-10
    reference.max_cycle = 1200
    with (folder/'kspies.log').open('w') as log, contextlib.redirect_stdout(log):
        for penalty in settings['lambdas']:
            reference.level_shift = max(.2, penalty)
            reference.zscf(penalty)
            if not reference.converged:
                raise RuntimeError('Independent KSPies finite-lambda SCF did not converge')
            with np.load(folder/f'gradscf_lambda_{penalty:g}.npz', allow_pickle=False) as saved:
                result = dict(saved)
            j = reference.mf.get_jk(mol, reference.dm)[0]
            factor = 1 if settings['layout']=='rks' else 2
            physical_fock = np.asarray(reference.F0)+factor*penalty*(j-reference.J_tar)
            density_error = float(np.max(np.abs(result['density']-reference.dm)))
            fock_error = float(np.max(np.abs(result['fock']-physical_fock)))
            row = dict(case=name, penalty=penalty, density_max_abs=density_error,
                        fock_max_abs_hartree=fock_error, kspies_converged=bool(reference.converged))
            rows.append(row)
            np.testing.assert_allclose(result['density'], reference.dm, atol=2e-7, rtol=0.)
            np.testing.assert_allclose(result['fock'], physical_fock, atol=1e-6, rtol=0.)
for row in rows:
    print(json.dumps(row), flush=True)
(root/'kspies_comparison.json').write_text(json.dumps(dict(status='passed', rows=rows, reference_controls=dict(diis_space=2, level_shift='max(0.2,lambda)', max_cycle=1200, density_tolerance=1e-11, commutator_tolerance=1e-10), legacy_scipy_alias_process_only=True), indent=2)+'\n')
