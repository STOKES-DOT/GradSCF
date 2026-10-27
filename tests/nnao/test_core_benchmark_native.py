"""Small native-integral gates before the full comparison campaign; no FD."""
from pathlib import Path
import sys

import jax
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'comparisons'))
from nnao_core.native import counterpoise, make_inputs, solve_hf, static_polarizability, traditional_shells

jax.config.update('jax_enable_x64', True)


def test_native_one_electron_hydrogen_has_no_self_interaction():
    row = dict(symbols=['H'], atomic_numbers=[1], coords_angstrom=[[0., 0., 0.]], charge=0, spin=1)
    data = make_inputs(row, traditional_shells(row, 'sto-3g'))
    result = solve_hf(data['overlap'], data['hcore'], data['factors'], 1, 1, data['enuc'])
    np.testing.assert_allclose(result['energy'], -.466581849557275, atol=1e-9)
    np.testing.assert_allclose(np.trace(result['density']@data['overlap']), 1., atol=1e-9)


def test_native_static_response_h2_is_symmetric_and_conserves_electrons():
    row = dict(symbols=['H', 'H'], atomic_numbers=[1, 1],
               coords_angstrom=[[0., 0., -.37], [0., 0., .37]], charge=0, spin=0)
    data = make_inputs(row, traditional_shells(row, 'sto-3g'))
    result = solve_hf(data['overlap'], data['hcore'], data['factors'], 2, 0, data['enuc'])
    response = static_polarizability(data, result, 2)
    alpha = np.asarray(response['tensor_au'])
    np.testing.assert_allclose(alpha, alpha.T, atol=1e-7)
    np.testing.assert_allclose(alpha[:2], 0., atol=1e-8)
    assert alpha[2, 2] > 1.
    assert max(abs(x) for x in response['electron_number_response']) < 1e-8
    assert all(response['converged'])


def test_native_counterpoise_keeps_common_metric_and_nested_spaces():
    row = dict(symbols=['He', 'He'], atomic_numbers=[2, 2],
               coords_angstrom=[[0., 0., -1.], [0., 0., 1.]], charge=0, spin=0)
    result = counterpoise(row, traditional_shells(row, '6-31g'), [[0], [1]])
    assert result['nao'] == 4
    assert result['bsse_hartree'] >= -1e-9
    np.testing.assert_allclose(result['cp_interaction_hartree']-result['uncorrected_interaction_hartree'],
                               result['bsse_hartree'], atol=1e-12)


def test_native_field_response_matches_independent_pyscf_cphf():
    pyscf = pytest.importorskip('pyscf')
    from pyscf.scf import cphf
    pyscf.lib.num_threads(2)
    row = dict(symbols=['H', 'H'], atomic_numbers=[1, 1],
               coords_angstrom=[[0., 0., -.37], [0., 0., .37]], charge=0, spin=0)
    data = make_inputs(row, traditional_shells(row, 'sto-3g'))
    state = solve_hf(data['overlap'], data['hcore'], data['factors'], 2, 0, data['enuc'])
    actual = static_polarizability(data, state, 2)
    mol = pyscf.gto.M(atom=list(zip(row['symbols'], row['coords_angstrom'])),
                      basis='sto-3g', cart=False, verbose=0)
    mf = pyscf.scf.RHF(mol).density_fit(auxbasis='def2-universal-jkfit')
    mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-10
    mf.kernel()
    assert mf.converged
    occupied, virtual = mf.mo_coeff[:, mf.mo_occ > 0], mf.mo_coeff[:, mf.mo_occ == 0]
    position = mol.intor_symmetric('int1e_r', comp=3)
    h1 = np.einsum('pa,xpq,qi->xai', virtual, position, occupied)
    response = mf.gen_response(hermi=1)

    def induced(mo1):
        density = 2*np.einsum('pa,xai,qi->xpq', virtual, mo1, occupied)
        density += density.transpose(0, 2, 1)
        return np.einsum('pa,xpq,qi->xai', virtual, response(density), occupied)

    mo1 = cphf.solve(induced, mf.mo_energy, mf.mo_occ, h1, max_cycle=100, tol=1e-12, verbose=0)[0]
    density = 2*np.einsum('pa,xai,qi->xpq', virtual, mo1, occupied)
    density += density.transpose(0, 2, 1)
    expected = -np.einsum('ipq,jqp->ij', position, density)
    np.testing.assert_allclose(state['energy'], mf.e_tot, atol=1e-9, rtol=0)
    np.testing.assert_allclose(actual['tensor_au'], expected, atol=1e-7, rtol=0)


@pytest.mark.parametrize('charge,spin', [(0, 2), (-1, 3)])
def test_open_shell_carbon_states_match_independent_df_uhf(charge, spin):
    pyscf = pytest.importorskip('pyscf')
    row = dict(symbols=['C'], atomic_numbers=[6], coords_angstrom=[[0., 0., 0.]],
               charge=charge, spin=spin)
    shells = traditional_shells(row, 'STO-6G+P')
    data = make_inputs(row, shells)
    actual = solve_hf(data['overlap'], data['hcore'], data['factors'], 6-charge, spin, data['enuc'])
    mol = pyscf.gto.M(atom='C 0 0 0', basis={'C': shells[0]}, charge=charge,
                      spin=spin, cart=False, verbose=0)
    reference = pyscf.scf.UHF(mol).density_fit(auxbasis='def2-universal-jkfit')
    reference.conv_tol, reference.conv_tol_grad, reference.max_cycle = 1e-12, 1e-9, 200
    reference.kernel()
    assert reference.converged
    np.testing.assert_allclose(actual['energy'], reference.e_tot, atol=1e-8, rtol=0)
    np.testing.assert_allclose(actual['electron_numbers'], [(6-charge+spin)/2, (6-charge-spin)/2], atol=1e-8)


def test_native_cp_five_energies_match_independent_common_auxiliary_pyscf():
    pyscf = pytest.importorskip('pyscf')
    from pyscf.df.addons import make_auxmol
    row = dict(symbols=['He', 'He'], atomic_numbers=[2, 2],
               coords_angstrom=[[0., 0., -1.], [0., 0., 1.]], charge=0, spin=0)
    shells = traditional_shells(row, '6-31g')
    actual = counterpoise(row, shells, [[0], [1]])
    basis = {'He': shells[0]}
    whole = pyscf.gto.M(atom=[('He', row['coords_angstrom'][0]), ('He', row['coords_angstrom'][1])],
                        basis=basis, cart=False, verbose=0)
    auxiliary = make_auxmol(whole, 'def2-universal-jkfit')
    configurations = [
        [('He', row['coords_angstrom'][0]), ('He', row['coords_angstrom'][1])],
        [('He', row['coords_angstrom'][0])], [('He', row['coords_angstrom'][1])],
        [('He', row['coords_angstrom'][0]), ('ghost-He', row['coords_angstrom'][1])],
        [('ghost-He', row['coords_angstrom'][0]), ('He', row['coords_angstrom'][1])],
    ]
    expected = []
    for atom in configurations:
        mol = pyscf.gto.M(atom=atom, basis=basis, cart=False, verbose=0)
        mf = pyscf.scf.RHF(mol).density_fit(auxbasis='def2-universal-jkfit')
        # DF.build() otherwise reconstructs auxmol from the fragment. Supply
        # the independently computed common-auxiliary factors themselves.
        mf.with_df._cderi = pyscf.df.incore.cholesky_eri(mol, auxmol=auxiliary, verbose=0)
        mf.with_df.auxmol = None
        mf.chkfile = None
        mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-10
        mf.kernel()
        assert mf.converged
        expected.append(mf.e_tot)
    observed = [actual['dimer_energy_hartree'], *actual['own_energies_hartree'], *actual['ghost_energies_hartree']]
    np.testing.assert_allclose(observed, expected, atol=1e-8, rtol=0)
