"""Data/grid APIs with lazy compatibility exports for basis construction."""
from importlib import import_module

_EXPORTS = {'CartesianAO': 'gradscf.integrals.basis',
 'CartesianBasis': 'gradscf.integrals.basis',
 'basis_from_molecule_spec': 'gradscf.integrals.basis',
 'basis_from_spec': 'gradscf.integrals.basis',
 'basis_from_pyscf_spec': 'gradscf.integrals.basis',
 'cartesian_angular_tuples': 'gradscf.integrals.basis',
 'basis_from_pyscf_mol_cart': 'gradscf.integrals.basis',
 'build_molecular_grid': 'gradscf.data.grid',
 'build_molecular_grid_from_spec': 'gradscf.data.grid',
 'MoleculeSpec': 'gradscf.data.molecule',
 'atomic_number': 'gradscf.data.molecule',
 'parse_molecule_spec': 'gradscf.data.molecule',
 'load_basis_from_snapshot': 'gradscf.data.pyscf_basis_loader',
 'evaluate_cartesian_ao': 'gradscf.data.grid_ao',
 'GRADDFT_GROUND_ATOM_SYMBOLS': 'gradscf.data.graddft_dataset',
 'GRADDFT_GROUND_TEST_ATOMS': 'gradscf.data.graddft_dataset',
 'GRADDFT_XND_ATOM_ENERGY_COLUMN': 'gradscf.data.graddft_dataset',
 'GradDFTGroundAtomSplit': 'gradscf.data.graddft_dataset',
 'GradDFTGroundAtomRecord': 'gradscf.data.graddft_dataset',
 'GradDFTGroundAtomTrainTestData': 'gradscf.data.graddft_dataset',
 'build_graddft_ground_atom_datum': 'gradscf.data.graddft_dataset',
 'build_graddft_ground_atom_molecule': 'gradscf.data.graddft_dataset',
 'build_graddft_ground_atom_train_test_data': 'gradscf.data.graddft_dataset',
 'graddft_ground_atom_split': 'gradscf.data.graddft_dataset',
 'load_graddft_ground_atom_records': 'gradscf.data.graddft_dataset',
 'neutral_atom_spin': 'gradscf.data.graddft_dataset',
 'parse_graddft_test_train_ratio': 'gradscf.data.graddft_dataset',
 'split_graddft_ground_atom_records': 'gradscf.data.graddft_dataset'}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
