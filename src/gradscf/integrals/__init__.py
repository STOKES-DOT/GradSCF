"""Lazy public exports for the integral layer."""
from importlib import import_module

_EXPORTS = {'DirectJKResult': 'gradscf.integrals.jk',
 'build_direct_jk_from_basis': 'gradscf.integrals.jk',
 'build_direct_jk_incremental': 'gradscf.integrals.jk',
 'build_hcore': 'gradscf.integrals.one_electron',
 'dipole_element': 'gradscf.integrals.one_electron',
 'dipole_matrix': 'gradscf.integrals.one_electron',
 'kinetic_element': 'gradscf.integrals.one_electron',
 'kinetic_matrix': 'gradscf.integrals.one_electron',
 'nuclear_attraction_element': 'gradscf.integrals.one_electron',
 'nuclear_attraction_matrix': 'gradscf.integrals.one_electron',
 'overlap_hcore_matrices': 'gradscf.integrals.one_electron',
 'overlap_element': 'gradscf.integrals.one_electron',
 'overlap_matrix': 'gradscf.integrals.one_electron',
 'rinv_element': 'gradscf.integrals.one_electron',
 'rinv_matrices': 'gradscf.integrals.one_electron',
 'rinv_matrix': 'gradscf.integrals.one_electron',
 'build_j_from_eri_pair_matrix': 'gradscf.integrals.layouts',
 'build_jk_from_eri_pair_matrix': 'gradscf.integrals.layouts',
 'eri_pair_matrix_to_mo_eri_slices': 'gradscf.integrals.layouts',
 'schwarz_bounds': 'gradscf.integrals.screening',
 'shell_pair_schwarz_bounds': 'gradscf.integrals.screening',
 'eri_element': 'gradscf.integrals.two_electron',
 'eri_pair_matrix_packed': 'gradscf.integrals.two_electron',
 'eri_tensor': 'gradscf.integrals.two_electron',
 'eri_tensor_screened': 'gradscf.integrals.two_electron',
 'precompile_eri_kernels': 'gradscf.integrals.two_electron',
 'build_libcint_mol': 'gradscf.integrals.backends.pyscf_mol',
 'libcint_intor_name': 'gradscf.integrals.backends.pyscf_mol',
 'BasisTopology': 'gradscf.integrals.basis',
 'BasisParameters': 'gradscf.integrals.basis',
 'prepare_basis': 'gradscf.gto.basis',
 'make_plan': 'gradscf.integrals.plan',
 'IntegralPlan': 'gradscf.integrals.plan',
 'backend_capabilities': 'gradscf.integrals.capabilities',
 'build_rks_integral_inputs': 'gradscf.integrals.assembly',
 'build_uks_integral_inputs': 'gradscf.integrals.assembly',
 'RKSIntegralInputs': 'gradscf.integrals.assembly',
 'UKSIntegralInputs': 'gradscf.integrals.assembly'}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
