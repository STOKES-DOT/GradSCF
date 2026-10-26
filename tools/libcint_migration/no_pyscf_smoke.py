"""Installed-package acceptance in a production environment without PySCF.

Run outside the source checkout with PYTHONPATH unset. This is a small interface
fixture, not a trained NNAO checkpoint or a force-supervision test.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def main():
    assert importlib.util.find_spec('pyscf') is None
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import numpy as np
    import gradscf
    from gradscf import bse, dft, gto, gw, mp, scf
    from gradscf.integrals._native import register_integrals
    from gradscf.model.nnao import prepare_direct_basis
    from gradscf.model.nnao.integrals.context import prepare_native_context
    from gradscf.model.nnao.integrals.variable_exponents import native_projection
    from gradscf.model.nnao.training.objective import _df_rhf_solver

    module = Path(gradscf.__file__).resolve()
    assert module.is_relative_to(Path(sys.prefix).resolve()), (module, sys.prefix)
    native = Path(register_integrals()._name).resolve()
    assert native.is_relative_to(Path(sys.prefix).resolve()), native
    mol = gto.M(atom='H 0 0 0; H 0 0 .74', basis='sto-3g', cart=True)
    hf = scf.RHF(mol, conv_tol=1e-12).run()
    pbe = dft.RKS(mol, xc='pbe', grids_level=1, conv_tol=1e-12).run()
    correlated = mp.MP2(hf, with_t2=True).run()
    assert hf.converged and pbe.converged and correlated.converged
    np.testing.assert_allclose(hf.e_tot, -1.1167593073964255, atol=1e-10, rtol=0)
    np.testing.assert_allclose(correlated.e_corr, -.013138073589533, atol=1e-9, rtol=0)
    qp = gw.GW(hf, nw=8, eta=.02).run()
    assert qp.converged
    excitations = {}
    for tda in (True, False):
        response = bse.BSE(qp, nroots=1, tda=tda).run()
        assert np.all(response.converged)
        excitations['tda' if tda else 'full'] = np.asarray(response.e).tolist()

    atom = [('H', (0., 0., 0.)), ('H', (0., 0., .74))]
    layout = prepare_direct_basis(atom, cart=False, basis_family='szp663_direct')
    context = prepare_native_context(layout, atom, 'def2-universal-jkfit')
    physical, pullback = native_projection(context, layout.exponent_scale_keys)
    coefficients = layout.reference_outputs()
    beta = jnp.array([.03, -.02])
    one, with_gradient = _df_rhf_solver(2, 1e-8, 0.)
    inputs = physical(coefficients, beta)
    enuc = scf.nuclear_repulsion_energy(layout.parameters.nuclear_coords,
                                      jnp.array(layout.topology.nuclear_charges))
    (value, info), cotangent = with_gradient(inputs, enuc)
    dc, db = pullback(coefficients, beta, cotangent)
    assert bool(info['converged'])
    assert np.all(np.isfinite(dc)) and np.all(np.isfinite(db))
    assert np.linalg.norm(db) > 1e-9
    updated = physical(coefficients-1e-4*dc, beta-1e-4*db)
    next_value, next_info = one(updated, enuc)
    assert bool(next_info['converged']) and np.isfinite(next_value)
    print(json.dumps(dict(status='ok', python=sys.executable, module=str(module),
        native=str(native), native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        pyscf_available=False, hf=float(hf.e_tot), pbe=float(pbe.e_tot),
        mp2=float(correlated.e_corr), t2_shape=list(correlated.t2.shape),
        qp_energy=np.asarray(qp.mo_energy).tolist(), bse=excitations,
        nnao_energy=float(value), nnao_updated_energy=float(next_value),
        coefficient_gradient_norm=float(jnp.linalg.norm(dc)),
        exponent_gradient_norm=float(jnp.linalg.norm(db))), indent=2))


if __name__ == '__main__':
    main()
