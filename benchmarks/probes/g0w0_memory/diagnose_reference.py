"""Shared-input numerical diagnosis, never an independent GradSCF workflow."""
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("case", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    from gradscf import gto, scf
    from gradscf.gw.rgw import _run_g0w0_numerics
    from pyscf import gto as pgto, scf as pscf, lib
    from pyscf.gw.gw_cd import GWCD
    case = json.loads(args.case.read_text())
    own = scf.RHF(gto.M(atom=case["atom"], basis=case["basis"], cart=True),
                  conv_tol=1e-12).density_fit(case["auxbasis"]).run()
    mol = pgto.M(atom=case["atom"], basis=case["basis"], cart=True, verbose=0)
    ref = pscf.RHF(mol).density_fit(case["auxbasis"]).run(conv_tol=1e-12)
    b = np.asarray(own._scf_inputs.df_factors)
    pyb = lib.unpack_tril(np.concatenate(list(ref.with_df.loop())))
    target = np.array(case["orbs"])
    pairs = np.tril_indices(mol.nao_nr())
    gram = lambda x: x[:, pairs[0], pairs[1]].T@x[:, pairs[0], pairs[1]]
    report = dict(scope="shared-input diagnostics; PySCF inputs not an independent GradSCF result",
        native_naux=b.shape[0], pyscf_scf_naux=pyb.shape[0], aux_cart=ref.with_df.auxmol.cart,
        hf_energy_native=float(own.e_tot), hf_energy_pyscf=float(ref.e_tot),
        max_overlap_difference=float(np.max(abs(np.asarray(own._scf_inputs.overlap)-ref.get_ovlp()))),
        max_hcore_difference=float(np.max(abs(np.asarray(own.scf_result.hcore_matrix)-ref.get_hcore()))),
        max_mf_energy_difference=float(np.max(abs(np.asarray(own.mo_energy)-ref.mo_energy))),
        max_df_gram_difference=float(np.max(abs(gram(b)-gram(pyb)))))
    model = GWCD(ref)
    model.with_df = ref.with_df
    model.eta = case["eta_hartree"]
    mo_factors = np.asarray(model.ao2mo(ref.mo_coeff))
    py_mo = np.einsum('Qij,ip,jq->Qpq', pyb, ref.mo_coeff, ref.mo_coeff)
    report['gw_naux'] = mo_factors.shape[0]
    report['mf_vs_gw_mo_gram_difference'] = float(np.max(abs(
        mo_factors.reshape(mo_factors.shape[0], -1).T@mo_factors.reshape(mo_factors.shape[0], -1)
        -py_mo.reshape(py_mo.shape[0], -1).T@py_mo.reshape(py_mo.shape[0], -1))))
    model.kernel(nw=case['nw'], orbs=case['orbs'], Lpq=mo_factors)
    report['pyscf_qp'] = model.mo_energy[target].tolist()
    f, h, d = ref.get_fock(), ref.get_hcore(), ref.make_rdm1()
    common = dict(mo_energy=jnp.asarray(ref.mo_energy), nocc=mol.nelectron//2,
        nw=case['nw'], eta=case['eta_hartree'], orbs=tuple(case['orbs']),
        qp_tol=1e-10, resolvent_expansion=False)
    native = _run_g0w0_numerics(**common, mo_coeff=jnp.asarray(ref.mo_coeff),
        df_factors=jnp.asarray(pyb), fock_matrix=jnp.asarray(f),
        hcore_matrix=jnp.asarray(h), density_matrix=jnp.asarray(d))
    report['gradscf_on_pyscf_scf_input'] = np.asarray(native.mo_energy)[target].tolist()
    c = ref.mo_coeff
    vmf = c.T@(f-h-ref.get_j(dm=d))@c
    gw_input = _run_g0w0_numerics(**common, mo_coeff=jnp.eye(len(ref.mo_energy)),
        df_factors=jnp.asarray(mo_factors), fock_matrix=jnp.diag(jnp.asarray(ref.mo_energy)),
        hcore_matrix=jnp.asarray(c.T@h@c), density_matrix=jnp.diag(jnp.asarray(ref.mo_occ)),
        v_mf_matrix=jnp.asarray(vmf))
    report['gradscf_on_actual_pyscf_gw_input'] = np.asarray(gw_input.mo_energy)[target].tolist()
    report['shared_input_error_ev'] = float(np.max(abs(np.asarray(gw_input.mo_energy)[target]-model.mo_energy[target]))*27.211386245988)
    exact = np.array(model.mo_energy, copy=True)
    model.kernel(nw=case['nw'], orbs=case['orbs'], Lpq=mo_factors, vhf_df=True)
    report['pyscf_ri_exchange_qp'] = model.mo_energy[target].tolist()
    report['matched_ri_exchange_error_ev'] = float(np.max(abs(np.asarray(gw_input.mo_energy)[target]-model.mo_energy[target]))*27.211386245988)
    report['default_exact_vs_ri_exchange_shift_ev'] = ((exact[target]-model.mo_energy[target])*27.211386245988).tolist()
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == "__main__":
    main()
