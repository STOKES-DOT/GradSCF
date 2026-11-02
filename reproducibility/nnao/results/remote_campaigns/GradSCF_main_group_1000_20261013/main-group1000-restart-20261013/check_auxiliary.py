"""Compare standard DF with native shell-direct J/K; never form full ERIs."""
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.linalg import eigh

from gradscf import integrals
from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.integrals.molecular.density_fitting import unpack_factors
from gradscf.integrals.molecular.jk import build_jk_from_df

results = []
for symbol, distance in [('Cl',1.27),('Br',1.414),('I',1.609),('Sb',1.7),('Te',1.7)]:
    count = 3 if symbol == 'Sb' else 2 if symbol == 'Te' else 1
    atom = [(symbol,[0.,0.,0.])]+[('H',[distance*(i==0), distance*(i==1), distance*(i==2)]) for i in range(count)]
    layout = prepare_direct_basis(atom, basis_family='nnao_szp_s', core_primitives='reference')
    params = layout.bind(layout.reference_outputs())
    plan = integrals.make_plan(layout.topology)
    s = np.asarray(plan.evaluate('overlap', params))
    h = np.asarray(plan.evaluate('kinetic', params)+plan.evaluate('nuclear', params))
    _, c = eigh(h, s)
    occupied = c[:,:layout.nelectron//2]
    density = jnp.asarray(2*occupied@occupied.T)
    exact_j, exact_k = plan.get_jk(params, density)
    at, ap = integrals.prepare_basis(atom, 'def2-universal-jkfit', cart=False)
    aux = integrals.make_auxiliary_plan(layout.topology, at)
    b = unpack_factors(aux.factors(params, ap), layout.topology.nao)
    df_j, df_k = build_jk_from_df(b, density)
    error = float(.5*jnp.sum(density*(df_j-exact_j))-.25*jnp.sum(density*(df_k-exact_k)))
    row = dict(element=symbol, density='core-Hamiltonian occupied orbitals, not self-consistent',
        nao=layout.topology.nao, naux=at.nao, df_two_electron_energy_error_hartree=error,
        max_j_error=float(jnp.max(jnp.abs(df_j-exact_j))), max_k_error=float(jnp.max(jnp.abs(df_k-exact_k))))
    results.append(row)
    print(json.dumps(row), flush=True)
Path(__file__).with_suffix('.json').write_text(json.dumps(results, indent=2)+'\n')
