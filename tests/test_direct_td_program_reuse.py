"""Direct response compilation preserves dynamic parameters and cache policy."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import integrals, tdscf
from gradscf.integrals.backends.native.jk import NativeDirectBasis
from gradscf.scf.molecules import RestrictedMolecule, QuadratureGrid
from test_td_execution_reuse import compilations


def reference(shift=0.):
    topology, parameters = integrals.prepare_basis("H 0 0 0; H 0 0 .8", "6-31g", cart=True)
    basis = NativeDirectBasis(integrals.make_plan(topology), parameters)
    n = topology.nao
    occupations = jnp.array([1.] + [0.]*(n-1))
    energies = jnp.linspace(1., 3., n).at[0].set(-2.)
    energies = energies.at[1:].add(shift)
    return RestrictedMolecule(ao=jnp.zeros((1,n)),grid=QuadratureGrid(jnp.ones(1)),
        dipole_integrals=jnp.zeros((3,n,n)),rep_tensor=jnp.zeros((0,0,0,0)),
        mo_coeff=jnp.stack([jnp.eye(n)]*2),mo_occ=jnp.stack([occupations]*2),
        mo_energy=jnp.stack([energies]*2),rdm1=jnp.stack([jnp.diag(occupations)]*2),
        h1e=jnp.diag(energies),nuclear_repulsion=jnp.asarray(0.),nocc=1,
        exact_exchange_fraction=1.,direct_basis=basis)


def driver(ref,full=False,*,cutoff=1e-12,cache='none'):
    cls=tdscf.TDDFT if full else tdscf.TDA
    return cls(ref,xc_functional='hf',nstates=2,davidson_tol=1e-9,davidson_max_iter=80,
        response_kernel_options=dict(two_electron_mode='direct',direct_tol=cutoff,
                                     integral_cache=cache,direct_workers=1,max_memory=1024.))


@pytest.mark.parametrize('full',[False,True],ids=['tda','full'])
def test_direct_same_signature_reuses_program_and_keeps_new_energies(monkeypatch,full):
    from gradscf.tddft.execution import _cacheable
    first=driver(reference(),full)
    assert _cacheable(first._build_solver()), 'No-ERI-cache direct path must use a stable program'
    result=jax.block_until_ready(first.kernel())
    moved=driver(reference(.08),full)
    with compilations(monkeypatch) as events:
        repeat=jax.block_until_ready(first.kernel())
        new=jax.block_until_ready(moved.kernel())
    assert result.converged and repeat.converged and new.converged
    np.testing.assert_allclose(result.excitation_energies,repeat.excitation_energies,atol=1e-12)
    assert float(jnp.min(new.excitation_energies-result.excitation_energies)) > .07
    assert not events, f'{len(events)} unintended direct-response compilations'


def test_small_automatic_exact_direct_cache_keeps_eager_selection():
    from gradscf.tddft.execution import _cacheable
    current=driver(reference(),cutoff=0.,cache='auto')._build_solver()
    assert not _cacheable(current), 'Tracing must not replace the existing exact packed cache'


def test_compiled_direct_parameter_gradient_keeps_basis_values_live(monkeypatch):
    ref=reference()
    def energy(scale):
        basis=ref.direct_basis
        exponents=list(basis.parameters.exponents)
        exponents[0]=exponents[0].at[0].multiply(jnp.exp(scale))
        changed=replace(basis,parameters=replace(basis.parameters,exponents=tuple(exponents)))
        return driver(replace(ref,direct_basis=changed)).kernel().excitation_energies[0]
    derivative=jax.jit(jax.value_and_grad(energy))
    point=jnp.asarray(.03)
    value,gradient=derivative(point)
    jax.block_until_ready((value,gradient))
    with compilations(monkeypatch) as events:
        updated=derivative(jnp.asarray(.04));jax.block_until_ready(updated)
    assert not events
    step=1e-4
    fd=(energy(point+step)-energy(point-step))/(2*step)
    assert abs(float(gradient))>1e-7
    np.testing.assert_allclose(gradient,fd,atol=3e-7,rtol=3e-5)
