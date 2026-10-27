"""Same-signature TD programs retain dynamic reference and model inputs."""
from dataclasses import replace
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf import dft, tdscf
from gradscf.scf.molecules import QuadratureGrid, RestrictedMolecule


@contextmanager
def compilations(monkeypatch):
    from jax._src import compiler
    original = compiler.backend_compile_and_load
    events = []
    def count(*args, **kwargs):
        events.append(1)
        return original(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(compiler, 'backend_compile_and_load', count)
        yield events


def reference(shift=0., dtype=jnp.float64):
    n = 5
    ao = jnp.asarray(np.linspace(.2, .8, 35).reshape(7, n), dtype=dtype)
    factors = jnp.asarray(np.cos(np.arange(50)).reshape(2, n, n)*.015, dtype=dtype)
    factors = .5*(factors+factors.transpose(0, 2, 1))
    occupation = jnp.asarray([1., 1., 0., 0., 0.], dtype=dtype)
    energies = jnp.asarray([-1.1, -.6, .2, .65, 1.1], dtype=dtype)
    energies = energies.at[2:].add(shift)
    return RestrictedMolecule(
        ao=ao, ao_deriv1=jnp.stack([ao, .1*ao, -.2*ao, .05*ao]),
        grid=QuadratureGrid(jnp.ones(7, dtype=dtype)*.05),
        dipole_integrals=jnp.zeros((3,n,n), dtype=dtype),
        rep_tensor=jnp.zeros((0,0,0,0), dtype=dtype), df_factors=factors,
        mo_coeff=jnp.stack([jnp.eye(n,dtype=dtype)]*2),
        mo_occ=jnp.stack([occupation]*2),mo_energy=jnp.stack([energies]*2),
        rdm1=jnp.stack([jnp.diag(occupation)]*2),
        h1e=jnp.diag(energies),nuclear_repulsion=jnp.asarray(0.,dtype=dtype),
        nocc=2,scf_converged=True,scf_cycles=8)


def driver(ref, full=False, xc='pbe', params=None, roots=2):
    cls=tdscf.TDDFT if full else tdscf.TDA
    return cls(ref,xc_functional=xc,xc_params=params,nstates=roots,
               eigensolver='davidson',davidson_tol=1e-9,davidson_max_iter=80)


@pytest.mark.parametrize('full',[False, True], ids=['tda','full'])
def test_new_td_objects_reuse_program_and_update_reference(monkeypatch,full):
    first=driver(reference(),full)
    old=jax.block_until_ready(first.kernel())
    moved=replace(reference(.07),scf_cycles=11)
    with compilations(monkeypatch) as events:
        repeated=jax.block_until_ready(first.kernel())
        new=jax.block_until_ready(driver(moved,full).kernel())
    assert old.converged and repeated.converged and new.converged
    np.testing.assert_allclose(old.excitation_energies,repeated.excitation_energies,atol=1e-12)
    assert float(jnp.min(new.excitation_energies-old.excitation_energies))>.06
    assert not events, f'{len(events)} unexpected backend compilations'


def density_inputs(state):
    return state.density_matrix


def neural_energy(params,density):
    hidden=jnp.tanh(density @ params['weight'])
    return params['scale']*jnp.sum(hidden**2)


def test_custom_neural_parameters_and_ad_remain_dynamic(monkeypatch):
    xc=dft.custom_xc(input_fn=density_inputs,energy_fn=neural_energy,name='reuse-test')
    ref=reference()
    weights=jnp.asarray(np.linspace(.1,.4,15).reshape(5,3))
    def energy(scale):
        return driver(ref,xc=xc,params={'weight':weights,'scale':scale}).kernel().excitation_energies[0]
    value=jax.jit(energy)
    gradient=jax.jit(jax.grad(energy))
    point=jnp.asarray(.03)
    a=value(point);g=gradient(point);jax.block_until_ready((a,g))
    with compilations(monkeypatch) as events:
        b=value(jnp.asarray(.04));dg=gradient(jnp.asarray(.04));jax.block_until_ready((b,dg))
    assert abs(float(a-b))>1e-7
    assert not events
    step=1e-4
    fd=(value(point+step)-value(point-step))/(2*step)
    np.testing.assert_allclose(g,fd,atol=2e-7,rtol=2e-5)


def test_solver_structure_changes_compile_then_stay_warm(monkeypatch):
    ref=reference()
    jax.block_until_ready(driver(ref,roots=2).kernel())
    with compilations(monkeypatch) as events:
        jax.block_until_ready(driver(ref,roots=1).kernel())
    assert events, 'root count selects a different fixed-shape program'
    with compilations(monkeypatch) as events:
        jax.block_until_ready(driver(reference(.03),roots=1).kernel())
    assert not events


def test_dynamic_occupation_inference_keeps_eager_compatibility():
    from gradscf.tddft.execution import _cacheable
    current=driver(replace(reference(),nocc=None))._build_solver()
    assert not _cacheable(current)
    actual=jax.block_until_ready(current.tda(2))
    assert actual.converged


def test_unknown_frozen_grid_hf_callback_is_not_statically_captured():
    from dataclasses import dataclass
    from gradscf.tddft.execution import _cacheable
    @dataclass(frozen=True)
    class GridHF:
        exact_exchange_fraction: float=0.
        def grid_hf_fraction(self,molecule):return jnp.asarray(.25)
        def local_kernel(self,density):return jnp.zeros_like(density)
    current=driver(reference(),xc=GridHF())._build_solver()
    assert not _cacheable(current)


@pytest.mark.parametrize('field', ['runtime_scf_options','hfx_nu_api','response_df_metadata'])
def test_hashable_mutable_producer_metadata_keeps_uncached_path(field):
    from gradscf.tddft.execution import _cacheable
    class Producer:
        value = .1
    # Identity hashing does not establish an immutable numerical contract.
    current = driver(replace(reference(), **{field: Producer()}))._build_solver()
    assert not _cacheable(current)


def test_eager_neural_grad_reuses_program(monkeypatch):
    xc=dft.custom_xc(input_fn=density_inputs,energy_fn=neural_energy,name='eager-reuse-test')
    ref=reference();weights=jnp.ones((5,3))*.2
    def energy(scale):return driver(ref,xc=xc,params={'weight':weights,'scale':scale}).kernel().excitation_energies[0]
    derivative=jax.value_and_grad(energy)
    jax.block_until_ready(derivative(jnp.asarray(.03)))
    with compilations(monkeypatch) as events:
        jax.block_until_ready(derivative(jnp.asarray(.04)))
    assert not events


def metadata_inputs(state):
    return state.density_matrix, state._molecule.scf_cycles


def metadata_energy(params,inputs):
    density,cycles=inputs
    return params['scale']*(1.+.01*jnp.asarray(cycles))*jnp.sum(density**2)


def test_response_does_not_erase_custom_input_metadata(monkeypatch):
    xc=dft.custom_xc(input_fn=metadata_inputs,energy_fn=metadata_energy)
    first=driver(reference(),xc=xc,params={'scale':jnp.asarray(.03)})
    a=jax.block_until_ready(first.kernel()).excitation_energies
    with compilations(monkeypatch) as events:
        b=jax.block_until_ready(driver(replace(reference(),scf_cycles=11),xc=xc,
            params={'scale':jnp.asarray(.03)}).kernel()).excitation_energies
    assert float(jnp.max(abs(a-b)))>1e-6
    assert not events


def test_numpy_mo_reference_uses_same_numerical_program(monkeypatch):
    ref=reference()
    host=replace(ref,mo_coeff=np.asarray(ref.mo_coeff),mo_energy=np.asarray(ref.mo_energy))
    jax.block_until_ready(driver(host).kernel())
    moved=replace(host,mo_energy=np.asarray(reference(.04).mo_energy))
    with compilations(monkeypatch) as events:
        result=jax.block_until_ready(driver(moved).kernel())
    assert result.converged
    assert not events


def test_compiled_response_preserves_eager_nonfinite_validation():
    ref=replace(reference(),fock_matrix=jnp.eye(5).at[0,0].set(jnp.nan))
    with pytest.raises(ValueError,match='NaN|finite'):
        driver(ref).kernel()


def test_dtype_and_grid_shape_changes_are_new_signatures(monkeypatch):
    ref=reference()
    jax.block_until_ready(driver(ref).kernel())
    bigger=replace(ref,ao=jnp.concatenate((ref.ao,ref.ao[:1])),
        ao_deriv1=jnp.concatenate((ref.ao_deriv1,ref.ao_deriv1[:,:1]),axis=1),
        grid=QuadratureGrid(jnp.ones(8)*.05))
    with compilations(monkeypatch) as events:
        result=jax.block_until_ready(driver(bigger).kernel())
    assert events and result.converged
    def hf_tda(molecule):
        return tdscf.TDA(molecule,xc_functional='hf',nstates=2,eigensolver='davidson',davidson_tol=1e-6)
    jax.block_until_ready(hf_tda(ref).kernel())
    changed=reference(dtype=jnp.float32)
    with compilations(monkeypatch) as events:
        low=jax.block_until_ready(hf_tda(changed).kernel())
    assert events and low.converged


def test_cached_tda_eigenvector_property_gradient_keeps_response():
    xc=dft.custom_xc(input_fn=density_inputs,energy_fn=neural_energy)
    ref=reference();weights=jnp.ones((5,3))*.2
    def observable(scale):
        td=tdscf.TDA(ref,xc_functional=xc,xc_params={'weight':weights,'scale':scale},
            nstates=2,eigensolver='davidson',davidson_tol=1e-10,
            tda_gradient_mode='implicit_eigenvector',eigenvector_adjoint_tol=1e-11)
        amplitude=td.kernel().amplitudes[0]
        return jnp.sum(amplitude[:,0]**2)
    point=jnp.asarray(.03);step=1e-4
    derivative=jax.jit(jax.grad(observable))(point)
    fd=(observable(point+step)-observable(point-step))/(2*step)
    np.testing.assert_allclose(derivative,fd,atol=3e-7,rtol=3e-5)
