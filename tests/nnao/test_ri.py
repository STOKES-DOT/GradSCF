"""NNAO RI never builds primitive ERIs; its coefficient gradient is variational."""
import runpy
import jax.numpy as jnp
import numpy as np
import pytest


def test_methane_ri_gradient_and_independent_df_energy(monkeypatch):
    from gradscf.integrals.plan import IntegralPlan
    original=IntegralPlan.evaluate
    def checked(self,op,*args,**kwargs):
        assert op!='eri','RI coefficient optimization must not allocate four-center ERIs'
        return original(self,op,*args,**kwargs)
    monkeypatch.setattr(IntegralPlan,'evaluate',checked)
    cls=runpy.run_path('tools/optimize_methane_nnao.py')['MethaneRHF']
    ex=cls(basis_family='szp663_direct',eri_backend='ri')
    assert ex.rep.ndim==2 and ex.rep.size < ex.primitive_topology.nao**4/10
    x=ex.layout.reference_outputs();energy,g,info=ex.evaluate(x)
    assert info['converged'] and info['reconstruction_error']<1e-9
    direction=jnp.zeros_like(x).at[0,0,4].set(.2).at[0,1,4].set(-.3).at[1:,1,1].set(.1)
    step=1e-4;e=lambda h:ex.evaluate(x+h*direction)[0]
    fd=(8*(e(step)-e(-step))-e(2*step)+e(-2*step))/(12*step)
    np.testing.assert_allclose(jnp.sum(g*direction),fd,atol=2e-6,rtol=2e-5)
    pytest.importorskip('pyscf')
    from pyscf import gto,scf
    labels=[f'{s}{i}' for i,s in enumerate(ex.symbols)]
    mol=gto.M(atom=list(zip(labels,ex.coords)),basis=dict(zip(labels,ex.layout.atom_shells(ex.layout.bind(x)))),
              unit='Angstrom',cart=False,verbose=0)
    mf=scf.RHF(mol).density_fit(auxbasis=ex.auxbasis)
    mf.conv_tol=1e-12;mf.conv_tol_grad=1e-9;mf.max_cycle=150
    expected=mf.kernel();assert mf.converged
    np.testing.assert_allclose(energy,expected,atol=1e-8,rtol=0)
