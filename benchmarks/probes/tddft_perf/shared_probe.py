"""Same orbitals, occupations, energies, grid, and random X/Y in both engines.

The reference is independently prepared by GradSCF. PySCF receives that state
only in this explicitly marked operator diagnostic. Each engine still supplies
its own exact J/K and XC kernel. Generation/transfer costs are recorded apart.
"""
import json
from pathlib import Path
import sys
import jax
import jax.numpy as jnp
import numpy as np
from worker import Recorder,reference,ready

jax.config.update('jax_enable_x64',True)
cfg=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
log=Recorder(cfg,out)
mf,ref=reference(cfg,log)
from pyscf import dft,gto,lib
lib.num_threads(cfg['threads'])
pmol=gto.M(atom=cfg['atom'],basis=cfg['basis'],cart=True,verbose=0)
pmf=dft.RKS(pmol);pmf.xc=cfg['xc']
pmf.mo_coeff=np.asarray(mf.mo_coeff);pmf.mo_energy=np.asarray(mf.mo_energy);pmf.mo_occ=np.asarray(mf.mo_occ)
pmf.grids.coords=np.asarray(ref.grid.coords);pmf.grids.weights=np.asarray(ref.grid.weights)
pmf.small_rho_cutoff=0
td=pmf.TDDFT() if cfg['full'] else pmf.TDA()
pvind,pdiag=log.timed('shared_pyscf_prepare',td.gen_vind)
from gradscf.tddft.response import build_restricted_tda_operator,build_restricted_tdhf_operator
from gradscf.tddft._semilocal_response import SemilocalResponseFunctional
functional=SemilocalResponseFunctional(cfg['xc'])
if cfg['full']:
    vind=log.timed('shared_gradscf_prepare',lambda:build_restricted_tdhf_operator(ref,functional))
else:
    vind,_,_=log.timed('shared_gradscf_prepare',lambda:build_restricted_tda_operator(ref,functional))
x=jnp.asarray(np.random.default_rng(7).normal(size=(5,pdiag.size)))
compiled=log.timed('shared_compile',lambda:jax.jit(vind).lower(x).compile())
expected=ready(pvind(np.asarray(x)));actual=ready(compiled(x))
error=float(np.max(abs(np.asarray(actual)-expected)))
log.add('shared_action_validation',None,action_max_abs_error=error,input_source='GradSCF SCF; exact shared grid/orbitals; separate J/K/XC',
        notes='kernel diagnostic only; excluded from independent end-to-end ratios')
assert error<2e-8
for repeat in range(10):
    log.timed('shared_gradscf_action',lambda:compiled(x),repeat)
    log.timed('shared_pyscf_action',lambda:pvind(np.asarray(x)),repeat)
original_gradient=log.timed('shared_xy_vjp',lambda:jax.jit(jax.grad(lambda x:jnp.sum(vind(x)**2)))(x))
if not cfg['full'] and cfg['xc']=='pbe0':
    from experiments import install
    install(dict(experiment='packed-public'))
    alt,_,_=build_restricted_tda_operator(ref,functional)
    alternative=ready(jax.jit(alt)(x))
    derivative=log.timed('shared_alternative_xy_vjp',lambda:jax.jit(jax.grad(lambda x:jnp.sum(alt(x)**2)))(x))
    log.add('shared_optimized_ad_check',None,action_max_abs_error=float(jnp.max(abs(alternative-actual))),
            gradient_error=float(jnp.max(abs(derivative-original_gradient))))
