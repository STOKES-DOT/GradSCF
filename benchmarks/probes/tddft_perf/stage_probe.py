"""Separately synchronized preparation/J-K/XC microbenchmarks.

These preserve native consumers but break end-to-end fusion and cache
boundaries. Nested timings are NOT additive and are NOT a production profile.
"""
import json
from pathlib import Path
import sys
from dataclasses import replace
import jax
import jax.numpy as jnp
from worker import Recorder,reference

jax.config.update('jax_enable_x64',True)
cfg=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
log=Recorder(cfg,out);mf,ref=reference(cfg,log)
from gradscf.integrals import prepare_basis,make_plan
from gradscf.integrals.molecular.density_fitting import make_auxiliary_plan
top,params=prepare_basis(cfg['atom'],cfg['basis'],unit='Angstrom',cart=True)
aux,ap=prepare_basis(cfg['atom'],cfg['auxbasis'],unit='Angstrom',cart=True)
plan=make_plan(top);ri=make_auxiliary_plan(top,aux)
for name,fn in [
    ('one_electron',lambda:(plan.evaluate('overlap',params),plan.evaluate('kinetic',params),plan.evaluate('nuclear',params))),
    ('eri_s4',lambda:plan.evaluate('eri',params,aosym='s4')),
    ('ri_2c3c',lambda:ri.evaluate(params,ap)),
    ('ri_factors_complete',lambda:ri.factors(params,ap)),
]:
    for repeat in range(4):log.timed(name,fn,repeat,notes='independent prepared-input microbenchmark; phases are not additive')
from gradscf.tddft._semilocal_response import SemilocalResponseFunctional
functional=SemilocalResponseFunctional(cfg['xc'])
for repeat in range(4):
    log.timed('xc_hessian_fresh_reference',lambda:functional.grid_response_tensor(replace(ref)),repeat)
    log.timed('xc_hessian_same_reference',lambda:functional.grid_response_tensor(ref),repeat)
from gradscf.tddft.response import _jk_from_df_factors
q=jnp.asarray(ref.df_factors);d=jnp.asarray(mf.scf_result.density_matrix)[None]
fn=jax.jit(_jk_from_df_factors)
log.timed('ao_df_jk_first',lambda:fn(q,d))
for repeat in range(10):log.timed('ao_df_jk_warm',lambda:fn(q,d),repeat)
log.add('resident_reference',None,arrays={name:None if getattr(ref,name,None) is None else
    dict(shape=list(getattr(ref,name).shape),bytes=int(getattr(ref,name).size*getattr(ref,name).dtype.itemsize))
    for name in ('ao','ao_deriv1','df_factors','mo_coeff','rdm1')},
    notes='logical shapes; no claim that all arrays are distinct allocator buffers')
