"""Check native FFI forward/VJP disk-cache reload, including changed operands."""
import importlib.util,json,os,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from gradscf.tools.jax_runtime import configure_jax_persistent_cache
from gradscf import integrals
from gradscf.model.nnao import prepare_direct_basis
out=Path(os.environ['CACHE_TASK'])/f'persistent-{os.environ["SLURM_JOB_ID"]}'
out.mkdir(parents=True,exist_ok=True)
configure_jax_persistent_cache(cache_dir=str(out/'disk'),min_compile_time_secs=0.,min_entry_size_bytes=-1,xla_caches=None)
spec=importlib.util.spec_from_file_location('persistent_native',Path.cwd()/'tools/train_nnao_joint_ad.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
atom='H 0 0 0; H 0 0 .74'
layout=prepare_direct_basis(atom,cart=False,basis_family='szp663_direct',core_primitives=6)
at,ap=integrals.prepare_basis(atom,'def2-universal-jkfit',cart=False)
aux=integrals.make_auxiliary_plan(layout.topology,at)
context=dict(layout=layout,plan=integrals.make_plan(layout.topology),aux=aux,ap=ap,metric=aux.metric_factor(layout.parameters,ap))
f,pb=m.native_projection(context);x=layout.reference_outputs();beta=jnp.zeros(14)
def evaluate():
 start=time.perf_counter();physical=f(x,beta);cot=jax.tree.map(jnp.ones_like,physical)
 gradient=pb(x,beta,cot);jax.block_until_ready((physical,gradient))
 return jax.tree.map(np.asarray,(physical,gradient)),time.perf_counter()-start
first,t1=evaluate();f.clear_cache();pb.clear_cache();second,t2=evaluate()
for a,b in zip(jax.tree.leaves(first),jax.tree.leaves(second)):np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-10)
changed=f(x,beta.at[0].set(.02));assert not np.allclose(changed[1],first[0][1],atol=1e-9)
report=dict(first_seconds=t1,reload_seconds=t2,cache_files=len(list((out/'disk').glob('*'))),
 max_difference=max(float(np.max(np.abs(a-b))) for a,b in zip(jax.tree.leaves(first),jax.tree.leaves(second))))
(out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
