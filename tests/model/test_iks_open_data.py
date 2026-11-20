"""Opt-in real-data acceptance of orbital inputs and nonlinear eight-parameter potentials."""
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gradscf.model.autoxc import Candidate,Parameter
from gradscf.model.autoxc.history import source_digest,versions
from gradscf.model.autoxc.workflows.xc100.molecule import XC100Case
from gradscf.model.autoxc.workflows.xc100.reference import reference_baseline
from gradscf.model.autoxc.workflows.xc100.campaign import write_json,memory


ROOT=os.environ.get('GRADSCF_XC100_ACCEPTANCE')
pytestmark=pytest.mark.skipif(not ROOT,reason='Opt-in qualified XC100 GPU data')


@pytest.mark.parametrize('identifier',('010','017','065'))
def test_open_orbital_potential_on_molecule(identifier):
    assert jax.default_backend()=='gpu' and len(jax.devices())==1 and jax.config.jax_enable_x64
    assert len(os.sched_getaffinity(0))<=64
    root=Path(ROOT);start=time.perf_counter()
    case=XC100Case(root/'data/xc100',identifier,block_size=8192,max_parameters=None)
    case.prepare(baseline=reference_baseline(root,case,include_energy=False))
    assert case.baseline_energy is None
    parameters=tuple(Parameter(n,.001,-.2,.2) for n in 'abcd')+tuple(Parameter(n,1.,.1,4.) for n in 'klpt')
    candidate=Candidate('n**0.3333333333333333*(a*log1p(k*u)+b*u/(1+l*u**2)+c*u*tanh(alpha)*exp(-p*u)+d*u*virtual_0/(n+virtual_0)*exp(-t*u))',
                         parameters=parameters)
    bound=case.bind(candidate)[identifier]
    assert not bound.cached_affine
    vector=jnp.asarray([p.value for p in parameters])
    value,gradient=jax.block_until_ready(bound.loss_and_grad(vector))
    assert np.isfinite(float(value)) and np.isfinite(gradient).all()
    forward=lambda v:-bound(dict(zip((p.name for p in parameters),v))).reward
    step=1e-4;errors=[]
    for index in (0,4):
        direction=jnp.eye(8)[index]
        finite=(forward(vector+step*direction)-forward(vector-step*direction))/(2*step)
        errors.append(float(jnp.abs(gradient[index]-finite)))
        np.testing.assert_allclose(gradient[index],finite,atol=2e-5,rtol=2e-4)
    result=bound.diagnose_result(bound(candidate.initial))
    assert bool(result.valid) and np.isfinite(float(result.density_l1))
    assert case.orbital_reference.method=='PBE@fixed-target-density'
    write_json(root/'validation'/f'open-{identifier}.json',{'passed':True,'system_id':identifier,
        'nao':case.target.density.shape[0],'grid_points':case.grid.ngrids,'source':source_digest(),
        'device':jax.devices()[0].device_kind,'versions':versions(),'job_id':os.environ['SLURM_JOB_ID'],
        'reference_method':case.orbital_reference.method,'parameters':8,'nonlinear':True,
        'energy_labels_used':False,'reward':float(result.reward),'density_l1':float(result.density_l1),
        'density_l2':float(result.density_error),'gradient_fd_max_abs_error':max(errors),
        'seconds':time.perf_counter()-start,'cpu_affinity':sorted(os.sched_getaffinity(0)),**memory()})
