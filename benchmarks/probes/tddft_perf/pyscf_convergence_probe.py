"""Explain reference convergence flags without relaxing any threshold."""
import json
from pathlib import Path
import sys
import numpy as np
from worker import Recorder,reference

cfg=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
log=Recorder(cfg,out);mf,_=reference(cfg,log)
for space in (None,40):
    td=mf.TDDFT() if cfg['full'] else mf.TDA()
    td.nstates=5;td.conv_tol=1e-8;td.max_cycle=100
    if space is not None:td.max_space=space
    original=td.gen_vind;calls=[]
    def instrument(*args,**kwargs):
        vind,diag=original(*args,**kwargs)
        def wrapped(values):
            calls.append(list(np.asarray(values).shape));return vind(values)
        return wrapped,diag
    td.gen_vind=instrument
    log.timed('reference_solver_diagnostic',td.kernel,notes='diagnostic call counter; not clean timing')
    vind,diag=original()
    if cfg['full']:
        vectors=np.array([np.concatenate((x.reshape(-1),y.reshape(-1))) for x,y in td.xy])*np.sqrt(2.)
    else:
        vectors=np.array([x.reshape(-1) for x,y in td.xy])*np.sqrt(2.)
    residual=np.linalg.norm(vind(vectors)-np.asarray(td.e)[:,None]*vectors,axis=1)
    log.add('reference_residual',None,reported_converged=list(map(bool,td.converged)),
            actual_residuals=residual,actual_max_space=getattr(td,'max_space',None),operator_calls=len(calls),
            operator_shapes=calls,excitations_ha=td.e,original_operator_residual=float(max(residual)),
            notes='same 1e-8 threshold, 100 iterations; only subspace cap changed for diagnosis')
