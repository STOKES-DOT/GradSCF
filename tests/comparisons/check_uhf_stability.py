"""Native-integral UHF stability/restart comparison; PySCF is reference only."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from ten_system_scf_matrix import SYSTEMS, prepare
from gradscf import scf
from pyscf import scf as pyscf_scf


def run(name):
    start=time.perf_counter()
    p=prepare(SYSTEMS[name],'3-21g',1)
    args=dict(overlap=p['s'],hcore=p['h'],eri=p['eri'],nalpha=p['na'],nbeta=p['nb'],
              nuclear_repulsion=p['enuc'],init_density_alpha=p['da'],init_density_beta=p['db'],
              config=scf.UHFConfig(max_cycle=200,conv_tol=1e-11,conv_tol_density=1e-9))
    outcome=scf.stabilize_uhf_from_integrals(**args)
    ref=pyscf_scf.UHF(p['mol'])
    ref.conv_tol=1e-11
    ref.conv_tol_grad=1e-7
    ref.max_cycle=200
    ref.kernel(dm0=np.stack([p['da'],p['db']]))
    _,_,ref_stable,_=ref.stability(return_status=True)
    error=abs(outcome.result.total_energy-ref.e_tot)
    row=dict(system=name,energy=outcome.result.total_energy,reference_energy=ref.e_tot,
             energy_error=error,converged=outcome.result.converged,stable=outcome.stable,
             reference_converged=bool(ref.converged),reference_stable=bool(ref_stable),
             minimum_curvature=outcome.minimum_curvature,
             eigenpair_residuals=np.asarray(outcome.stability.residual_norms).tolist(),
             restarts=outcome.restarts,energy_history=outcome.energy_history,
             input_errors=p['input_errors'],elapsed_seconds=time.perf_counter()-start)
    row['passed']=bool(row['converged'] and row['stable'] and ref.converged and ref_stable and error<1e-8)
    print(json.dumps(row),flush=True)
    return row


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--systems',nargs='+',choices=SYSTEMS,default=['OH'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    start=time.perf_counter()
    rows=[run(name) for name in args.systems]
    root=Path(__file__).resolve().parents[2]
    metadata=dict(command=sys.argv,hostname=platform.node(),platform=platform.platform(),
                  versions={k:importlib.metadata.version(k) for k in ['jax','pyscf','numpy']},
                  backend='cpu',dtype='float64',basis='3-21g',energy_tolerance=1e-8,
                  stability_tolerance=1e-5,eigensolver_tolerance=1e-7,
                  affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
                  hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (root/'src/gradscf/scf').glob('*.py')})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(metadata=metadata,rows=rows,
        elapsed_seconds=time.perf_counter()-start,passed=sum(r['passed'] for r in rows)),indent=2)+'\n')
    sys.exit(0 if all(r['passed'] for r in rows) else 1)
