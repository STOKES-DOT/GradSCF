"""Explicit seeded SCF restarts; select lowest converged candidate, not a proven ground state."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from ten_system_scf_matrix import SYSTEMS, prepare, run_gradscf, run_pyscf, json_safe
from gradscf.scf.init_guess import orbital_rotation_guesses


def run(args):
    p=prepare(SYSTEMS[args.system],args.basis,args.grid_level)
    guesses=orbital_rotation_guesses(p['c'],amplitudes=args.amplitudes,seed=args.seed)
    rows=[]
    args.output.mkdir(parents=True,exist_ok=True)
    for index,(amplitude,c) in enumerate(zip(args.amplitudes,guesses)):
        da=c[:,:p['na']]@c[:,:p['na']].T;db=c[:,:p['nb']]@c[:,:p['nb']].T;z=np.zeros_like(da)
        trial=dict(p,c=c,da=da,db=db,dspin=np.block([[da,z],[z,db]]).astype(complex))
        start=time.perf_counter()
        g,dg=run_gradscf(args.method,args.xc,trial,args.max_cycle,level_shift=args.level_shift)
        r,dr=run_pyscf(args.method,args.xc,trial,args.max_cycle,args.grid_level,
                      level_shift=args.level_shift,final_cycle=not args.preserve_reference_root)
        rows.append(dict(index=index,amplitude=amplitude,gradscf=g,pyscf=r,elapsed_seconds=time.perf_counter()-start))
        np.savez_compressed(args.output/f'attempt-{index}.npz',initial_density=np.stack([da,db]),
                            density_gradscf=dg,density_pyscf=dr,overlap=np.asarray(p['s']))
        print(json.dumps(json_safe(rows[-1])),flush=True)
    selected={}
    for engine in ['gradscf','pyscf']:
        valid=[row for row in rows if row[engine]['converged'] and np.isfinite(row[engine]['energy'])]
        selected[engine]=min(valid,key=lambda row:row[engine]['energy'])['index'] if valid else None
    error=None
    if all(index is not None for index in selected.values()):
        error=abs(rows[selected['gradscf']]['gradscf']['energy']-rows[selected['pyscf']]['pyscf']['energy'])
    summary=dict(system=args.system,method=args.method,xc=args.xc,basis=args.basis,grid_level=args.grid_level,
        seed=args.seed,amplitudes=args.amplitudes,level_shift=args.level_shift,max_cycle=args.max_cycle,
        reference_extra_cycle=not args.preserve_reference_root,attempts=rows,selected=selected,energy_error=error,
        pass_comparison=error is not None and error<(1e-8 if args.xc=='hf' else 1e-6),
        selection='lowest converged candidate among explicit guesses; no global stability guarantee')
    (args.output/'summary.json').write_text(json.dumps(json_safe(summary),indent=2)+'\n')
    print('Selected:',selected,'energy error:',error,flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--system',choices=SYSTEMS,default='OH');p.add_argument('--method',default='UHF')
    p.add_argument('--xc',default='hf');p.add_argument('--basis',default='3-21g');p.add_argument('--grid-level',type=int,default=1)
    p.add_argument('--seed',type=int,default=20260913);p.add_argument('--amplitudes',nargs='+',type=float,default=[0.,.1])
    p.add_argument('--level-shift',type=float,default=0.);p.add_argument('--max-cycle',type=int,default=200)
    p.add_argument('--preserve-reference-root',action='store_true');p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
