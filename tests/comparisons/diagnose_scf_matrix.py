"""Cross-evaluate converged mismatches; preserve original matrix outcomes."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import jax.numpy as jnp
from ten_system_scf_matrix import SYSTEMS, prepare, run_gradscf, json_safe, XC_REFERENCE


def diagnose(root, systems):
    from pyscf import scf,dft
    output=[]
    for name in systems:
        path=root/name/'results.jsonl'
        if not path.exists():continue
        rows=[json.loads(l) for l in path.read_text().splitlines()]
        rows=[r for r in rows if r['status']=='energy_mismatch' and r.get('gradscf',{}).get('converged') and r.get('pyscf',{}).get('converged')]
        if not rows:continue
        meta=json.loads((root/name/'metadata.json').read_text())
        p=prepare(SYSTEMS[name],meta['basis'],meta['grid_level'])
        for row in rows:
            method,xc=row['method'],row['xc']
            data=np.load(root/name/(row['case']+'.npz'))
            dg,dr=data['density_gradscf'],data['density_pyscf']
            record=dict(system=name,case=row['case'],original_energy_error=row['abs_energy_error'])
            mf=getattr(scf if xc=='hf' else dft,method.replace('_ncol',''))(p['mol'])
            mf._eri=p['eri_reference'];mf.direct_scf=False
            if xc!='hf':
                mf.xc=XC_REFERENCE[xc];mf.grids.coords=p['coords'];mf.grids.weights=p['weights'];mf.small_rho_cutoff=0.
                if method.startswith('GKS'):mf.collinear='ncol' if method.endswith('ncol') else 'col'
            ecross=float(mf.energy_tot(dm=dg))
            record['pyscf_energy_at_gradscf_density']=ecross
            record['fixed_density_energy_error']=abs(ecross-row['gradscf']['energy'])
            metric=mf.get_ovlp();v=mf.get_veff(dm=dg);f=mf.get_fock(dm=dg,vhf=v)
            if method in {'ROHF','ROKS'}:
                record['reference_fock_commutator_note']='Common-orbital ROHF uses a different orbital-gradient condition'
            else:
                record['reference_fock_commutator']=float(np.max(np.abs(f@dg@metric-metric@dg@f)))
            if method in {'UHF','UKS','ROHF','ROKS'}:
                restart=dict(p,da=dr[0],db=dr[1])
                s=np.asarray(p['s']);na,nb=p['na'],p['nb']
                record['gradscf_s_squared']=(na-nb)**2/4+(na+nb)/2-float(np.trace(dg[0]@s@dg[1]@s).real)
                record['pyscf_s_squared']=(na-nb)**2/4+(na+nb)/2-float(np.trace(dr[0]@s@dr[1]@s).real)
            elif method in {'GHF','GKS','GKS_ncol'}:restart=dict(p,dspin=dr)
            else:
                record['restart_note']='RHF/RKS restricted restart not implemented in this diagnostic';output.append(record);continue
            try:
                start=time.perf_counter();new,_=run_gradscf(method,xc,restart,200)
                record['restart_from_pyscf_density']=new
                record['restart_energy_error']=abs(new['energy']-row['pyscf']['energy'])
                record['restart_seconds']=time.perf_counter()-start
            except Exception as e:record['restart_error']=str(e)
            output.append(record)
            (root/'mismatch-diagnostics.json').write_text(json.dumps(json_safe(output),indent=2)+'\n')
            print(json.dumps(json_safe(record)),flush=True)
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--systems',nargs='+',default=list(SYSTEMS));a=p.parse_args();diagnose(a.root,a.systems)
