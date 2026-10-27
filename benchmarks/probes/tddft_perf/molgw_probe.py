"""Independent stock MOLGW TDDFT capability and grid-convergence check.

Only basis DATA are converted from the bundled common baseline. No GradSCF
orbitals, integrals, restart, QP file, or modified comparison executable enters
MOLGW. Different quadratures prohibit an automatic strict speed ratio.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import yaml
from worker import Recorder


def main():
    cfg=json.loads(Path(sys.argv[1]).read_text()); out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=True)
    log=Recorder(cfg,out)
    exe=Path(cfg['executable']).expanduser().resolve()
    basis=out/'basis';basis.mkdir(exist_ok=True)
    from gradscf.integrals.basis.data import load_basis_from_snapshot
    for symbol in ('H','O'):
        shells=load_basis_from_snapshot('def2-svp',symbol)
        lines=[];count=0
        for shell in shells:
            angular=int(shell[0]);primitives=shell[1:]
            for column in range(1,len(primitives[0])):
                count+=1;lines.append(f'{len(primitives)} {angular}')
                lines.extend(f'{row[0]:.16e} {row[column]:.16e}' for row in primitives)
        (basis/f'{symbol}_probe-def2-svp').write_text(str(count)+'\n'+'\n'.join(lines)+'\n')
    env={k:v for k,v in os.environ.items() if not k.startswith('MOLGW_COMPARE_')}
    env['OMP_STACKSIZE']='256M'
    env['LD_LIBRARY_PATH']=cfg['intel_runtime']+':'+env.get('LD_LIBRARY_PATH','')
    _,hard=resource.getrlimit(resource.RLIMIT_STACK)
    resource.setrlimit(resource.RLIMIT_STACK,(256*1024**2 if hard==resource.RLIM_INFINITY else min(hard,256*1024**2),hard))
    for quality in cfg.get('qualities',('very high','insane')):
        repeats=cfg.get('repeats',4 if quality=='very high' else 1)
        for repeat in range(repeats):
            work=out/(quality.replace(' ','-')+f'-r{repeat}');work.mkdir(exist_ok=True)
            text=f'''&molgw
 scf='PBE'
 postscf='td'
 tda='{'no' if cfg['full'] else 'yes'}'
 gaussian_type='cart'
 basis='probe-def2-svp'
 basis_path='{basis}/'
 read_restart='no'
 print_restart='no'
 frozencore='no'
 init_hamiltonian='core'
 nscf=100
 tolscf=1.0e-9
 nexcitation={cfg.get('molgw_nexcitation',5)}
 toldav=1.0e-8
 grid_quality='{quality}'
 tddft_grid_quality='{quality}'
 natom=3
/
8 0 0 0
1 0 -0.757 0.587
1 0 0.757 0.587
'''
            (work/'input.in').write_text(text)
            start=time.perf_counter()
            with (work/'stdout.log').open('w') as stream:
                done=subprocess.run([str(exe),'input.in'],cwd=work,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=180)
            elapsed=time.perf_counter()-start
            details=dict(status='ok' if done.returncode==0 else 'error',grid_quality=quality,returncode=done.returncode,
                         executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                         notes='independent stock executable; custom basis data only; quadrature differs; no speed ratio')
            if (work/'molgw.yaml').exists():
                data=yaml.safe_load((work/'molgw.yaml').read_text())
                exc=data.get('optical spectrum',{}).get('excitations',{})
                if 'energies' in exc:
                    details['excitations_ev']=[float(exc['energies'][k]) for k in sorted(k for k in exc['energies'] if isinstance(k,int))][:5]
                    if not all(math.isfinite(e) for e in details['excitations_ev']):
                        details['status']='numerical_mismatch'
            log.add('molgw_end_to_end',elapsed,repeat,**details)


if __name__=='__main__':main()
