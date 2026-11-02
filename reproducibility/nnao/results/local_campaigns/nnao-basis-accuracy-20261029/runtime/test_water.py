"""Official-stack water DF-MP2 and omegaB97M-V GPU/CPU comparison."""
from pathlib import Path
import contextlib
import faulthandler
import io
import json
import os
import time
import importlib.metadata as metadata

import numpy as np
import cupy as cp
from pyscf import gto,scf,dft,mp,df,lib
from gpu4pyscf.scf import hf
from gpu4pyscf.dft import rks
from gpu4pyscf.mp import dfmp2
from gpu4pyscf.lib import cutensor

ROOT=Path(__file__).resolve().parent.parent
BASIS_KEY=os.environ.get('WATER_BASIS_KEY','t14')
faulthandler.dump_traceback_later(120,repeat=True)
lib.num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','8')))


def main():
    start=time.perf_counter();out=ROOT/'runtime'
    report={'status':'running','scope':'water minimal validation only','basis_key':BASIS_KEY,'results':[]}
    def save():
        (out/f'water-validation-{BASIS_KEY}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    assert cp.cuda.runtime.getDeviceCount()==1
    config=io.StringIO()
    with contextlib.redirect_stdout(config):cp.show_config()
    (out/'cuda-config-official.txt').write_text(config.getvalue())
    tick=time.perf_counter();np.testing.assert_array_equal(cp.arange(4).get(),np.arange(4))
    assert cutensor.cutensor is not None and cutensor.contract_engine is None,'cuTENSOR not enabled'
    a=cp.arange(6,dtype=cp.float64).reshape(2,3);b=cp.arange(12,dtype=cp.float64).reshape(3,4)
    np.testing.assert_allclose(cutensor.contract('ij,jk->ik',a,b).get(),a.get()@b.get(),rtol=1e-13,atol=1e-13)
    report.update(cupy_cutensor_seconds=time.perf_counter()-tick,cutensor_enabled=True,
        device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        versions={n:metadata.version(n) for n in ['pyscf','gpu4pyscf-cuda12x','cupy-cuda12x','cutensor-cu12','numpy','scipy']})
    save()
    rows={r['structure_id']:r for r in json.loads((ROOT/'inputs/structures.json').read_text())}
    row=rows['S22/02/A'];basis=json.loads((ROOT/'frozen'/f'{BASIS_KEY}.json').read_text())
    shells=next(r['atom_shells'] for r in basis['records'] if r['structure_id']==row['structure_id'])
    aux=json.loads((ROOT/'inputs/auxiliary-bases.json').read_text())['bases']
    labels=[s+str(i) for i,s in enumerate(row['symbols'])]
    mol=gto.M(atom=list(zip(labels,row['coords_angstrom'])),basis=dict(zip(labels,shells)),
        unit='Angstrom',cart=False,charge=0,spin=0,verbose=3,max_memory=40000)
    assert mol.nelectron==10 and mol.nao_nr()=={'t14':24,'nnao_szp_s':18}[BASIS_KEY]
    for method in ['mp2','wb97m-v']:
        record={'method':method,'basis':basis.get('name',BASIS_KEY),'nao':mol.nao_nr(),'grid_level':3,'status':'running'}
        try:
            tick=time.perf_counter()
            gpu=(hf.RHF(mol) if method=='mp2' else rks.RKS(mol,xc='wb97m-v')).density_fit(auxbasis=aux['def2-universal-jkfit'])
            gpu.conv_tol=1e-11;gpu.conv_tol_grad=1e-8;gpu.max_cycle=200;gpu.level_shift=0.;gpu.chkfile=None
            if method!='mp2':
                gpu.grids.level=3;gpu.nlcgrids.level=3;gpu.small_rho_cutoff=0.;assert gpu.do_nlc()
            eg=float(gpu.kernel());cp.cuda.get_current_stream().synchronize();assert gpu.converged
            record.update(gpu_scf_energy=eg,gpu_class=type(gpu).__module__)
            assert record['gpu_class'].startswith('gpu4pyscf')
            if method=='mp2':
                pt=dfmp2.DFMP2(gpu,frozen=1,auxbasis=aux['aug-cc-pvqz-ri'])
                pt.with_t2=False;pt.fp_type='FP64';pt.kernel(with_t2=False)
                cp.cuda.get_current_stream().synchronize();assert pt.t2 is None
                record.update(gpu_correlation_energy=float(pt.e_corr),mp2_class=type(pt).__module__,frozen_core=1)
                assert record['mp2_class'].startswith('gpu4pyscf')
                eg+=float(pt.e_corr)
            else:
                record.update(nlc_enabled=bool(gpu.do_nlc()),nlc_grid_points=len(gpu.nlcgrids.weights))
                assert record['nlc_enabled'] and record['nlc_grid_points']>0
            record['gpu_seconds']=time.perf_counter()-tick
            cpu=(scf.RHF(mol) if method=='mp2' else dft.RKS(mol,xc='wb97m-v')).density_fit(auxbasis=aux['def2-universal-jkfit'])
            cpu.conv_tol=1e-11;cpu.conv_tol_grad=1e-8;cpu.max_cycle=200;cpu.chkfile=None
            if method!='mp2':
                cpu.grids.level=3;cpu.nlcgrids.level=3;cpu.small_rho_cutoff=0.;assert cpu.do_nlc()
            ec=float(cpu.kernel());assert cpu.converged
            if method=='mp2':
                fitting=df.DF(mol,auxbasis=aux['aug-cc-pvqz-ri'])
                ref=mp.MP2(cpu,frozen=1).density_fit(auxbasis=aux['aug-cc-pvqz-ri'],with_df=fitting)
                assert ref.with_df._cderi is None
                ref.kernel(with_t2=False);ec+=float(ref.e_corr)
                record.update(gpu_mp2_auxiliary_count=pt.auxmol.nao_nr(),
                              cpu_mp2_auxiliary_count=ref.with_df.get_naoaux())
                assert record['gpu_mp2_auxiliary_count']==record['cpu_mp2_auxiliary_count']
            error=abs(eg-ec);tolerance=1e-7 if method=='mp2' else 1e-6
            record.update(gpu_energy_hartree=eg,cpu_energy_hartree=ec,error_hartree=error,tolerance_hartree=tolerance)
            assert error<tolerance,(method,error,tolerance)
            record['status']='passed'
        except Exception as error:
            record.update(status='failed',error=str(error))
        report['results'].append(record);save();print('WATER_RESULT',record,flush=True)
    report.update(status='passed' if all(r['status']=='passed' for r in report['results']) else 'failed',
                  elapsed_seconds=time.perf_counter()-start)
    save();faulthandler.cancel_dump_traceback_later()
    if report['status']!='passed':raise RuntimeError('Water validation failed; benchmark remains held')


if __name__=='__main__':main()
