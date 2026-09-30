"""Opt-in molecular MolGW/GradSCF comparison; never downloads or installs software.

Usage: PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 python this_file.py
       /path/to/molgw --workdir /tmp/molgw-comparison --output report.json

Set the executable's runtime library environment before calling this script.
MolGW source: b831818d7a845c36f295d036dc8ceef59daf9991 (3.4).
HF/STO-3G; full ERIs for six cases and matched Weigend RI for the
independent virtual-window case. Atomic units unless explicitly marked.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import subprocess
import time

import jax
import numpy as np
import yaml

jax.config.update('jax_enable_x64', True)
from gradscf import bse, gto, gw, scf

MOLGW_REVISION = 'b831818d7a845c36f295d036dc8ceef59daf9991'
MOLGW_HA_EV = 27.21138505
CASES = (
    ('h2_g0w0', 'H 0 0 0; H 0 0 .74', 'g0w0', 0, 0, None),
    ('h2_evgw0', 'H 0 0 0; H 0 0 .74', 'evgw0', 0, 0, None),
    ('h2_evgw', 'H 0 0 0; H 0 0 .74', 'evgw', 0, 0, None),
    ('water_g0w0', 'O 0 0 0; H 0 -.757 .587; H 0 .757 .587', 'g0w0', 0, 0, None),
    ('water_w_core', 'O 0 0 0; H 0 -.757 .587; H 0 .757 .587', 'g0w0', 0, 1, None),
    ('water_gw_core', 'O 0 0 0; H 0 -.757 .587; H 0 .757 .587', 'g0w0', 1, 1, None),
    ('water_w_virtual_ri', 'O 0 0 0; H 0 -.757 .587; H 0 .757 .587', 'g0w0', 0, 0, 6),
)


def execute(executable, folder, name, text):
    (folder / (name+'.in')).write_text(text)
    with (folder / (name+'.out')).open('w') as stream:
        subprocess.run([str(executable),name+'.in'],cwd=folder,check=True,
                       stdout=stream,stderr=subprocess.STDOUT)
    result=yaml.safe_load((folder/'molgw.yaml').read_text())
    (folder/(name+'.yaml')).write_text((folder/'molgw.yaml').read_text())
    return result


def ordered(mapping):
    return np.array([mapping[k] for k in sorted(k for k in mapping if isinstance(k,int))])


def compare_case(executable, folder, case):
    name, atom, method, gcore, wcore, wend = case
    folder.mkdir(parents=True,exist_ok=True)
    # Refuse to silently reuse a cached W or an incompatible SCF restart.
    if any(folder.iterdir()):
        raise FileExistsError(f'Use a fresh comparison directory: {folder}')
    start=time.perf_counter()
    auxbasis = 'weigend' if name.endswith('_ri') else None
    mf=scf.RHF(gto.M(atom=atom,basis='sto-3g'),conv_tol=1e-12)
    if auxbasis is not None:
        mf.density_fit(auxbasis)
        mf.df_tol = 1e-6  # MolGW's absolute Coulomb-metric eigenvalue cutoff
    mf.run()
    if not mf.converged: raise ArithmeticError('GradSCF HF did not converge')
    nmo=len(mf.mo_energy);nocc=int(np.count_nonzero(mf.mo_occ))
    wend=nmo if wend is None else wend
    # MolGW restricts QP targets along with the G core cutoff.
    # The independent W-virtual cutoff uses RI: MolGW's no-RI route also
    # truncates its W product representation, so it is not the same model.
    gend = nmo
    targets=tuple(range(gcore,gend))
    gi=tuple(range(gcore,gend)); oi=tuple(range(wcore,nocc)); va=tuple(range(nocc,wend))
    lines=atom.split(';')
    common=f"""&molgw
 scf='HF'
 basis='STO-3G'
 gaussian_type='CART'
 tolscf=1.0e-12
 nscf=100
 frozencore='no'
 ncoreg={gcore}
 ncorew={wcore}
 nvirtualg={gend+1}
 nvirtualw={wend+1}
 selfenergy_state_min={gcore+1}
 selfenergy_state_max={nmo}
 nomega_sigma=40001
 step_sigma=0.00005
 length_unit='angstrom'
 natom={len(lines)}
"""
    if auxbasis is not None: common += " auxil_basis='weigend'\n"
    geometry='/\n'+'\n'.join(x.strip() for x in lines)+'\n'
    keyword={'g0w0':'G0W0','evgw0':'GnW0','evgw':'evGW'}[method]
    steps=1 if method == 'g0w0' else 40
    print_w='yes' if auxbasis is None else 'no'
    gw_input=common+f" postscf='{keyword}'\n nstep_gw={steps}\n eta=0.00001\n print_w='{print_w}'\n"+geometry
    upstream=execute(executable,folder,'gw',gw_input)
    if not upstream.get('scf is converged',False):raise ArithmeticError('MolGW SCF did not converge')
    qp_reference=np.loadtxt(folder/'ENERGY_QP',skiprows=2)[:,1]
    own=gw.GW(mf,method=method,nw=200,eta=1e-5,max_cycle=100,conv_tol=1e-10,
              g_orbitals=gi,screening_occupied=oi,screening_virtual=va).run(orbs=targets)
    if not own.converged:raise ArithmeticError('GradSCF GW did not converge')
    own_qp=np.asarray(own.mo_energy)
    target=np.asarray(targets)
    errors={'hf_hartree':abs(float(mf.e_tot)-upstream['scf energy']['total']),
            'qp_hartree':float(np.max(np.abs(own_qp[target]-qp_reference[target])))}
    datum=dict(name=name,atom=atom,basis='sto-3g',auxbasis=auxbasis,
               hf_hartree=upstream['scf energy']['total'],method=method,g_orbitals=gi,
               screening_occupied=oi,screening_virtual=va,qp_targets=targets,
               molgw_input=gw_input,qp_hartree=qp_reference.tolist(),
               gradscf_qp_hartree=own_qp.tolist(),
               gradscf_qp_residual_max=float(np.max(np.abs(own.result.qp_residual))),
               gradscf_qp_weight=np.asarray(own.result.qp_weight)[target].tolist(),
               errors=errors,optical=[])
    # Nonlinear MolGW iterations do not emit the same weight section.
    se=upstream.get('gw selfenergy',{})
    if 'renormalization factor' in se:
        weight=ordered(se['renormalization factor']['spin channel 1'])
        datum['qp_weight']=weight.tolist()
        errors['qp_weight']=float(np.max(np.abs(np.asarray(own.result.qp_weight)[target]-weight)))
    for tda,singlet in ((True,True),(False,True),(False,False)):
        label=('tda' if tda else 'full')+('_singlet' if singlet else '_triplet')
        text=common+f" postscf='BSE'\n eta=0.01\n tda='{'yes' if tda else 'no'}'\n triplet='{'no' if singlet else 'yes'}'\n read_restart='yes'\n"+geometry
        external=execute(executable,folder,label,text)['optical spectrum']['excitations']
        energies=ordered(external['energies'])/MOLGW_HA_EV
        strengths=ordered(external['oscillator strengths'])
        response=bse.BSE(own,occupied=oi,virtual=va,nroots=len(oi)*len(va),
                         tda=tda,singlet=singlet,solver='dense').run()
        if not response.converged.all():raise ArithmeticError('BSE did not converge')
        err=dict(energy_hartree=float(np.max(np.abs(response.e-energies))),
                 oscillator_strength=float(np.max(np.abs(response.oscillator_strength()-strengths))))
        out=dict(tda=tda,singlet=singlet,energies_hartree=energies.tolist(),
                 oscillator_strengths=strengths.tolist(),errors=err)
        if singlet:
            static=np.array(external['static polarizability']).reshape(3,3)
            err['static_polarizability_au']=float(np.max(np.abs(response.polarizability()-static)))
            # Sample MolGW's actual output grid, including its maximum intensity.
            data=np.loadtxt(folder/'photoabsorption_cross_section.dat')
            indices=np.unique(np.r_[np.linspace(0,len(data)-1,17,dtype=int),np.argmax(data[:,1])])
            frequency=data[indices,0]/MOLGW_HA_EV
            cross=np.asarray(response.absorption_cross_section(frequency,eta=.01))
            err['cross_section_au']=float(np.max(np.abs(cross-data[indices,1])))
            out.update(static_polarizability_au=static.tolist(),omega_hartree=frequency.tolist(),
                       cross_section_au=data[indices,1].tolist())
        datum['optical'].append(out)
    datum['seconds']=time.perf_counter()-start
    return datum


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable',type=Path)
    parser.add_argument('--source',type=Path,help='Pinned MolGW source; default executable parent')
    parser.add_argument('--workdir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',action='append',choices=[c[0] for c in CASES])
    args=parser.parse_args()
    exe=args.executable.resolve()
    source=(args.source or exe.parent).resolve()
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    if revision != MOLGW_REVISION:
        raise ValueError(f'Expected MolGW {MOLGW_REVISION}, got {revision}')
    if subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=source,text=True).strip():
        raise ValueError('Reference MolGW tracked sources must be unmodified')
    hashes={f:hashlib.sha256((source/f).read_bytes()).hexdigest() for f in
            ('src/m_selfenergy_tools.f90','src/m_selfenergy_evaluation.f90',
             'src/m_linear_response.f90','src/m_spectral_function.f90','src/m_spectra.f90')}
    report=dict(molgw_revision=revision,source_sha256=hashes,executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                backend=jax.default_backend(),jax=jax.__version__,platform=platform.platform(),
                python=platform.python_version(),dtype='float64',molgw_ha_ev=MOLGW_HA_EV,
                settings=dict(nw=200,gw_eta_hartree=1e-5,optical_eta_hartree=.01,
                              molgw_sigma_points=40001,molgw_sigma_step_hartree=.00005),cases=[])
    for case in CASES:
        if args.case and case[0] not in args.case:continue
        record=compare_case(exe,args.workdir/case[0],case)
        report['cases'].append(record)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(case[0],record['errors'],[x['errors'] for x in record['optical']],flush=True)
    # Conservative limits fixed before measuring the seven-case comparison.
    for c in report['cases']:
        assert c['errors']['hf_hartree']<1e-7,(c['name'],c['errors'])
        assert c['errors']['qp_hartree']<2e-6,(c['name'],c['errors'])
        assert c['errors'].get('qp_weight',0)<5e-5,(c['name'],c['errors'])
        for o in c['optical']:
            assert o['errors']['energy_hartree']<3e-6,o
            assert o['errors']['oscillator_strength']<2e-5,o
            assert o['errors'].get('static_polarizability_au',0)<3e-4,o
            assert o['errors'].get('cross_section_au',0)<2e-4,o


if __name__=='__main__':main()
