"""Complete-case scoring and matched-size bubbles; no electronic calculations."""
from pathlib import Path
import csv
import hashlib
import json
import os
import numpy as np
from protocols import balanced_mae

ROOT=Path(__file__).resolve().parent
FACTOR=627.5094740631


def csv_file(path,rows):
    with path.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    gate=json.loads((ROOT/'export-gate.json').read_text());assert gate['status']=='passed'
    engine_sha=hashlib.sha256((ROOT/'run_gpu_engine.py').read_bytes()+(ROOT/'protocols.py').read_bytes()).hexdigest()
    settings=json.loads((ROOT/'inputs/settings.json').read_text())
    structures=json.loads((ROOT/'inputs/structures.json').read_text())
    observables=json.loads((ROOT/'inputs/observables.json').read_text())
    files={f.stem:json.loads(f.read_text()) for f in (ROOT/'frozen').glob('*.json')}
    data={};missing=[]
    for key,basis in files.items():
        for method in settings['methods']:
            found={}
            for f in (ROOT/'results'/key/method).glob('*.json'):
                r=json.loads(f.read_text())
                if (r.get('driver_sha256')==engine_sha and r['status']=='passed'
                    and r['variant']=='physical' and r['grid_level']==3 and r.get('active_nuclei') is None):
                    found[r['structure_id']]=r
            for s in structures:
                sid=s['structure_id']
                if sid not in found:missing.append((key,method,sid))
                else:data[key,method,sid]=found[sid]
    if missing:
        (ROOT/'missing-results.json').write_text(json.dumps(missing,indent=2)+'\n')
        raise RuntimeError(f'{len(missing)} missing physical cases; no partial-set scatter will be published')
    scores=[];errors=[]
    for key,basis in files.items():
        if key=='reference':continue
        counts=[r['counts'] for r in basis['records']]
        label=basis.get('name',key.replace('nnao_','NNAO-').replace('_','-').upper())
        for method in settings['methods']:
            categories={'isomer':[],'dimer':[]};basis_errors={'isomer':[],'dimer':[]}
            for o in observables:
                predicted=FACTOR*sum(t['coefficient']*data[key,method,t['structure_id']]['energy_hartree'] for t in o['terms'])
                same_method=FACTOR*sum(t['coefficient']*data['reference',method,t['structure_id']]['energy_hartree'] for t in o['terms'])
                error=predicted-o['reference_energy_kcal_mol']
                categories[o['category']].append(error)
                basis_errors[o['category']].append(predicted-same_method)
                row=dict(basis_key=key,label=label,method=method,observable=o['reaction_id'],category=o['category'],
                    predicted_kcal_mol=predicted,reference_kcal_mol=o['reference_energy_kcal_mol'],
                    signed_error_kcal_mol=error,same_method_augccpvqz_kcal_mol=same_method,
                    same_method_basis_error_kcal_mol=predicted-same_method)
                errors.append(row)
            electron_error=np.array([abs(data[key,method,s['structure_id']]['energy_hartree']-
                data['reference',method,s['structure_id']]['energy_hartree'])/s['nelectron']*1000 for s in structures])
            records=[data[key,method,s['structure_id']] for s in structures]
            scores.append(dict(basis_key=key,label=label,method=method,kind=basis['kind'],samples=30,
                mean_ao=float(np.mean([r['nao_spherical'] for r in counts])),
                mean_primitive=float(np.mean([r['primitive_spherical_unique'] for r in counts])),
                balanced_mae_kcal_mol=balanced_mae(categories['isomer'],categories['dimer']),
                isomer_mae_kcal_mol=float(np.mean(abs(np.array(categories['isomer'])))),
                dimer_mae_kcal_mol=float(np.mean(abs(np.array(categories['dimer'])))),
                same_method_relative_mae_kcal_mol=balanced_mae(basis_errors['isomer'],basis_errors['dimer']),
                ground_state_mae_mEh_per_electron=float(np.mean(electron_error)),
                physical_case_seconds=float(sum(r['seconds'] for r in records)),
                devices=';'.join(sorted({r['device'] for r in records}))))
    assert len(scores)==54 and len(errors)==648
    (ROOT/'analysis').mkdir(exist_ok=True)
    csv_file(ROOT/'analysis/summary.csv',scores)
    csv_file(ROOT/'analysis/observable-errors.csv',errors)
    (ROOT/'analysis/summary.json').write_text(json.dumps(scores,indent=2)+'\n')
    plot(scores)
    (ROOT/'analysis/verification.json').write_text(json.dumps(dict(status='passed',
        physical_method_energy_records=len(data),summary_rows=len(scores),observable_rows=len(errors),
        scoring='All 30 structures and all 12 observables required per basis/method',
        cp_bsse_included=False,numerical_engine_sha256=engine_sha,
        cost_note='AO/primitive scale proxies; no cross-device wall-time ranking'),indent=2)+'\n')


def plot(scores):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'plot-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    palette={'STO':'#77838D','Pople':'#3D7EAA','def2':'#71813A','cc':'#795C9E','NNAO':'#D46831'}
    def group(row):
        if row['kind']=='nnao':return 'NNAO'
        label=row['label'].lower()
        if label.startswith('sto'):return 'STO'
        if label.startswith('def2'):return 'def2'
        if 'cc-pv' in label:return 'cc'
        return 'Pople'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,
        'xtick.direction':'in','ytick.direction':'in'})
    (ROOT/'figures').mkdir(exist_ok=True)
    for metric,stem,ylabel in [
        ('balanced_mae_kcal_mol','accuracy-vs-ao','Balanced isomer / dimer MAE (kcal/mol)'),
        ('same_method_relative_mae_kcal_mol','basis-error-vs-ao','Relative-energy MAE vs same method / aug-cc-pVQZ (kcal/mol)'),
        ('ground_state_mae_mEh_per_electron','ground-state-error-vs-ao','Mean absolute total-energy difference (mEh/electron)')]:
        fig,axes=plt.subplots(1,2,figsize=(12.6,6.1),sharey=True,layout='constrained')
        for ax,method,title in zip(axes,['mp2','wb97m-v'],['Frozen-core DF-MP2',r'$\omega$B97M-V, grid level 3']):
            rows=[r for r in scores if r['method']==method]
            for index,r in enumerate(rows):
                size=r['mean_primitive']*.6
                ax.scatter(r['mean_ao'],r[metric],s=size,c=palette[group(r)],alpha=.78,
                           edgecolors='#333333' if r['kind']=='nnao' else 'white',linewidths=.7)
                short=r['label'].replace('NNAO-','').replace('nnao_','').replace('def2-','d2-')
                ax.annotate(short,(r['mean_ao'],r[metric]),xytext=(4,4 if index%2 else -8),
                    textcoords='offset points',fontsize=6.5)
            ax.set_yscale('log');ax.set_xlabel('Mean contracted spherical AOs / structure')
            ax.set_title(title);ax.grid(color='#E4E8EB',lw=.55);ax.set_axisbelow(True)
        axes[0].set_ylabel(ylabel)
        handles=[Line2D([],[],marker='o',linestyle='',color=c,label=g,markersize=7) for g,c in palette.items()]
        legend=axes[1].legend(handles=handles,frameon=False,fontsize=8,loc='upper right')
        axes[1].add_artist(legend)
        sizes=sorted(set(round(float(v)/10)*10 for v in np.quantile([r['mean_primitive'] for r in scores],[.1,.5,.9])))
        bubbles=[axes[1].scatter([],[],s=.6*v,c='#B8BEC4',edgecolors='white',label=str(int(v))) for v in sizes]
        axes[1].legend(handles=bubbles,frameon=False,fontsize=8,loc='lower left',title='Mean primitive AOs',title_fontsize=8)
        fig.suptitle('27 frozen bases; identical 30-structure benchmark; bubble area proportional to primitive AOs',fontsize=11)
        for extension in ['png','pdf','svg']:fig.savefig(ROOT/'figures'/f'{stem}.{extension}',dpi=220)
        plt.close(fig)


if __name__=='__main__':main()
