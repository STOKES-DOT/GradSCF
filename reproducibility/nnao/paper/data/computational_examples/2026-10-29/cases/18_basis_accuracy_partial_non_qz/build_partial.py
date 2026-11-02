"""Reproduce the non-QZ database comparison without electronic calculations."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import re

import numpy as np
from protocols import basis_counts, balanced_mae

ROOT=Path(__file__).resolve().parent
FACTOR=627.5094740631
PALETTE={'STO':'#7C8791','Pople':'#2878A9','def2':'#47856C','cc':'#8763A6','NNAO':'#D05A27'}


def dump_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def label_basis(key,basis):
    if basis['kind']=='nnao':return key.replace('nnao_','NNAO-').replace('_','-').upper()
    label=basis['name'].replace('sto-','STO-').replace('cc-pv','cc-pV')
    if label.startswith('STO-'):label=label.replace('g','G')
    label=label.replace('vdz','vDZ').replace('vtz','vTZ')
    label=label.replace('pVdz','pVDZ').replace('pVtz','pVTZ')
    if label.startswith('def2-'):return 'def2-'+label[5:].upper()
    if label[0].isdigit():label=label.replace('g','G')
    return label


def family(label):
    if label.startswith('NNAO-'):return 'NNAO'
    if label.startswith('STO-'):return 'STO'
    if label.startswith('def2-'):return 'def2'
    if 'cc-pV' in label:return 'cc'
    return 'Pople'


def build_tables():
    manifest=json.loads((ROOT/'snapshot-manifest.json').read_text())
    for path,sha in manifest['file_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
    engine_sha=hashlib.sha256((ROOT/'run_gpu_engine.py').read_bytes()+(ROOT/'protocols.py').read_bytes()).hexdigest()
    assert engine_sha==manifest['numerical_engine_sha256']
    keys=manifest['basis_keys'];assert len(keys)==24 and len(set(keys))==24
    rows=json.loads((ROOT/'inputs/structures.json').read_text())
    observables=json.loads((ROOT/'inputs/observables.json').read_text())
    settings=json.loads((ROOT/'inputs/settings.json').read_text())
    reference_levels=json.loads((ROOT/'inputs/reference-levels.json').read_text())
    assert len(rows)==30 and len(observables)==12
    assert {o['category'] for o in observables}=={'isomer','dimer'}
    assert {t['structure_id'] for o in observables for t in o['terms']}=={s['structure_id'] for s in rows}
    methods=settings['methods'];data={};bases={};costs=[];energies=[]
    for key in keys:
        basis=json.loads((ROOT/'frozen'/f'{key}.json').read_text());bases[key]=basis
        basis_rows={r['structure_id']:r for r in basis['records']}
        assert set(basis_rows)=={s['structure_id'] for s in rows}
        counts=[basis_counts(basis_rows[s['structure_id']]['atom_shells']) for s in rows]
        label=label_basis(key,basis)
        costs.append(dict(basis_key=key,label=label,kind=basis['kind'],family=family(label),
            mean_ao=float(np.mean([c['nao_spherical'] for c in counts])),
            mean_primitive=float(np.mean([c['primitive_spherical_unique'] for c in counts])),
            mean_radial_primitive=float(np.mean([c['primitive_radial_unique'] for c in counts])),
            min_ao=min(c['nao_spherical'] for c in counts),max_ao=max(c['nao_spherical'] for c in counts),
            update_step=basis.get('update_step',''),checkpoint_sha256=basis.get('checkpoint_sha256',''),
            scientific_source_sha256=basis.get('scientific_source_sha256','')))
        for method in methods:
            found={}
            for path in (ROOT/'results'/key/method).glob('*.json'):
                record=json.loads(path.read_text())
                assert record['status']=='passed' and record['variant']=='physical'
                assert record.get('active_nuclei') is None and record['grid_level']==3
                assert record['driver_sha256']==engine_sha
                sid=record['structure_id'];assert sid not in found
                assert np.isfinite(record['energy_hartree'])
                assert record['counts']==basis_counts(basis_rows[sid]['atom_shells'])
                if method=='mp2':
                    assert record['fp_type']=='FP64' and record['with_t2'] is False
                    structure=next(s for s in rows if s['structure_id']==sid)
                    assert record['frozen_core']==sum(z>2 for z in structure['atomic_numbers'])
                else:
                    assert record['nlc_enabled'] is True and record['nlc_grid_points']>0
                found[sid]=record;data[key,method,sid]=record
                energies.append(dict(basis_key=key,label=label,method=method,structure_id=sid,
                    energy_hartree=record['energy_hartree'],scf_energy_hartree=record['scf_energy_hartree'],
                    correlation_energy_hartree=record.get('correlation_energy_hartree',''),
                    nao=record['counts']['nao_spherical'],primitive=record['counts']['primitive_spherical_unique'],
                    device=record['device'],job_id=record['job_id'],seconds=record['seconds'],
                    record_file=str(path.relative_to(ROOT))))
            assert set(found)=={s['structure_id'] for s in rows},(key,method)
    assert len(data)==1440
    scores=[];relative=[]
    for cost in costs:
        key=cost['basis_key']
        for method in methods:
            errors={'isomer':[],'dimer':[]}
            for obs in observables:
                predicted=FACTOR*sum(t['coefficient']*data[key,method,t['structure_id']]['energy_hartree'] for t in obs['terms'])
                predicted+=obs.get('energy_add_kcal_mol',0.)
                reference=obs['reference_energy_kcal_mol'];error=(predicted-reference)*obs.get('error_multiplier',1.)
                assert np.isfinite([predicted,reference,error]).all()
                errors[obs['category']].append(error)
                level=reference_levels['ISO34' if obs['category']=='isomer' else 'S22B']
                relative.append(dict(basis_key=key,label=cost['label'],method=method,observable=obs['reaction_id'],
                    category=obs['category'],prediction_kcal_mol=predicted,reference_kcal_mol=reference,
                    signed_error_kcal_mol=error,absolute_error_kcal_mol=abs(error),
                    reference_method=level['reference_method'],reference_doi=level['doi']))
            records=[data[key,method,s['structure_id']] for s in rows]
            scores.append(dict(**cost,method=method,structures=30,isomerizations=6,dimers=6,
                balanced_mae_kcal_mol=balanced_mae(errors['isomer'],errors['dimer']),
                isomer_mae_kcal_mol=float(np.mean(np.abs(errors['isomer']))),
                dimer_mae_kcal_mol=float(np.mean(np.abs(errors['dimer']))),
                devices=';'.join(sorted({r['device'] for r in records}))))
    assert len(scores)==48 and len(relative)==576
    (ROOT/'tables').mkdir(exist_ok=True)
    for name,table in [('summary',scores),('basis-costs',costs),('total-energies',energies),('relative-energies',relative)]:
        dump_csv(ROOT/'tables'/f'{name}.csv',table)
    for method in methods:
        dump_csv(ROOT/'tables'/f'summary-{method}.csv',[r for r in scores if r['method']==method])
    reference_rows=[]
    for obs in observables:
        level=reference_levels['ISO34' if obs['category']=='isomer' else 'S22B']
        reference_rows.append(dict(observable=obs['reaction_id'],category=obs['category'],
            reference_kcal_mol=obs['reference_energy_kcal_mol'],reference_method=level['reference_method'],
            reference_doi=level['doi'],source_url=level['subset_source_url'],
            upstream_commit=level['upstream_commit']))
    dump_csv(ROOT/'tables/reference-energies.csv',reference_rows)
    xyz=ROOT/'structures';xyz.mkdir(exist_ok=True);combined=[];index=[]
    for row in rows:
        slug=row['structure_id'].replace('/','-')
        text=str(len(row['symbols']))+'\n'+row['structure_id']+'; coordinates Angstrom; charge=0 spin=0\n'
        text+=''.join(f'{s:2s} {c[0]:.12f} {c[1]:.12f} {c[2]:.12f}\n' for s,c in zip(row['symbols'],row['coords_angstrom']))
        (xyz/(slug+'.xyz')).write_text(text);combined.append(text)
        index.append(dict(structure_id=row['structure_id'],xyz_file=f'structures/{slug}.xyz',
                          atoms=len(row['symbols']),electrons=row['nelectron'],units='Angstrom'))
    (xyz/'all-30.xyz').write_text(''.join(combined));dump_csv(ROOT/'tables/structure-index.csv',index)
    verification=dict(status='passed',scope='complete non-QZ snapshot; partial 27-basis campaign',
        bases=24,traditional_bases=15,nnao_bases=9,structures=30,physical_energy_records=len(energies),
        reference_observables=12,summary_rows=len(scores),observable_rows=len(relative),
        numerical_engine_sha256=engine_sha,all_30_structures_used=True,cp_bsse_included=False,
        reference='frozen ISO34 and S22B database relative energies',
        reference_methods={key:level['reference_method'] for key,level in reference_levels.items()},
        excluded_bases=['def2-QZVP','def2-QZVPP','cc-pVQZ','aug-cc-pVQZ reference'],
        ground_state_basis_error='not reported; same-method QZ reference pending')
    (ROOT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    return scores


def place_labels(ax,rows,scale,blocked):
    from matplotlib.transforms import Bbox
    renderer=ax.figure.canvas.get_renderer();limits=ax.get_window_extent().padded(-5)
    markers=[]
    for row in rows:
        x,y=ax.transData.transform((row['mean_ao'],row['balanced_mae_kcal_mol']))
        radius=np.sqrt(scale*row['mean_primitive'])*ax.figure.dpi/72/2+2
        markers.append(Bbox.from_extents(x-radius,y-radius,x+radius,y+radius))
    occupied=list(blocked);labels=[]
    order=sorted(range(len(rows)),key=lambda i:(rows[i]['kind']!='nnao',rows[i]['balanced_mae_kcal_mol']))
    def overlap(a,b):return max(0,min(a.x1,b.x1)-max(a.x0,b.x0))*max(0,min(a.y1,b.y1)-max(a.y0,b.y0))
    for i in order:
        row=rows[i];point=ax.transData.transform((row['mean_ao'],row['balanced_mae_kcal_mol']))
        label=row['label'].replace('NNAO-','');fontsize=8.7
        text=ax.text(0,0,label,fontsize=fontsize,fontweight='semibold' if row['kind']=='nnao' else 'normal')
        bbox=text.get_window_extent(renderer);width,height=bbox.width+4,bbox.height+3;text.remove()
        candidates=[]
        for dx in [9,22,42,65,-9,-22,-42,-65]:
            for dy in [6,-6,17,-17,30,-30,46,-46,65,-65,86,-86]:
                left=point[0]+dx if dx>0 else point[0]+dx-width
                bottom=point[1]+dy-height/2
                box=Bbox.from_bounds(left,bottom,width,height)
                if not (box.x0>=limits.x0 and box.x1<=limits.x1 and box.y0>=limits.y0 and box.y1<=limits.y1):continue
                collision=sum(overlap(box,b) for b in occupied)*1e6+sum(overlap(box,b) for b in markers)*2e4
                score=collision+dx*dx+dy*dy+(80 if dx<0 else 0)
                candidates.append((score,box,left+2,bottom+1))
        assert candidates,row['label']
        _,box,x,y=min(candidates,key=lambda x:x[0]);occupied.append(box)
        position=ax.transData.inverted().transform((x,y))
        ax.annotate(label,xy=(row['mean_ao'],row['balanced_mae_kcal_mol']),xytext=position,
            textcoords='data',ha='left',va='bottom',fontsize=fontsize,
            fontweight='semibold' if row['kind']=='nnao' else 'normal',
            color='#A83D18' if row['kind']=='nnao' else '#34424E',
            arrowprops=dict(arrowstyle='-',color='#86929B',lw=.5,shrinkA=1.5,
                            shrinkB=np.sqrt(scale*row['mean_primitive'])/2+1),zorder=5)
        labels.append(dict(basis_key=row['basis_key'],text=label,text_x=float(position[0]),text_y=float(position[1])))
    return labels


def plot(scores):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'plot-cache'))
    os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'plot-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator,FuncFormatter
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none',
                         'xtick.direction':'in','ytick.direction':'in'})
    figdir=ROOT/'figures';figdir.mkdir(exist_ok=True)
    ys=[r['balanced_mae_kcal_mol'] for r in scores];assert min(ys)>0
    xlim=(0,max(r['mean_ao'] for r in scores)*1.18)
    ylim=(min(ys)*.65,max(ys)*1.5);scale=1.0
    layout={}
    for method,title in [('mp2','DF-MP2'),('wb97m-v',r'$\omega$B97M-V')]:
        rows=[r for r in scores if r['method']==method];assert len(rows)==24
        fig,ax=plt.subplots(figsize=(8.8,7.6))
        fig.subplots_adjust(left=.115,right=.975,bottom=.13,top=.85)
        ax.set(xlim=xlim,ylim=ylim,yscale='log',xlabel='Mean contracted spherical AOs / structure',
               ylabel=r'MAE vs database (kcal mol$^{-1}$)')
        ax.grid(color='#DFE5E9',lw=.6,which='major');ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
        ax.yaxis.set_major_locator(LogLocator(base=10,subs=[1,2,5]));ax.yaxis.set_major_formatter(FuncFormatter(lambda y,pos:f'{y:g}'))
        ax.yaxis.set_minor_locator(LogLocator(base=10,subs=[]))
        for row in sorted(rows,key=lambda r:-r['mean_primitive']):
            ax.scatter(row['mean_ao'],row['balanced_mae_kcal_mol'],s=scale*row['mean_primitive'],
                color=PALETTE[row['family']],alpha=.83,
                edgecolors='#252D33' if row['kind']=='nnao' else 'white',linewidths=.8,zorder=3)
        fig.suptitle(title+('  |  grid level 3' if method!='mp2' else '  |  frozen core'),y=.975,fontsize=17)
        family_handles=[Line2D([],[],linestyle='',marker='o',markersize=7,color=color,label=name,
                              markeredgecolor='#252D33' if name=='NNAO' else 'white') for name,color in PALETTE.items()]
        fig.legend(handles=family_handles,loc='upper center',bbox_to_anchor=(.52,.932),ncol=5,frameon=False,fontsize=10)
        sizes=sorted(set(int(round(v/10)*10) for v in np.quantile(
            [r['mean_primitive'] for r in scores],[.1,.5,.9])))
        handles=[ax.scatter([],[],s=scale*v,color='#B6BEC4',edgecolors='white',label=str(v)) for v in sizes]
        legend=ax.legend(handles=handles,title='Mean primitive GTOs',loc='upper right',frameon=False,
                         labelspacing=1.25,borderpad=.5,fontsize=9,title_fontsize=9)
        fig.canvas.draw();layout[method]=place_labels(ax,rows,scale,[legend.get_window_extent().padded(5)])
        fig.text(.115,.043,'24 non-QZ bases  |  30 structures  |  12 reference energy differences  |  no CP',fontsize=9,color='#53616B')
        fig.text(.115,.018,'References: ISO34 — W1-F12;  S22B — estimated CCSD(T)/CBS',fontsize=9,color='#53616B')
        for extension in ['png','pdf','svg']:
            fig.savefig(figdir/f'accuracy-vs-ao-{method}.{extension}',dpi=240,facecolor='white')
        plt.close(fig)
    (ROOT/'figure-layout.json').write_text(json.dumps(layout,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--tables-only',action='store_true');args=parser.parse_args()
    scores=build_tables()
    if not args.tables_only:plot(scores)
    for method in ['mp2','wb97m-v']:
        print(method,[(r['label'],round(r['balanced_mae_kcal_mol'],5)) for r in scores if r['method']==method])
