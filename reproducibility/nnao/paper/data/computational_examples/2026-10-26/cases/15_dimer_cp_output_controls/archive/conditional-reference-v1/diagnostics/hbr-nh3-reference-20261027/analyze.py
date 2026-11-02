"""Exact energy/reference decomposition of the reported adaptive-basis anomaly."""
from pathlib import Path
import csv
import json
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'nnao-dimer-diagnostic-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


HERE=Path(__file__).resolve().parent
CASE=Path('/Users/jiaoyuan/Documents/GitHub/NNAO_paper/data/computational_examples/2026-10-26/cases/15_dimer_cp_output_controls')
KCAL=627.5094740631


def table(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    records=json.loads((CASE/'combined/results.json').read_text())['results']
    selected=[r for r in records if r['basis']=='NNAO/MACE']
    rows=[]
    for record in selected:
        shift=(sum(record['own_energies_hartree'])-sum(record['isolated_energies_hartree']))*KCAL
        assert abs(record['model_interaction_kcal_mol']-
            record['uncorrected_interaction_kcal_mol']-shift)<2e-8
        rows.append(dict(identifier=record['identifier'],dataset=record['dataset'],group=record['group'],
            scale=record['scale'],model_interaction_kcal_mol=record['model_interaction_kcal_mol'],
            conditional_uncorrected_kcal_mol=record['uncorrected_interaction_kcal_mol'],
            conditional_cp_kcal_mol=record['cp_interaction_kcal_mol'],
            own_reference_shift_kcal_mol=shift,conditional_bsse_kcal_mol=record['bsse_kcal_mol']))
    table(HERE/'reference-shifts-all-systems.csv',rows)
    summaries=[]
    for dataset,group in sorted({(r['dataset'],r['group']) for r in rows}):
        points=sorted((r for r in rows if (r['dataset'],r['group'])==(dataset,group)),key=lambda r:r['scale'])
        values=[r['own_reference_shift_kcal_mol'] for r in points]
        summaries.append(dict(dataset=dataset,group=group,minimum_shift=min(values),maximum_shift=max(values),
            range_kcal_mol=max(values)-min(values),shortest_scale=points[0]['scale'],
            shortest_scale_shift=values[0]))
    table(HERE/'reference-shift-summary.csv',summaries)
    lookup={(r['basis'],r['scale']):r for r in records if r['group']=='1.28'}
    scales=sorted({r['scale'] for r in records if r['group']=='1.28'})
    differences=[]
    for scale in scales:
        a=lookup['NNAO/MACE',scale];f=lookup['NNAO/MACE-fixed',scale]
        differences.append(dict(scale=scale,
            delta_dimer_kcal_mol=(a['dimer_energy_hartree']-f['dimer_energy_hartree'])*KCAL,
            delta_own_sum_kcal_mol=(sum(a['own_energies_hartree'])-sum(f['own_energies_hartree']))*KCAL,
            delta_ghost_sum_kcal_mol=(sum(a['ghost_energies_hartree'])-sum(f['ghost_energies_hartree']))*KCAL,
            delta_conditional_raw_kcal_mol=a['uncorrected_interaction_kcal_mol']-f['uncorrected_interaction_kcal_mol'],
            delta_conditional_cp_kcal_mol=a['cp_interaction_kcal_mol']-f['cp_interaction_kcal_mol']))
    table(HERE/'hbr-adaptive-minus-fixed.csv',differences)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    for label,basis,field,color,style,marker in [
        ('Model: independently predicted monomers','NNAO/MACE','model_interaction_kcal_mol','#1B6CA8','-','o'),
        ('Conditional: dimer-predicted monomers','NNAO/MACE','uncorrected_interaction_kcal_mol','#D64045','-','s'),
        ('Fixed output at scale 1.00','NNAO/MACE-fixed','uncorrected_interaction_kcal_mol','#86569a',':','^'),
        ('DF-HF / cc-pVTZ, no CP','cc-pvtz','uncorrected_interaction_kcal_mol','#444444','--',None)]:
        axes[0].plot(scales,[lookup[basis,s][field] for s in scales],label=label,
                     color=color,ls=style,marker=marker,lw=1.7,ms=4)
    axes[0].set_title('HBr–NH3: different no-CP observables')
    axes[0].set_ylabel('Interaction energy (kcal/mol)')
    axes[0].legend(frameon=False,fontsize=8.5,loc='lower right')
    shifts=[lookup['NNAO/MACE',s]['model_interaction_kcal_mol']-
            lookup['NNAO/MACE',s]['uncorrected_interaction_kcal_mol'] for s in scales]
    axes[1].plot(scales,shifts,color='#bc6c25',marker='s',lw=1.8,ms=4)
    axes[1].set_title('Term removed by the conditional reference')
    axes[1].set_ylabel('Own-monomer reference shift (kcal/mol)')
    for ax in axes:
        ax.axhline(0,color='.5',lw=.7);ax.set_xlabel('Distance scale (dimensionless)')
        ax.tick_params(direction='in',top=True,right=True)
    for extension in ('png','pdf','svg'):
        fig.savefig(HERE/f'reference-decomposition.{extension}',dpi=180)
    plt.close(fig)
    report=dict(status='arithmetic_decomposition_complete',all_systems=12,geometry_points=108,
        shortest_hbr_scale=differences[0],
        adaptive_at_scale_08=next(r for r in rows if r['group']=='1.28' and r['scale']==.8),
        original_data_modified=False,training_state_modified=False,
        cp_redefinition_proposed_not_validated=True)
    (HERE/'decomposition.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
