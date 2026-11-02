"""Export category errors at the exact common reaction intersection."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
rows=list(csv.DictReader((OUT/'category-metrics.csv').open()))
snapshot=json.loads((OUT/'snapshot.json').read_text())
bases=['6-31g(d,p)','cc-pvdz','NNAO-200-best','NNAO-500-initial','NNAO-500-current']
labels=['6-31G(d,p)','cc-pVDZ','NNAO 200 / epoch 629','NNAO 500 / initial','NNAO 500 / current']
colors=['#397EB4','#397EB4','#78818B','#BEC4CB','#D97539']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                    'axes.spines.right':False,'axes.edgecolor':'#60656B','text.color':'#25282D'})
fig,axes=plt.subplots(1,3,figsize=(13.2,4.5),sharey=True)
for ax,category,title in zip(axes,['reaction','barrier','noncovalent'],['Reaction energies','Barrier heights','Noncovalent binding']):
    selected={r['basis']:r for r in rows if r['category']==category}
    values=[float(selected[b]['mae_vs_hf_ccpvtz_kcal_mol']) for b in bases]
    assert all(int(selected[b]['count'])==6 for b in bases)
    ax.barh(range(len(bases)),values,color=colors,height=.64)
    ax.set_yticks(range(len(bases)),labels);ax.set_xlim(0,max(values)*1.24)
    ax.set_title(title,pad=12);ax.set_xlabel('MAE (kcal/mol)');ax.set_axisbelow(True)
    ax.grid(axis='x',color='#E3E6E8',linewidth=.7)
    for i,value in enumerate(values):ax.text(value+max(values)*.025,i,f'{value:.3f}',va='center',fontsize=9)
axes[0].invert_yaxis()
fig.suptitle('Historical relative-energy panel: error against HF/cc-pVTZ',fontsize=14,y=.98)
fig.text(.01,.015,f"Frozen update {snapshot['update_step']}; 6 entries/category. DF-RHF, def2-universal-jkfit; NCI without counterpoise.\n"
    'Models also differ in training set, exponent pool and core reference; this is not a controlled architecture ablation.',fontsize=8,color='#525861')
fig.tight_layout(rect=(0,.13,1,.91))
fig.savefig(OUT/'relative-energy-mae.png',dpi=180,facecolor='white')
fig.savefig(OUT/'relative-energy-mae.pdf',facecolor='white')
