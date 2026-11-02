"""Redraw the BSSE supplement from exported tables; no electronic calculations."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


SERIES = [
    ('sto-6g','STO-6G','#89939c','-',1.2),
    ('STO-6G+P','STO-6G+P','#86569a','-',1.3),
    ('6-31g(d,p)','6-31G(d,p)','#8a913a','-',1.4),
    ('def2-svp','def2-SVP','#ba718b','-',1.4),
    ('cc-pvdz','cc-pVDZ','#568da1','-',1.5),
    ('cc-pvtz','cc-pVTZ','#0057b8','-',2.2),
    ('ma-def2-qzvpp','ma-def2-QZVPP','#343434','-',1.5),
    ('NNAO/MACE','NNAO/MACE','#c43032','--',2.1),
]


def slug(text):
    import re
    return re.sub(r'[^A-Za-z0-9_-]+','-',text).strip('-')


def display_label(text):
    return text.replace('--',' – ').replace('_TS',' (T-shaped)').replace('_pi-pi',' (π–π)').replace('_OH-pi',' (OH–π)')


def load(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def save(figure,path):
    for suffix in ('png','pdf','svg'):
        figure.savefig(path.with_suffix('.'+suffix),dpi=180,bbox_inches='tight')


def draw_distance(ax,rows,*,legend=False):
    for basis,label,color,style,width in SERIES:
        subset=sorted((r for r in rows if r['basis']==basis),key=lambda r:float(r['scale']))
        if not subset:continue
        ax.plot([float(r['scale']) for r in subset],
                [float(r['bsse_kcal_mol']) for r in subset],
                label=label,color=color,ls=style,lw=width,
                marker='s' if basis=='NNAO/MACE' else 'o',ms=3.1)
    ax.set_yscale('log')
    ax.set_xlabel('Distance scale (dimensionless)')
    ax.set_ylabel(r'BSSE (kcal mol$^{-1}$; log scale)')
    ax.tick_params(direction='in',top=True,right=True,which='both')
    ax.grid(axis='y',which='major',color='.91',lw=.6)
    if legend:ax.legend(frameon=False,fontsize=9,loc='best')


def main(root):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                        'axes.spines.top':True,'axes.spines.right':True})
    root=Path(root);out=root/'figures';out.mkdir(exist_ok=True)
    s22=load(root/'tables/s22-summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,5.6),layout='constrained')
    cohorts=[('full_22','All 22 dimers; seven complete bases'),
             ('common_12','Same 12 dimers; all eight bases')]
    plotted=[]
    for ax,(cohort,title) in zip(axes,cohorts):
        selected={r['basis']:r for r in s22 if r['cohort']==cohort}
        series=[s for s in SERIES if s[0] in selected]
        for i,(basis,label,color,_,_) in enumerate(series):
            value=float(selected[basis]['mean_bsse_kcal_mol'])
            ax.scatter(value,i,color=color,s=48,zorder=3)
            ax.annotate(f'{value:.4f}',(value,i),xytext=(8,0),
                        textcoords='offset points',va='center',fontsize=9)
            plotted.append(dict(figure='s22-bsse-summary',cohort=cohort,basis=basis,
                                value_kcal_mol=value,n=int(selected[basis]['n'])))
        ax.set_yticks(range(len(series)),[s[1] for s in series]);ax.invert_yaxis()
        ax.set_xscale('log');ax.set_title(title,fontsize=11,pad=12)
        values=[float(selected[s[0]]['mean_bsse_kcal_mol']) for s in series]
        ax.set_xlim(min(values)*.55,max(values)*7)
        ax.set_xlabel(r'Mean BSSE (kcal mol$^{-1}$; log scale)')
        ax.grid(axis='x',color='.9',lw=.6);ax.tick_params(direction='in',top=True,right=True)
    save(fig,out/'s22-bsse-summary');plt.close(fig)

    rows=load(root/'tables/distance-counterpoise-all-cases.csv')
    groups=sorted({(r['dataset'],r['group']) for r in rows})
    fig,axes=plt.subplots(3,4,figsize=(15,12),layout='constrained')
    individual=out/'distance-bsse';individual.mkdir(exist_ok=True)
    with PdfPages(out/'distance-bsse-all-systems.pdf') as book:
        for ax,(dataset,group) in zip(axes.flat,groups):
            subset=[r for r in rows if (r['dataset'],r['group'])==(dataset,group)]
            draw_distance(ax,subset)
            label=display_label(subset[0]['label'])
            ax.set_title(f'{dataset} · {group}\n{label}' if dataset=='HB300SPXx10' else f'{dataset}\n{label}',fontsize=9)
            figure,single=plt.subplots(figsize=(6.3,6.3),layout='constrained')
            draw_distance(single,subset,legend=True)
            single.set_title(label,fontsize=11,pad=10)
            name=slug(dataset+'-'+group)
            save(figure,individual/name);book.savefig(figure);plt.close(figure)
            for row in subset:
                plotted.append(dict(figure='distance-bsse/'+name,dataset=dataset,group=group,
                    identifier=row['identifier'],scale=float(row['scale']),basis=row['basis'],
                    value_kcal_mol=float(row['bsse_kcal_mol'])))
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=6,frameon=False,fontsize=10)
    save(fig,out/'distance-bsse-overview');plt.close(fig)
    (root/'plot-values.json').write_text(json.dumps(plotted,indent=2)+'\n')
    return groups


if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent)
