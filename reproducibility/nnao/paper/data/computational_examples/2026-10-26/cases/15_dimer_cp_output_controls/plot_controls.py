"""Draw CP and no-CP dimer panels from complete exported numeric records."""
from io import BytesIO
import csv
import hashlib
from html import escape
import json
import os
from pathlib import Path
import re
import tempfile

os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'gradscf-dimer-controls-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image,ImageChops

HERE=Path(__file__).resolve().parent
PAPER=Path('/Users/jiaoyuan/Documents/GitHub/NNAO_paper')
SERIES=[
    ('Reference','Reference','#171717','-',1.7,None),
    ('sto-6g','STO-6G','#89939c','-',1.2,'o'),
    ('STO-6G+P','STO-6G+P','#86569a',':',1.5,'D'),
    ('6-31g(d,p)','6-31G(d,p)','#8d933e','-.',1.5,'o'),
    ('def2-svp','def2-SVP','#ba718b','--',1.4,'o'),
    ('cc-pvdz','cc-pVDZ','#5c8c9e',':',1.6,'o'),
    ('cc-pvtz','cc-pVTZ','#0057b8','-',2.8,'o'),
    ('NNAO/MACE-fixed','MACE, fixed output','#9c1a1f','-',1.8,'^'),
    ('NNAO/MACE','MACE, adaptive output','#ef233c','--',1.9,'s'),
]


def slug(text):return re.sub(r'[^A-Za-z0-9_-]+','-',text).strip('-')


def picture(record):
    if 'cached_png' in record:
        image=Image.open(HERE/'representative-structures'/record['cached_png']).convert('RGBA')
    else:
        from xyzrender.export import svg_to_png_bytes
        source=PAPER/'output/case08_xyzrender'/record['svg']
        image=Image.open(BytesIO(svg_to_png_bytes(source.read_text(),size=2400))).convert('RGBA')
    white=Image.new('RGBA',image.size,'white')
    image=Image.alpha_composite(white,image).convert('RGB')
    bbox=ImageChops.difference(image,Image.new('RGB',image.size,'white')).getbbox()
    assert bbox
    return image.crop((max(0,bbox[0]-20),max(0,bbox[1]-20),min(image.width,bbox[2]+20),min(image.height,bbox[3]+20)))


def main():
    verification=json.loads((HERE/'combined/verification.json').read_text());assert verification['status']=='passed'
    data=json.loads((HERE/'inputs/cases.json').read_text())
    with (HERE/'combined/curves.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==1728
    cache=HERE/'representative-structures/index.json'
    render_manifest=json.loads((cache if cache.exists() else PAPER/'output/case08_xyzrender/manifest.json').read_text())
    renders={r['id']:r for r in render_manifest['records']}
    groups=sorted({(p['dataset'],p['group']) for p in data['definitions']})
    out=HERE/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'axes.linewidth':.9,
                        'axes.labelsize':10.75,'xtick.direction':'in','ytick.direction':'in'})
    plotted=[];figure_records=[]
    books={mode:PdfPages(out/f'dimers-{mode}-with-structures.pdf') for mode in ('no-cp','cp')}
    try:
        for dataset,group in groups:
            points=sorted([p for p in data['definitions'] if (p['dataset'],p['group'])==(dataset,group)],key=lambda p:p['scale'])
            anchor=next(p for p in points if p['scale']==1.)
            image=picture(renders[anchor['identifier']])
            selected=[r for r in rows if (r['dataset'],r['group'])==(dataset,group)]
            values=[float(r['interaction_kcal_mol']) for r in selected]+[p['reference_kcal_mol'] for p in points]
            low,high=min(min(values),0.),max(max(values),0.)
            span=max(high-low,1e-3);limits=(low-.08*span,high+.65*span)
            for mode,cp_mode in (('no-cp','no_cp'),('cp','cp')):
                directory=out/mode;directory.mkdir(exist_ok=True)
                figure=plt.figure(figsize=(6.3,6.3));ax=figure.add_axes([.15,.13,.81,.81])
                handles=[]
                for basis,label,color,style,width,marker in SERIES:
                    if basis=='Reference':ys=[p['reference_kcal_mol'] for p in points]
                    else:
                        by_scale={float(r['scale']):r for r in selected if r['basis']==basis and r['cp_mode']==cp_mode}
                        assert set(by_scale)=={p['scale'] for p in points}
                        ys=[float(by_scale[p['scale']]['interaction_kcal_mol']) for p in points]
                    xs=[p['scale'] for p in points]
                    size=4.2 if basis=='cc-pvtz' else 3.5 if basis.startswith('NNAO') else 2.6
                    ax.plot(xs,ys,color=color,ls=style,lw=width,marker=marker,ms=size,markeredgewidth=.45)
                    handles.append(Line2D([],[],color=color,ls=style,lw=width,marker=marker,ms=size,label=label))
                    plotted.extend(dict(dataset=dataset,group=group,cp_mode=cp_mode,basis=basis,
                                        scale=x,value_kcal_mol=y) for x,y in zip(xs,ys))
                ax.set_ylim(*limits);ax.set_xlabel('Distance scale (dimensionless)')
                ax.set_ylabel(('CP-adjusted interaction energy' if mode=='cp' else 'Interaction energy')+r' (kcal mol$^{-1}$)')
                ax.tick_params(top=True,right=True);ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                ax.axhline(0.,color='.5',lw=.55,zorder=0)
                legend=ax.legend(handles=handles,loc='upper left',frameon=False,fontsize=8.5,
                                 labelspacing=.28,handlelength=2.,borderaxespad=.6)
                width=.43;height=min(.30,width*image.height/image.width)
                width=height*image.width/image.height
                inset=ax.inset_axes([.97-width,.97-height,width,height]);inset.imshow(image);inset.set_axis_off()
                figure.canvas.draw();renderer=figure.canvas.get_renderer()
                assert not legend.get_window_extent(renderer).overlaps(inset.get_window_extent(renderer)),(dataset,group)
                stem=slug(dataset+'-'+group)
                for extension in ('png','pdf','svg'):figure.savefig(directory/f'{stem}.{extension}',dpi=220)
                books[mode].savefig(figure);plt.close(figure)
                figure_records.append(dict(dataset=dataset,group=group,label=anchor['label'],mode=mode,
                    path=f'{mode}/{stem}',same_y_limits=list(limits),representative_structure=anchor['identifier']))
    finally:
        for book in books.values():book.close()
    assert len(plotted)==1944
    (out/'plot-values.json').write_text(json.dumps(plotted,indent=2)+'\n')
    (out/'plot-report.json').write_text(json.dumps(dict(status='complete',figures=figure_records,
        calculated_values=1728,reference_values=216,checkpoint_update=3351,
        frozen_reference_scale=1.,same_y_limits_for_cp_and_no_cp=True,
        independent_monomer_reference=True, basis_response_term_retained=True, cp_definition="model_plus_frozen_parameter_correction"),indent=2)+'\n')
    cards=[]
    for dataset,group in groups:
        stem=slug(dataset+'-'+group)
        cards.append('<h2>'+escape(dataset+' · '+group)+'</h2><div class="pair">'+''.join(
            '<section><h3>'+label+'</h3><a href="'+mode+'/'+stem+'.pdf"><img src="'+mode+'/'+stem+'.png"></a></section>'
            for mode,label in [('no-cp','No CP correction'),('cp','Model + frozen-parameter CP correction')])+'</div>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Dimer CP and MACE output controls</title><style>body{max-width:1450px;margin:30px auto;padding:0 18px;font:16px/1.5 Arial;color:#202328}'
        '.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}img{width:100%}h2{font-size:20px}h3{font-size:16px}'
        '@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>'
        '<h1>Dimer interactions: CP × MACE output policy</h1><p>Update 3351. Fixed output: coefficients at scale 1.00. '
        'Adaptive output: coefficients predicted at each geometry. Network weights and primitive exponents are fixed in both.</p>'
        '<p>Both dimer-output policies use the same constant independently predicted monomer references. The CP-adjusted curve adds the same-point frozen-parameter BSSE correction; it retains the basis-response term. Fixed output applies to the dimer scan only; no asymptotic zero is imposed.</p><p><a href="dimers-no-cp-with-structures.pdf">No-CP PDF</a> · <a href="dimers-cp-with-structures.pdf">CP PDF</a></p>'+''.join(cards))
    print('Rendered 24 figures and two 12-page books; 1944 source values.')


if __name__=='__main__':main()
