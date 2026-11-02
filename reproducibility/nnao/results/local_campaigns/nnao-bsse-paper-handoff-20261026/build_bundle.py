"""Assemble existing BSSE evidence, export tables/XYZ, and redraw only figures."""
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import statistics

from plot_bsse import main as plot, slug, display_label
from verify_bundle import digest, verify


HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
PAPER=Path('/Users/jiaoyuan/Documents/GitHub/NNAO_paper')
OUT=HERE/'payload/2026-10-26'
CASE=OUT/'cases/14_bsse_counterpoise'
CORE=REPO/'artifacts/nnao-core-benchmarks-20261020'
CURVES=REPO/'artifacts/nnao-curves-barriers-repair-20261021'
ORIGINAL=REPO/'artifacts/nnao-curves-barriers-20261020'


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def main():
    assert not OUT.exists(),'Refuse to overwrite an assembled supplement.'
    CASE.mkdir(parents=True)
    sources=[]
    def copy(source,relative):
        source=Path(source);target=CASE/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        assert digest(source)==digest(target)
        sources.append(dict(source_path=str(source),packaged_path=relative,sha256=digest(source)))
        return relative

    for family,base,files in [
        ('s22',CORE,['dataset.json','RUN_SPEC.md','README.md','submission.json','completion.json']),
        ('s22',CORE/'full',['results.json','configuration.json','s22-cases.csv','s22-summary.csv','plot-summary.json']),
        ('distance',ORIGINAL,['dataset.json','RUN_SPEC.md','README.md','submission.json','completion.json']),
        ('distance',CURVES,['README.md','snapshot.json','source-patch.json']),
        ('distance',CURVES/'merged',['results.json','configuration.json','interaction-cases.csv','plot-summary.json']),
        ('distance',CURVES/'combined-retry',['independent-check.json']),
    ]:
        for name in files:
            # The repair README intentionally supersedes the original in this
            # focused copy; preserve both separately rather than overwrite.
            label=('repair-' if base==CURVES else '')+name
            copy(base/name,f'source/{family}/{label}')
    for extension in ('png','pdf'):
        copy(CORE/f'full/s22-counterpoise.{extension}',f'figures/original-s22-counterpoise.{extension}')
        for dataset in ('S66x8','HB300SPXx10'):
            copy(CURVES/f'merged/{dataset}-cp-control.{extension}',f'figures/{dataset}-frozen-cp-original.{extension}')
    model=PAPER/'data/computational_examples/2026-10-21/models/update-3351'
    export=json.loads((model/'parameter-export.json').read_text())
    assert digest(model/'parameters.npz')==export['inference_checkpoint_sha256']
    copy(model/'parameter-export.json','source/model-parameter-export.json')
    configs=[json.loads((CORE/'full/configuration.json').read_text()),
             json.loads((CURVES/'merged/configuration.json').read_text())]
    assert all(c['checkpoint_sha256']==export['original_checkpoint_sha256'] for c in configs)
    write_json(CASE/'model-provenance.json',dict(checkpoint_update=3351,
        original_checkpoint_sha256=export['original_checkpoint_sha256'],
        shared_model='../../../2026-10-21/models/update-3351/',
        parameters_copied=False,not_current_family500_training=True))

    s22data=json.loads((CORE/'dataset.json').read_text())
    curvedata=json.loads((ORIGINAL/'dataset.json').read_text())
    core_report=json.loads((CORE/'full/results.json').read_text())
    curve_report=json.loads((CURVES/'merged/results.json').read_text())
    records=[r for r in core_report['results'] if r['task']=='bsse']
    curve_records=[r for r in curve_report['results'] if r['task']=='curve']
    assert len(records)==176 and len(curve_records)==648
    datasets={'s22':{r['structure_id']:r for r in s22data['s22']},
              'distance':{r['identifier']:r for r in curvedata['curves']}}
    structure_index=[];lookup={}

    def xyz(row,relative,comment):
        target=CASE/relative;target.parent.mkdir(parents=True,exist_ok=True)
        lines=[str(len(row['symbols'])),comment+'; units=Angstrom']
        lines += [symbol+' '+' '.join(format(float(x),'.17g') for x in coord)
                  for symbol,coord in zip(row['symbols'],row['coords_angstrom'])]
        content='\n'.join(lines)+'\n'
        if target.exists():assert target.read_text()==content
        else:target.write_text(content)
        return relative

    for family,items in datasets.items():
        for identifier,item in items.items():
            row=item if family=='s22' else item['structure']
            groups=item['fragment_indices']
            assert row['charge']==0 and row['spin']==0
            assert sorted(groups[0]+groups[1])==list(range(len(row['symbols'])))
            name=slug(identifier)
            relative=f'structures/{family}/{name}'
            files=dict(dimer_xyz=xyz(row,relative+'-dimer.xyz',identifier+'; charge=0; spin=0'))
            for i,key in enumerate(('fragment_a_xyz','fragment_b_xyz')):
                indices=groups[i]
                part={k:[row[k][n] for n in indices] for k in ('symbols','coords_angstrom')}
                files[key]=xyz(part,relative+f'-fragment-{i+1}.xyz',
                               identifier+f'; fragment={i+1}; charge=0; spin=0; dimer geometry')
            if family=='distance':
                for i,monomer in enumerate(item['monomers']):
                    suffix=hashlib.sha256(json.dumps(monomer,sort_keys=True).encode()).hexdigest()[:12]
                    files[f'independent_monomer_{i+1}_xyz']=xyz(monomer,
                        'structures/independent-monomers/'+slug(monomer['structure_id'])+'-'+suffix+'.xyz',
                        monomer['structure_id']+f'; charge={monomer["charge"]}; spin={monomer["spin"]}; independent reference geometry')
            entry=dict(identifier=identifier,dataset='S22' if family=='s22' else item['dataset'],
                       group=item.get('group',item.get('label')),symbols=row['symbols'],
                       coords_angstrom=row['coords_angstrom'],fragment_indices=groups,
                       cp_fragment_charge=[0,0],cp_fragment_spin=[0,0],**files)
            structure_index.append(entry);lookup[family,identifier]=entry

    exported={};basis_checks=0
    for family,rows in (('s22',records),('distance',curve_records)):
        table=[]
        for row in rows:
            identifier=row['structure_id'] if family=='s22' else row['identifier']
            entry=lookup[family,identifier]
            folder=CORE/'full' if family=='s22' else ORIGINAL/'representative'
            basis=folder/(row['key']+'-basis.json')
            # The repaired solve retains the exported original basis exactly.
            expected=configs[1].get('native_retry',{}).get('basis_sha256',{}).get(row['key'])
            if family=='distance' and expected:
                assert digest(basis)==expected,(identifier,row['basis']);basis_checks+=1
            basis_file=copy(basis,f'frozen-bases/{family}/{basis.name}')
            data=json.loads(basis.read_text())
            assert data['row']['symbols']==entry['symbols']
            assert data['row']['coords_angstrom']==entry['coords_angstrom']
            if row['basis']=='NNAO/MACE':entry['example_basis_file']=basis_file
            record=dict(identifier=identifier,dataset=entry['dataset'],group=entry['group'],
                scale=row.get('scale',''),basis=row['basis'],status=row['status'],key=row['key'],
                label=row.get('label',''),nao=row.get('nao',''),basis_file=basis_file,**{
                    key:value for key,value in entry.items() if key.endswith('_xyz')})
            for field in ('dimer_energy_hartree','bsse_hartree','bsse_kcal_mol',
                'bsse_percent_reference_binding','uncorrected_interaction_hartree',
                'uncorrected_interaction_kcal_mol','cp_interaction_hartree',
                'cp_interaction_kcal_mol','model_interaction_kcal_mol','reference_kcal_mol'):
                record[field]=row.get(field,'')
            for field,prefix in (('own_energies_hartree','own'),('ghost_energies_hartree','ghost'),
                                 ('isolated_energies_hartree','independent')):
                pair=row.get(field,['','']);record[prefix+'_a_hartree']=pair[0];record[prefix+'_b_hartree']=pair[1]
            record.update(seconds=row['seconds'],error=row.get('error',''),
                          source_report=f'source/{family}/results.json')
            table.append(record)
        exported[family]=table
        write_csv(CASE/f'tables/{family}-counterpoise-all-cases.csv',table)
    write_json(CASE/'structure-index.json',structure_index)
    failures=[r for rows in exported.values() for r in rows if r['status']!='passed']
    write_csv(CASE/'tables/failed-cases.csv',failures)
    by_basis={basis:[r for r in records if r['basis']==basis and r['status']=='passed']
              for basis in configs[0]['bases']}
    common=set.intersection(*({r['structure_id'] for r in rows} for rows in by_basis.values()))
    assert len(common)==12
    summary=[]
    for cohort in ('full_22','common_12'):
        for basis,rows in by_basis.items():
            if cohort=='full_22' and len(rows)!=22:continue
            selected=[r for r in rows if cohort=='full_22' or r['structure_id'] in common]
            values=[r['bsse_kcal_mol'] for r in selected]
            summary.append(dict(cohort=cohort,basis=basis,n=len(values),
                mean_bsse_kcal_mol=statistics.mean(values),median_bsse_kcal_mol=statistics.median(values),
                max_bsse_kcal_mol=max(values),structure_ids=';'.join(sorted(r['structure_id'] for r in selected))))
    write_csv(CASE/'tables/s22-summary.csv',summary)
    write_json(CASE/'source/common-s22-dimer-ids.json',sorted(common))
    write_json(CASE/'SOURCE_MANIFEST.json',sources)
    shutil.copy2(HERE/'plot_bsse.py',CASE/'plot_bsse.py')
    # Plots are regenerated directly from the local CSV tables, without SCF.
    groups=plot(CASE)
    shutil.copy2(HERE/'verify_bundle.py',CASE/'verify_bundle.py')
    (CASE/'README.md').write_text((HERE/'case_README.md').read_text())
    labels={(r['dataset'],r['group']):display_label(r['label']) for r in curve_records}
    cards=''.join('<section><h3>'+html.escape(dataset+' · '+group+' · '+labels[dataset,group])+'</h3><a href="figures/distance-bsse/'+
        slug(dataset+'-'+group)+'.pdf"><img loading="lazy" src="figures/distance-bsse/'+
        slug(dataset+'-'+group)+'.png" alt="'+html.escape(group)+' BSSE versus distance"></a></section>'
        for dataset,group in groups)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NNAO BSSE evidence</title><style>body{max-width:1400px;margin:28px auto;padding:0 20px;font:16px/1.5 Arial;color:#202328;background:#fff}h1{font-size:28px}h2{margin-top:35px}a{color:#0057b8}img{width:100%;height:auto}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}section{min-width:0}h3{font-size:16px}nav a{margin-right:18px} @media(max-width:760px){.grid{grid-template-columns:1fr}}</style>
<h1>NNAO BSSE / counterpoise evidence</h1>
<p>Historical MACE update 3351 · DF-HF / def2-universal-jkfit · kcal mol⁻¹. This is an archive/redraw, with no new electronic calculations.</p>
<nav><a href="README.md">Methods and limitations</a><a href="tables/s22-counterpoise-all-cases.csv">S22: all 176 attempted cases</a><a href="tables/distance-counterpoise-all-cases.csv">Distance: 648 cases</a><a href="structure-index.json">Structures and frozen bases</a></nav>
<p>S22: 166/176 passed; ten ma-def2-QZVPP GPU-memory failures are retained as missing results. Distance: 648/648 passed, on twelve representative dimers and 108 geometries.</p>
<h2>S22 mean BSSE</h2><a href="figures/s22-bsse-summary.pdf"><img src="figures/s22-bsse-summary.png" alt="S22 mean BSSE: full 22-dimer and common 12-dimer cohorts"></a>
<h2>BSSE versus distance</h2><p>Positive BSSE is shown on logarithmic y axes to retain the long-distance tails. Markers are original calculated points; lines only join them.</p>
<p>For MACE, counterpoise holds the dimer coefficients fixed. It is not obtained by adding BSSE to the independently predicted dimer–monomer interaction energy.</p>
<p><a href="figures/distance-bsse-all-systems.pdf">Twelve-system PDF</a> · <a href="figures/distance-bsse-overview.png">Overview</a></p><div class="grid">'''+cards+'''</div>
<h2>Original archived figures</h2><ul><li><a href="figures/original-s22-counterpoise.pdf">Original S22 common-subset figure</a></li><li><a href="figures/S66x8-frozen-cp-original.pdf">S66×8 frozen-basis CP interactions</a></li><li><a href="figures/HB300SPXx10-frozen-cp-original.pdf">HB300SPX×10 frozen-basis CP interactions</a></li></ul></html>'''
    (CASE/'index.html').write_text(page)
    (OUT/'README.md').write_text('# BSSE evidence supplement — 2026-10-26\n\n'
        '[Case 14: BSSE / counterpoise figures and tables](cases/14_bsse_counterpoise/README.md)\n\n'
        'This focused supplement reorganizes unchanged historical data and adds plots derived from them. '
        'The 2026-10-21 and 2026-10-22 archives remain unchanged; no manuscript or training state is edited.\n')
    manifest={p.relative_to(CASE).as_posix():digest(p) for p in sorted(CASE.rglob('*')) if p.is_file()}
    write_json(CASE/'FILE_MANIFEST.json',manifest)
    report=verify(CASE);report['repaired_basis_hashes_verified']=basis_checks
    write_json(CASE/'VERIFY_REPORT.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
