"""Prepare only missing fixed-output and STO-6G+P dimer cases."""
import json
import hashlib
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'tests/comparisons'))
from nnao_dimer_controls.protocols import fingerprint,prepare_cases


def main():
    target=HERE/'inputs';assert not target.exists()
    dataset=ROOT/'artifacts/nnao-curves-barriers-20261020/dataset.json'
    parent=ROOT/'artifacts/nnao-curves-barriers-repair-20261021/merged/results.json'
    data=json.loads(dataset.read_text());report=json.loads(parent.read_text())
    records=[r for r in report['results'] if r['task']=='curve']
    assert len(records)==648 and all(r['status']=='passed' for r in records)
    exports={r['key']:json.loads((ROOT/f'artifacts/nnao-curves-barriers-20261020/representative/{r["key"]}-basis.json').read_text())
             for r in records if r['basis']=='NNAO/MACE'}
    definitions={p['identifier']:p for p in data['curves']}
    for row in records:
        if row['basis']=='NNAO/MACE':
            assert exports[row['key']]['row']['coords_angstrom']==definitions[row['identifier']]['structure']['coords_angstrom']
    assets=json.loads((HERE/'sto6g-p-reference-assets.json').read_text())
    result=prepare_cases(data['curves'],records,exports,assets)
    assert len(result['jobs'])==204 and len(result['anchor_reuse'])==12
    priority=[]
    for basis,group,scale in [('NNAO/MACE-fixed','Water-Water',.9),
                              ('STO-6G+P','1.28',1.),
                              ('NNAO/MACE-fixed','Benzene-Benzene_pi-pi',2.),
                              ('STO-6G+P','AcOH-AcOH',1.)]:
        selected=[j for j in result['jobs'] if (j['basis'],j['group'],j['scale'])==(basis,group,scale)]
        assert len(selected)==1;priority.append(selected[0]['key'])
    result['jobs'].sort(key=lambda j:(priority.index(j['key']) if j['key'] in priority else len(priority),
                                    j['dataset'],j['group'],j['scale'],j['basis']))
    reused=[dict(r,policy='adaptive' if r['basis']=='NNAO/MACE' else 'conventional') for r in records]+result['anchor_reuse']
    document=dict(format='nnao-dimer-cp-output-controls-v1',checkpoint_update=3351,
        checkpoint_sha256=data['checkpoint_sha256'],
        frozen_scientific_source_sha256='8c77b02a219296e4085a29a977529f2ef2e8817584d39c880675747b98be05f0',
        method='DF-RHF',dtype='float64',auxbasis=data['auxbasis'],
        reference_scale=1.,ghost_coefficient_policy='same selected dimer coefficients in all five energies',
        dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        parent_report_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
        priority_keys=priority,jobs=result['jobs'],reused_records=reused,
        definitions=data['curves'],new_scf_energy_problems=1020,
        plot_fields=dict(no_cp='uncorrected_interaction_kcal_mol',cp='cp_interaction_kcal_mol'))
    target.mkdir()
    (target/'cases.json').write_text(json.dumps(document,indent=2)+'\n')
    shutil.copy2(HERE/'sto6g-p-reference-assets.json',target/'polarization-assets.json')
    print(json.dumps(dict(new_five_energy_cases=len(result['jobs']),reused_cases=len(reused),
                         total_final_cases=864,priority_cases=len(priority),
                         final_plot_values=1728,cases_fingerprint=fingerprint(document)),indent=2))


if __name__=='__main__':main()
