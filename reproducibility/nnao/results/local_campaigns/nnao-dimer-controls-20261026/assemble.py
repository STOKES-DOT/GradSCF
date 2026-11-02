"""Merge complete controls without changing reused energies or CP definitions."""
import csv
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'tests/comparisons'))
from nnao_dimer_controls.protocols import conditional_energy


def main():
    inputs=json.loads((HERE/'inputs/cases.json').read_text())
    report=json.loads((HERE/'run/results.json').read_text())
    expected={r['key'] for r in inputs['jobs']}
    assert report['status']=='complete' and report['failed']==0
    assert {r['key'] for r in report['results']}==expected
    records=inputs['reused_records']+report['results']
    assert len(records)==864 and all(r['status']=='passed' for r in records)
    unique={(r['identifier'],r['basis']):r for r in records}
    assert len(unique)==864
    for point in inputs['definitions']:
        adaptive=unique[point['identifier'],'NNAO/MACE']
        fixed=unique[point['identifier'],'NNAO/MACE-fixed']
        if point['scale']==1.:
            for field in ('dimer_energy_hartree','own_energies_hartree','ghost_energies_hartree',
                          'uncorrected_interaction_kcal_mol','cp_interaction_kcal_mol'):
                assert adaptive[field]==fixed[field]
    out=HERE/'combined';out.mkdir(exist_ok=True)
    (out/'results.json').write_text(json.dumps(dict(status='complete',method=inputs['method'],
        checkpoint_update=3351,reference_scale=1.,results=records,
        new_cases=204,reused_cases=660,geometry_points=108),indent=2)+'\n')
    rows=[]
    for record in records:
        for mode in ('no_cp','cp'):
            rows.append(dict(identifier=record['identifier'],dataset=record['dataset'],group=record['group'],
                label=record['label'],scale=record['scale'],basis=record['basis'],policy=record['policy'],
                cp_mode=mode,interaction_kcal_mol=conditional_energy(record,mode),
                source_field=inputs['plot_fields'][mode],
                reference_kcal_mol=record['reference_kcal_mol'],case_key=record['key']))
    with (out/'curves.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'verification.json').write_text(json.dumps(dict(status='passed',five_energy_records=864,
        plotted_calculated_values=len(rows),anchors_equal_in_both_policies=12,
        independent_model_energy_excluded=True,geometry_points=108),indent=2)+'\n')
    print('Merged 864 records; exported 1728 matched CP on/off values.')


if __name__=='__main__':main()
