"""Check all merged records and the exact data supplied to both plot sets."""
import csv
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
KCAL=627.5094740631


def main():
    data=json.loads((HERE/'inputs/cases.json').read_text())
    computed=json.loads((HERE/'run/results.json').read_text())
    merged=json.loads((HERE/'combined/results.json').read_text())
    assert computed['status']=='complete' and computed['failed']==0
    assert len(computed['results'])==204 and len(merged['results'])==864
    look={(r['identifier'],r['basis']):r for r in merged['results']}
    for record in data['reused_records']:
        actual=look[record['identifier'],record['basis']]
        assert actual==record,'An archived record changed during merging.'
    errors=[]
    for record in merged['results']:
        assert record['status']=='passed'
        a,b=record['own_energies_hartree'];ga,gb=record['ghost_energies_hartree'];ab=record['dimer_energy_hartree']
        assert ga<=a+1e-8 and gb<=b+1e-8
        for left,right in [((ab-a-b)*KCAL,record['uncorrected_interaction_kcal_mol']),
                           ((ab-ga-gb)*KCAL,record['cp_interaction_kcal_mol']),
                           ((a+b-ga-gb)*KCAL,record['bsse_kcal_mol'])]:
            errors.append(abs(left-right));assert math.isclose(left,right,abs_tol=2e-8,rel_tol=0)
    assert all(len(r['scf_diagnostics'])==5 and all(s['stability']['stable'] for s in r['scf_diagnostics'])
               for r in computed['results'])
    with (HERE/'combined/curves.csv').open() as stream:rows=list(csv.DictReader(stream))
    curve={(r['identifier'],r['basis'],r['cp_mode']):r for r in rows}
    assert len(curve)==1728
    values=json.loads((HERE/'figures/plot-values.json').read_text())
    assert len(values)==1944
    points={(p['dataset'],p['group'],p['scale']):p for p in data['definitions']}
    for item in values:
        point=points[item['dataset'],item['group'],item['scale']]
        expected=point['reference_kcal_mol'] if item['basis']=='Reference' else float(
            curve[point['identifier'],item['basis'],item['cp_mode']]['interaction_kcal_mol'])
        assert item['value_kcal_mol']==expected
    plots=json.loads((HERE/'figures/plot-report.json').read_text())['figures']
    assert len(plots)==24
    for point in data['definitions']:
        for basis in {r['basis'] for r in rows}:assert (point['identifier'],basis,'cp') in curve
    report=dict(status='passed',new_cases=204,reused_cases=660,total_cases=864,
        new_five_energy_stability_checks=1020,geometry_points=108,
        calculated_plot_values=1728,reference_plot_values=216,figures=24,
        max_cp_identity_error_kcal_mol=max(errors),independent_model_energy_excluded=True,
        frozen_reference_scale=1.,source_checkpoint_update=3351)
    (HERE/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
