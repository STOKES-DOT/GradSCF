"""Check file hashes, source rows, CP arithmetic, plot points and geometries."""
import csv
import hashlib
import json
import math
from pathlib import Path


KCAL=627.5094740631


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def verify(root):
    root=Path(root)
    manifest=json.loads((root/'FILE_MANIFEST.json').read_text())
    for relative,expected in manifest.items():
        assert digest(root/relative)==expected,relative
    reports=[('s22','source/s22/results.json','bsse','tables/s22-counterpoise-all-cases.csv'),
             ('distance','source/distance/results.json','curve','tables/distance-counterpoise-all-cases.csv')]
    counts={};errors=[]
    for name,source,task,table in reports:
        rows=read_csv(root/table)
        originals={r['key']:r for r in json.loads((root/source).read_text())['results'] if r['task']==task}
        assert len(rows)==len(originals)
        for row in rows:
            original=originals[row['key']]
            assert row['status']==original['status'] and row['basis']==original['basis']
            if row['status']!='passed':
                assert row['bsse_kcal_mol']=='' and row['error']==original['error']
                continue
            for field in ('bsse_hartree','bsse_kcal_mol','cp_interaction_kcal_mol',
                          'uncorrected_interaction_kcal_mol','dimer_energy_hartree'):
                assert float(row[field])==original[field],(name,row['key'],field)
            ea,eb=map(float,(row['own_a_hartree'],row['own_b_hartree']))
            ga,gb=map(float,(row['ghost_a_hartree'],row['ghost_b_hartree']))
            ab=float(row['dimer_energy_hartree'])
            assert ga<=ea+1e-8 and gb<=eb+1e-8
            tests=[(ea+eb-ga-gb,float(row['bsse_hartree'])),
                   ((ab-ea-eb)*KCAL,float(row['uncorrected_interaction_kcal_mol'])),
                   ((ab-ga-gb)*KCAL,float(row['cp_interaction_kcal_mol'])),
                   (float(row['bsse_hartree'])*KCAL,float(row['bsse_kcal_mol']))]
            for actual,expected in tests:
                errors.append(abs(actual-expected));assert math.isclose(actual,expected,abs_tol=2e-8,rel_tol=0)
        counts[name]=dict(cases=len(rows),passed=sum(r['status']=='passed' for r in rows),
                          failed=sum(r['status']!='passed' for r in rows))
    for group in json.loads((root/'structure-index.json').read_text()):
        for field in ('dimer_xyz','fragment_a_xyz','fragment_b_xyz'):
            lines=(root/group[field]).read_text().splitlines()
            assert int(lines[0])==len(lines)-2
        basis=json.loads((root/group['example_basis_file']).read_text())
        assert basis['row']['symbols']==group['symbols']
    cases={r['identifier']+'|'+r['basis']:r for r in read_csv(root/'tables/distance-counterpoise-all-cases.csv')}
    plot_values=json.loads((root/'plot-values.json').read_text())
    summaries={(r['cohort'],r['basis']):r for r in read_csv(root/'tables/s22-summary.csv')}
    for row in plot_values:
        expected=(float(cases[row['identifier']+'|'+row['basis']]['bsse_kcal_mol'])
            if 'identifier' in row else float(summaries[row['cohort'],row['basis']]['mean_bsse_kcal_mol']))
        assert row['value_kcal_mol']==expected
    assert counts=={'s22':{'cases':176,'passed':166,'failed':10},
                    'distance':{'cases':648,'passed':648,'failed':0}},counts
    return dict(status='passed',files_verified=len(manifest),cases=counts,
                structure_groups=130,plotted_values=len(plot_values),
                maximum_cp_arithmetic_error=max(errors),finite_differences=False,
                new_electronic_calculations=False)


if __name__=='__main__':
    import sys
    print(json.dumps(verify(sys.argv[1]),indent=2))
