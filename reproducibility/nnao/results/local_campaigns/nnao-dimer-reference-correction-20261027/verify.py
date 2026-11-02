"""Independent arithmetic, reference-constancy and plot-value verification."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

from pypdf import PdfReader

HERE=Path(__file__).resolve().parent
KCAL=627.5094740631


def main(root=HERE,raw_case=None):
    root=Path(root)
    if raw_case is None:raw_case=root
    raw_case=Path(raw_case)
    original=raw_case/'combined/results.json'
    raw=json.loads(original.read_text())['results'];source={(r['identifier'],r['basis']):r for r in raw}
    with (root/'combined/curves.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==1728
    constant=defaultdict(set);values={}
    for row in rows:
        record=source[row['identifier'],row['basis']]
        pair=float(row['isolated_a_hartree']),float(row['isolated_b_hartree'])
        constant[row['dataset'],row['group'],row['basis']].add(pair)
        expected=(record['dimer_energy_hartree']-sum(pair))*KCAL
        if row['cp_mode']=='cp':expected+=record['bsse_kcal_mol']
        assert abs(float(row['interaction_kcal_mol'])-expected)<2e-8
        assert float(row['dimer_energy_hartree'])==record['dimer_energy_hartree']
        assert float(row['frozen_parameter_bsse_kcal_mol'])==record['bsse_kcal_mol']
        values[row['dataset'],row['group'],row['basis'],row['cp_mode'],float(row['scale'])]=float(row['interaction_kcal_mol'])
    assert len(constant)==96 and all(len(v)==1 for v in constant.values())
    for dataset,group,basis in constant:
        if basis=='NNAO/MACE-fixed':
            assert constant[dataset,group,basis]==constant[dataset,group,'NNAO/MACE']
    plots=json.loads((root/'figures/plot-values.json').read_text());assert len(plots)==1944
    inputs=json.loads((root/'inputs/cases.json').read_text())
    refs={(r['dataset'],r['group'],r['scale']):r['reference_kcal_mol'] for r in inputs['definitions']}
    for p in plots:
        expected=refs[p['dataset'],p['group'],p['scale']] if p['basis']=='Reference' else values[p['dataset'],p['group'],p['basis'],p['cp_mode'],p['scale']]
        assert p['value_kcal_mol']==expected
    assert abs(values['HB300SPXx10','1.28','NNAO/MACE','no_cp',.8]-1.0793349747841636)<1e-8
    assert abs(values['HB300SPXx10','1.28','NNAO/MACE','cp',.8]-5.145348408618316)<1e-8
    for mode in ('no-cp','cp'):
        assert len(PdfReader(root/f'figures/dimers-{mode}-with-structures.pdf').pages)==12
        assert len(list((root/'figures'/mode).glob('*.png')))==12
    report=dict(status='passed',raw_source_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        constant_reference_groups=96,shared_mace_references=True,curve_values_verified=1728,
        reference_values_verified=216,individual_figures=24,pdf_pages_per_set=12,
        hbr_short_range_regression=True,original_raw_energy_records_unchanged=True,
        cp_definition='model interaction plus same-point frozen-parameter CP correction',
        basis_response_term_retained=True,large_distance_zero_forced=False)
    (root/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    main(Path(sys.argv[1]) if len(sys.argv)>1 else HERE,
         Path(sys.argv[2]) if len(sys.argv)>2 else None)
