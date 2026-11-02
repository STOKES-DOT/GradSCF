"""Rebuild plot tables with common, constant independent-monomer references."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent
CASE=HERE
sys.path.insert(0,str(HERE/'reference-correction'))
from protocols import plotted_energy,conditional_energy


def write_json(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def main():
    mono=json.loads((HERE/'isolated-results.json').read_text())
    assert mono['status']=='complete' and mono['passed']==24
    raw=CASE/'combined/results.json';records=json.loads(raw.read_text())['results'];assert len(records)==864
    refs={};by_group=defaultdict(list)
    for record in records:
        if record['basis']=='NNAO/MACE-fixed':continue
        if 'isolated_energies_hartree' in record:
            by_group[record['dataset'],record['group'],record['basis']].append(record)
    for key,items in by_group.items():
        reference=next(r for r in items if r['scale']==1.)['isolated_energies_hartree']
        assert all(r['isolated_energies_hartree']==reference for r in items),'Variable isolated reference: '+str(key)
        refs[key]=list(reference)
    for item in mono['results']:
        assert item['status']=='passed' and item['stability']['stable']
        key=item['dataset'],item['group'],'STO-6G+P'
        refs.setdefault(key,[None,None])[item['fragment']]=item['energy_hartree']
    assert len(refs)==84 and all(len(v)==2 and all(x is not None for x in v) for v in refs.values())
    references=[]
    for key,values in sorted(refs.items()):
        references.append(dict(dataset=key[0],group=key[1],basis=key[2],
            isolated_energies_hartree=values,constant_over_scan=True,
            source='new native monomer calculation' if key[2]=='STO-6G+P' else 'archived independent monomer calculations'))
    write_json(HERE/'isolated-reference-map.json',references)
    rows=[];max_reconstruction=0.
    for record in records:
        key=(record['dataset'],record['group'],'NNAO/MACE' if record['basis']=='NNAO/MACE-fixed' else record['basis'])
        reference=refs[key]
        raw_value=plotted_energy(record,'no_cp',isolated_energies=reference)
        cp_value=plotted_energy(record,'cp',isolated_energies=reference)
        shift=raw_value-conditional_energy(record,'no_cp')
        assert abs(cp_value-conditional_energy(record,'cp')-shift)<2e-8
        if record['basis']!='NNAO/MACE-fixed' and 'model_interaction_kcal_mol' in record:
            error=abs(raw_value-record['model_interaction_kcal_mol'])
            max_reconstruction=max(max_reconstruction,error);assert error<2e-8
        for mode,value in (('no_cp',raw_value),('cp',cp_value)):
            rows.append(dict(identifier=record['identifier'],dataset=record['dataset'],group=record['group'],
                label=record['label'],scale=record['scale'],basis=record['basis'],policy=record['policy'],
                cp_mode=mode,interaction_kcal_mol=value,
                source_field='E_AB - E_A(isolated) - E_B(isolated)' if mode=='no_cp' else 'isolated_reference_energy + frozen_parameter_BSSE',
                isolated_a_hartree=reference[0],isolated_b_hartree=reference[1],
                dimer_energy_hartree=record['dimer_energy_hartree'],
                frozen_parameter_bsse_kcal_mol=record['bsse_kcal_mol'],
                conditional_diagnostic_kcal_mol=conditional_energy(record,mode),
                reference_response_kcal_mol=shift,
                reference_kcal_mol=record['reference_kcal_mol'],case_key=record['key']))
    assert len(rows)==1728
    by_key={(r['identifier'],r['basis'],r['cp_mode']):r for r in rows}
    for point in json.loads((CASE/'inputs/cases.json').read_text())['definitions']:
        for basis in {r['basis'] for r in rows}:
            a=by_key[point['identifier'],basis,'no_cp'];b=by_key[point['identifier'],basis,'cp']
            assert abs(b['interaction_kcal_mol']-a['interaction_kcal_mol']-a['frozen_parameter_bsse_kcal_mol'])<2e-8
        if point['scale']==1.:
            for mode in ('no_cp','cp'):
                assert by_key[point['identifier'],'NNAO/MACE',mode]['interaction_kcal_mol']==by_key[point['identifier'],'NNAO/MACE-fixed',mode]['interaction_kcal_mol']
    out=HERE/'combined';out.mkdir(exist_ok=True)
    with (out/'curves.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    inputs=HERE/'inputs';inputs.mkdir(exist_ok=True)
    if (CASE/'inputs/cases.json').resolve()!=(inputs/'cases.json').resolve():
        shutil.copy2(CASE/'inputs/cases.json',inputs/'cases.json')
    if not (HERE/'representative-structures').exists():
        shutil.copytree(CASE/'representative-structures',HERE/'representative-structures')
    report=dict(status='passed',plot_values=1728,geometry_points=108,independent_reference_pairs=84,
        new_sto6g_p_monomer_calculations=24,all_references_constant=True,
        fixed_and_adaptive_share_independent_mace_references=True,
        model_energy_reconstruction_max_error_kcal_mol=max_reconstruction,
        cp_definition='isolated-reference interaction plus frozen-parameter CP correction',
        basis_response_term_retained=True,large_distance_zero_forced=False,
        raw_source_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
        original_raw_energy_records_modified=False,training_modified=False)
    write_json(out/'verification.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
