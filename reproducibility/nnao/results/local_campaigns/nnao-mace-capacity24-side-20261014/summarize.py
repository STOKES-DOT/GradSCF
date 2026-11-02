"""Compare matched epochs and structures; distinguish partial and complete runs."""
import csv
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent

def write_csv(name,rows):
    if not rows:return
    with (OUT/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    configuration=json.loads((OUT/'experiment.json').read_text())
    records={};histories={};molecules={};curve=[]
    for name in ['small','default']:
        root=OUT/name
        if not (root/'history.json').exists():
            records[name]=dict(status='not_started_or_no_accepted_checkpoint')
            continue
        history=json.loads((root/'history.json').read_text())
        progress=json.loads((root/'progress.json').read_text())
        metadata=json.loads((root/'metadata.json').read_text())
        assert metadata['identity']['model_config']==configuration['variants'][name]
        with np.load(root/'best.npz',allow_pickle=False) as checkpoint:
            best=json.loads(str(checkpoint['metadata'].item()))
        histories[name]={r['epoch']:r for r in history}
        initial,last=history[0],history[-1]
        records[name]=dict(status=progress['status'],accepted_epoch=last['epoch'],
            initial_energy_hartree=initial['loss_hartree'],last_energy_hartree=last['loss_hartree'],
            best_energy_hartree=best['loss_hartree'],best_epoch=best['epoch'],
            improvement_millihartree_per_molecule=1000*(initial['loss_hartree']-last['loss_hartree']),
            parameter_count=metadata['identity']['parameter_count'],
            trainable_parameter_count=metadata['identity']['trainable_parameter_count'],
            training_elapsed_seconds=progress.get('elapsed_seconds'),
            hot_batch_seconds=next((m.get('seconds') for m in metadata['preparation_measurements']
                                     if m.get('phase')=='hot_full_batch'),None),
            gradient_gate_passed=json.loads((root/'gate.json').read_text())['passed'])
        molecules[name]={r['structure_id']:r['energy_hartree'] for r in last['molecules']}
        for r in history:
            curve.append(dict(variant=name,epoch=r['epoch'],mean_energy_hartree=r['loss_hartree'],
                improvement_millihartree_per_molecule=1000*(initial['loss_hartree']-r['loss_hartree'])))
    paired=[];comparison={}
    if len(histories)==2:
        common_epoch=max(histories['small'].keys() & histories['default'].keys())
        a,b=histories['small'][common_epoch],histories['default'][common_epoch]
        np.testing.assert_allclose(histories['small'][0]['loss_hartree'],
            histories['default'][0]['loss_hartree'],atol=1e-10,rtol=0)
        initial_a={r['structure_id']:r['energy_hartree'] for r in histories['small'][0]['molecules']}
        initial_b={r['structure_id']:r['energy_hartree'] for r in histories['default'][0]['molecules']}
        assert initial_a.keys()==initial_b.keys()
        for key in initial_a:np.testing.assert_allclose(initial_a[key],initial_b[key],atol=1e-9,rtol=0)
        aa={r['structure_id']:r['energy_hartree'] for r in a['molecules']}
        bb={r['structure_id']:r['energy_hartree'] for r in b['molecules']}
        assert aa.keys()==bb.keys() and len(aa)==24
        for identifier in aa:
            paired.append(dict(structure_id=identifier,matched_epoch=common_epoch,
                initial_energy_hartree=initial_a[identifier],small_energy_hartree=aa[identifier],
                default_energy_hartree=bb[identifier],
                default_minus_small_millihartree=1000*(bb[identifier]-aa[identifier])))
        comparison=dict(matched_epoch=common_epoch,
            default_minus_small_mean_millihartree=1000*(b['loss_hartree']-a['loss_hartree']),
            default_lower_count=sum(r['default_minus_small_millihartree'] < -1e-6 for r in paired),
            complete=common_epoch==configuration['epochs'] and all(r['status']=='complete' for r in records.values()))
    report=dict(configuration=configuration,variants=records,comparison=comparison,
        interpretation='One-seed training-set capacity pilot; width and angular order both change. Not a held-out generalization benchmark.')
    (OUT/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    write_csv('loss-curves.csv',curve);write_csv('per-molecule.csv',paired)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
