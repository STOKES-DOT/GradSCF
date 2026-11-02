"""Summarize accepted, matched-epoch training outcomes, including failures."""
import csv
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent

def main():
    experiment=json.loads((OUT/'experiment.json').read_text())
    baseline_root=Path(experiment['baseline_directory'])/'wide'
    base_history={r['epoch']:r for r in json.loads((baseline_root/'history.json').read_text())}
    result=dict(experiment=experiment,variants={});per_molecule=[];curves=[]
    for epoch,row in base_history.items():
        curves.append(dict(variant='baseline64',epoch=epoch,mean_energy_hartree=row['loss_hartree']))
    for name in experiment['variants']:
        root=OUT/name/'training'
        if not (root/'history.json').exists():
            result['variants'][name]=dict(status='no_accepted_checkpoint',
                preflight_log=str(OUT/f'{name}-preflight.log'),training_log=str(OUT/f'{name}-training.log'))
            continue
        history={r['epoch']:r for r in json.loads((root/'history.json').read_text())}
        progress=json.loads((root/'progress.json').read_text())
        metadata=json.loads((root/'metadata.json').read_text())
        gate=json.loads((root/'gate.json').read_text())
        assert gate['passed']
        np.testing.assert_allclose(history[0]['loss_hartree'],base_history[0]['loss_hartree'],atol=1e-10,rtol=0)
        aa={r['structure_id']:r['energy_hartree'] for r in history[0]['molecules']}
        bb={r['structure_id']:r['energy_hartree'] for r in base_history[0]['molecules']}
        assert aa.keys()==bb.keys()
        for key in aa:np.testing.assert_allclose(aa[key],bb[key],atol=1e-9,rtol=0)
        epoch=max(history.keys() & base_history.keys())
        row=history[epoch];base=base_history[epoch]
        baseline_energies={r['structure_id']:r['energy_hartree'] for r in base['molecules']}
        with np.load(root/'best.npz',allow_pickle=False) as checkpoint:
            best=json.loads(str(checkpoint['metadata'].item()))
        decreases=[]
        for molecule in row['molecules']:
            delta=1000*(molecule['energy_hartree']-baseline_energies[molecule['structure_id']])
            decreases.append(delta)
            per_molecule.append(dict(variant=name,epoch=epoch,structure_id=molecule['structure_id'],
                energy_hartree=molecule['energy_hartree'],baseline_energy_hartree=baseline_energies[molecule['structure_id']],
                variant_minus_baseline_millihartree=delta))
        result['variants'][name]=dict(status=progress['status'],matched_epoch=epoch,
            complete=progress['status']=='complete' and epoch==200,
            initial_energy_hartree=history[0]['loss_hartree'],energy_hartree=row['loss_hartree'],
            decrease_millihartree=1000*(history[0]['loss_hartree']-row['loss_hartree']),
            variant_minus_baseline_millihartree=1000*(row['loss_hartree']-base['loss_hartree']),
            lower_count=sum(x < -1e-6 for x in decreases),best_epoch=best['epoch'],
            best_energy_hartree=best['loss_hartree'],parameter_count=metadata['identity']['parameter_count'],
            training_seconds=progress.get('elapsed_seconds'),
            hot_batch_seconds=next(m['seconds'] for m in metadata['preparation_measurements'] if m['phase']=='hot_full_batch'))
        for e,r in history.items():curves.append(dict(variant=name,epoch=e,mean_energy_hartree=r['loss_hartree']))
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for name,rows in [('loss-curves.csv',curves),('per-molecule.csv',per_molecule)]:
        if rows:
            with (OUT/name).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(result['variants'],indent=2))

if __name__=='__main__':main()
