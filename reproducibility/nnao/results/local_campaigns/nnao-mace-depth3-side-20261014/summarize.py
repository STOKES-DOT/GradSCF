"""Matched 200-epoch comparison including the 64-channel depth sweep."""
import csv
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
BASE=OUT.parent/'nnao-mace-capacity24-side-20261014'

def main():
    paths={'small8':BASE/'small','default32':BASE/'default','wide64':OUT.parent/'nnao-mace-width64-side-20261014/wide','deep64':OUT/'deep'}
    records={};histories={};curve=[]
    for name,root in paths.items():
        history=json.loads((root/'history.json').read_text())
        progress=json.loads((root/'progress.json').read_text())
        metadata=json.loads((root/'metadata.json').read_text())
        assert json.loads((root/'gate.json').read_text())['passed']
        identity=metadata['identity'];histories[name]={r['epoch']:r for r in history}
        with np.load(root/'best.npz',allow_pickle=False) as checkpoint:
            best=json.loads(str(checkpoint['metadata'].item()))
        first,last=history[0],history[-1]
        records[name]=dict(status=progress['status'],accepted_epoch=last['epoch'],
            initial_energy_hartree=first['loss_hartree'],last_energy_hartree=last['loss_hartree'],
            best_energy_hartree=best['loss_hartree'],best_epoch=best['epoch'],
            decrease_millihartree=1000*(first['loss_hartree']-last['loss_hartree']),
            parameter_count=identity['parameter_count'],model_config=identity['model_config'],
            training_elapsed_seconds=progress.get('elapsed_seconds'),
            hot_batch_seconds=next(m['seconds'] for m in metadata['preparation_measurements'] if m['phase']=='hot_full_batch'))
        for r in history:curve.append(dict(variant=name,epoch=r['epoch'],mean_energy_hartree=r['loss_hartree']))
    for name in paths:
        np.testing.assert_allclose(records[name]['initial_energy_hartree'],
            records['default32']['initial_energy_hartree'],atol=1e-10,rtol=0)
    epoch=max(set.intersection(*(set(h) for h in histories.values())))
    energies={name:{r['structure_id']:r['energy_hartree'] for r in h[epoch]['molecules']}
              for name,h in histories.items()}
    assert all(energies[name].keys()==energies['default32'].keys() for name in paths)
    rows=[dict(structure_id=i,matched_epoch=epoch,
        **{name+'_energy_hartree':energies[name][i] for name in paths},
        deep64_minus_wide64_millihartree=1000*(energies['deep64'][i]-energies['wide64'][i]))
        for i in energies['default32']]
    report=dict(variants=records,matched_epoch=epoch,
        complete=epoch==200 and all(r['status']=='complete' for r in records.values()),
        deep64_minus_wide64_mean_millihartree=1000*(histories['deep64'][epoch]['loss_hartree']-histories['wide64'][epoch]['loss_hartree']),
        deep64_lower_count=sum(r['deep64_minus_wide64_millihartree'] < -1e-6 for r in rows),
        interpretation='deep64 vs wide64 changes depth only (3 vs 2). Single-seed train-set comparison; earlier runs shown as context.')
    (OUT/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for filename,data in [('loss-curves.csv',curve),('per-molecule.csv',rows)]:
        with (OUT/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
