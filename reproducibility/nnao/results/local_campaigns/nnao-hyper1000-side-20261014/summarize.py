"""Compare resumed runs only at epochs accepted by every available run."""
import csv
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent

def main():
    experiment=json.loads((OUT/'continuation.json').read_text())
    runs={};histories={};curve=[]
    for name in experiment['execution_order']:
        root=OUT/name/'training'
        if not (root/'history.json').exists():
            runs[name]=dict(status='no_resumed_history',parent_epoch=200)
            continue
        history=json.loads((root/'history.json').read_text())
        p=json.loads((root/'progress.json').read_text())
        histories[name]={r['epoch']:r for r in history}
        runs[name]={k:p.get(k) for k in ['status','accepted_epoch','loss_hartree','best_loss_hartree','elapsed_seconds','error']}
        runs[name]['resume_gate']=json.loads((root/'gate.json').read_text())['resume_replay_passed']
        for h in history:curve.append(dict(variant=name,epoch=h['epoch'],energy_hartree=h['loss_hartree']))
    common_epoch=max(set.intersection(*(set(h) for h in histories.values()))) if len(histories)==4 else None
    matched={};molecules=[]
    if common_epoch is not None:
        base=histories['baseline'][common_epoch]
        base_energies={r['structure_id']:r['energy_hartree'] for r in base['molecules']}
        for name,h in histories.items():
            at=h[common_epoch]
            matched[name]=dict(energy_hartree=at['loss_hartree'],
                difference_from_baseline_millihartree=1000*(at['loss_hartree']-base['loss_hartree']),
                extra_decrease_since_epoch200_millihartree=1000*(h[200]['loss_hartree']-at['loss_hartree']))
            for row in at['molecules']:
                molecules.append(dict(variant=name,matched_epoch=common_epoch,structure_id=row['structure_id'],
                    energy_hartree=row['energy_hartree'],difference_from_baseline_millihartree=
                    1000*(row['energy_hartree']-base_energies[row['structure_id']])))
    result=dict(runs=runs,matched_epoch=common_epoch,matched=matched,
        complete=common_epoch==1000 and all(r['status']=='complete' for r in runs.values()))
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,rows in [('loss-curves.csv',curve),('per-molecule.csv',molecules)]:
        if rows:
            with (OUT/name).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
