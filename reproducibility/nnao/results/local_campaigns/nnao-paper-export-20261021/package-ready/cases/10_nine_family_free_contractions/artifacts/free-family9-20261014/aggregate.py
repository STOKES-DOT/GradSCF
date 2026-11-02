"""Honest partial/full matrix; no missing reaction components are imputed."""
import csv
import json
import math
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'nnao-relative200-20261014'
OLD=ROOT.parent/'nnao-free-sz-diagnostic-20261014'
FACTOR=627.5094740631

def write_csv(name,rows):
    if rows:
        with (ROOT/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    reactions=json.loads((OLD/'reactions.json').read_text())
    standard=json.loads((PARENT/'gpu4/comparison.json').read_text())['results']
    reference={r['structure_id']:r['energy_hartree'] for r in standard if r['basis']=='cc-pvtz' and r['status']=='passed'}
    relative=lambda reaction,e:math.fsum(t['coefficient']*e[t['structure_id']] for t in reaction['terms'])*FACTOR+reaction['energy_add_kcal_mol']
    rows=[];structures=[];states=[]
    for tier in ['s','m','l']:
        for zeta in [1,2,3]:
            name=f'{tier}-{zeta}';folder=ROOT/'results'/name;p=folder/'summary.json'
            d=json.loads(p.read_text()) if p.exists() else {}
            vp=folder/'gpu4-verification/comparison.json';v=json.loads(vp.read_text()) if vp.exists() else {}
            verified={r['structure_id'] for r in v.get('results',[]) if r['status']=='passed'}
            e={r['structure_id']:r['best_energy_hartree'] for r in d.get('results',[])
               if r['status'] in ['complete','partial'] and r['structure_id'] in verified}
            counts=Counter(s['status'] for r in d.get('results',[]) for s in r['starts'])
            states.append(dict(member=name,status=d.get('status','pending'),verified=len(verified),
                converged=counts['converged'],max_steps=counts['max_steps'],rejected=counts['rejected'],failed=counts['failed']))
            for r in d.get('results',[]):
                structures.append(dict(member=name,**{k:r.get(k) for k in ['structure_id','status','nao','pool_nao',
                    'best_energy_hartree','best_gradient_norm','converged_starts','endpoint_spread_millihartree']}))
            for reaction in reactions:
                ids={t['structure_id'] for t in reaction['terms']};value=relative(reaction,e) if ids<=e.keys() else None
                anchor=relative(reaction,reference)
                rows.append(dict(member=name,tier=tier,zeta=zeta,reaction_id=reaction['reaction_id'],category=reaction['category'],
                    status='verified' if value is not None else 'pending_or_failed',energy_kcal_mol=value,
                    hf_ccpvtz_kcal_mol=anchor,gmtkn55_kcal_mol=reaction['reference_energy_kcal_mol'],
                    error_vs_hf_kcal_mol=None if value is None else value-anchor,
                    error_vs_gmtkn55_kcal_mol=None if value is None else value-reaction['reference_energy_kcal_mol']))
    write_csv('relative-energies.csv',rows);write_csv('per-structure.csv',structures);write_csv('optimization-status.csv',states)
    fmt=lambda v:'—' if v is None else f'{v:.4f}'
    lines=['# 九基组自由收缩对照','',
        '三个完整诊断条目、7 个结构；每档固定同一指数池。S/M/L 不构成严格嵌套的 primitive 指数层级。',
        '以下仅列独立 GPU4PySCF 已复核的值；未完成/失败显示为空。每起点最多 500 步，不宣称全局结构极限。','',
        '| 基组 | 反应能 | HCN 势垒 | 甲酸二聚体结合能 | 收敛起点/28 | 验证结构/7 |',
        '|---|---:|---:|---:|---:|---:|']
    for state in states:
        selected=[r for r in rows if r['member']==state['member']]
        lines.append(f"| {state['member']} | "+' | '.join(fmt(r['energy_kcal_mol']) for r in selected)+
            f" | {state['converged']} | {state['verified']} |")
    lines += ['', '单位 kcal/mol；反应式及正负号沿用 GMTKN55。非共价能未做 CP 校正。',
        '相对 HF/cc-pVTZ 与高级别参考的误差分别在 relative-energies.csv；AO 数及多起点跨度在 per-structure.csv。',
        'S-TZ 的三列价层已张成全部三 primitive 价层空间，固定其可逆列表示并验证旋转不变性，仍优化极化收缩和 SCF 轨道。',
        '同一档中高 zeta 起点继承低 zeta 最佳解。L 档另保留之前的 SZ/DZ 候选并核验重放；历史优化预算不同，不作同耗时优化器排名。',
        '本族 TZ 仅增加价层径向通道，仍使用一套极化壳层；不等同于传统 cc-pVTZ 的完整角动量结构。',
        '模型与逐分子自由收缩的差距不能单独区分网络表达、训练不足和泛化。']
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n')
    (ROOT/'comparison.json').write_text(json.dumps(dict(rows=rows,statuses=states),indent=2)+'\n')

if __name__=='__main__':main()
