"""Assemble a dated, immutable computational-example handoff; no QC reruns."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np


CASES = [
    dict(id='01_hcn_mace_vs_free', title='HCN training geometry and HCN/HNC transition geometry',
         sources=['nnao-free500-hcn-train-20261018','nnao-free500-hcnts-20261018'], cohort='MACE-update-3351',
         status='complete', result='Two finite-budget two-start free-contraction comparisons. MACE-to-free gaps are 0.8033984204 mEh for the training HCN geometry and 30.0321905941 mEh for the transition geometry.',
         limitation='The free endpoints satisfy the recorded coefficient-gradient criterion; they are not certified global minima.'),
    dict(id='02_ethane_torsion', title='Rigid ethane torsion: MACE, free contraction and conventional bases',
         sources=['nnao-ethane-torsion-20261018'], cohort='MACE-update-3351', status='complete',
         result='25 geometries, 0 to 120 degrees in 5-degree steps. Both free starts converged at every angle; 50 frozen bases were independently checked. Grid barriers: MACE 3.651135074 and free contraction 3.495155486 kcal/mol.',
         limitation='Rigid scan, fixed exponents/core and 44 spherical AOs. It is not a relaxed torsional barrier or a global free-contraction limit.'),
    dict(id='03_ethane_second_p', title='Ethane with a second carbon valence-p radial channel',
         sources=['nnao-ethane-p2-20261018'], cohort='free-contraction-only/update-3351-geometry', status='complete',
         result='25 checked geometries with 50 rather than 44 AOs. Fifteen angles were independently optimized and ten were transferred by exact reflection/atom permutation and rechecked. Free grid barrier: 3.133362078 kcal/mol.',
         limitation='The 44-AO MACE checkpoint was not enlarged or retrained. Transferred points are not independent optimizations; no interpolation was used.'),
    dict(id='04_ethane_radial_response', title='Carbon valence s/p radial functions during ethane torsion',
         sources=['nnao-ethane-radial-20261019'], cohort='MACE-update-3351', status='complete',
         result='Radial functions, coefficients and moments for C0 over 25 angles. The source reports maximum normalization error 4.44e-16 and analytic/quadrature moment difference 8.88e-16.',
         limitation='These are normalized basis functions, not occupied molecular orbitals or electron densities. The interactive HTML is retained; browser interaction was not newly verified.'),
    dict(id='05_ethane_cc_stretch', title='Restricted-HF ethane C-C stretching stress test',
         sources=['nnao-ethane-dissociation-20261019'], cohort='MACE-update-3351', status='complete_with_unconverged_endpoints',
         result='All 56 distances were processed; 51 free endpoints met the coefficient-gradient criterion. Five endpoints at 2.70, 2.75, 2.80, 2.85 and 2.90 Angstrom did not. Both valid and unconverged records/geometries are retained.',
         limitation='This follows a restricted RHF branch with rigid methyl groups, not physical homolytic dissociation. The nonconverged interval and nearby branch changes must remain explicit.'),
    dict(id='06_ethane_fixed_polarization', title='Fixed STO-6G polarization controls on the ethane torsion scan',
         sources=['nnao-ethane-sto6g-polarized-20261020'], cohort='update-3351-and-fixed-basis-controls', status='complete',
         result='75 new SCFs for STO-6G, STO-6G+C(d), and STO-6G+C(d)+H(p), all passed. Combined tables contain 11 methods/bases over 25 geometries.',
         limitation='The augmented STO-6G bases are explicitly defined custom controls, not a unique standard STO-6G* or STO-6G**. Absolute-energy and relative-curve quality are separate.'),
    dict(id='07_radial_bsse_polarizability', title='Atomic radial density, S22 counterpoise and static polarizability',
         sources=['nnao-core-benchmarks-20261020'], cohort='MACE-update-3351', status='complete_with_failures',
         result='284 attempted cases: 272 passed, 12 failed. All 35 NNAO cases passed. Failures are ma-def2-QZVPP GPU-memory errors: ten S22 dimers and two polarizabilities. Atomic cases 36/36; CP 166/176; polarizability 70/72.',
         limitation='Polarizabilities hold the predicted basis fixed under the field. Atomic charge/spin states extrapolate from neutral closed-shell molecular training. Shared-subset S22 statistics must not be described as complete QZ coverage.'),
    dict(id='08_distance_curves_and_barriers', title='Repaired representative S66x8, HB300SPXx10 and BH9 comparisons',
         sources=['nnao-curves-barriers-repair-20261021'], cohort='MACE-update-3351/native-UKS-repair', status='complete',
         result='Canonical merged report: 852/852 passed (S66x8 288, HB300SPXx10 360, BH9 204 species), yielding 54 reaction/basis profiles. Independent energies 31/31, maximum error 2.09184e-11 Eh; 13 selected internal-stability checks passed.',
         limitation='Only the 242-case repair subset received the specified native real-internal-stability checks. Global HF minima and external spin/complex stability are not certified. BH9 connectors are schematic R/TS/P levels, not IRC paths; no ZPE/thermal terms. HF lacks correlated dispersion.'),
    dict(id='09_historical_relative_energies', title='Frozen 500-system model inference on 18 historical relative energies',
         sources=['nnao-relative500-20261018'], cohort='MACE-update-3351', status='complete',
         result='39 structures, two frozen coefficient predictions per structure: native and independent energy checks each passed 78/78. Maximum native/reference difference: 7.6738615462e-12 Eh. Eighteen relative energies span reaction, barrier and noncovalent categories.',
         limitation='This is not full GMTKN55 or WTMAD2. Binding here is positive (monomer sum minus dimer), without CP; curve examples use negative interaction energies. Earlier 200-system models differ in data, pools, core assets and architecture.'),
    dict(id='10_nine_family_free_contractions', title='Nine S/M/L x SZP/DZP/TZP finite-budget free-contraction diagnostics',
         sources=['free-family9-20261014'], cohort='historical-legacy-core6/free-contraction', status='complete_finite_budget',
         result='Seven structures x nine families = 63 independently checked cases. Four starts per case have different optimizer outcomes retained in the status table. Maximum independent energy difference: 3.93106e-9 Eh.',
         limitation='Historical numerical core6 and candidate pools differ from the later BSE-reference-core cohort. Many starts reached budgets or were rejected. S/M/L pools are not strictly nested; this is not a global basis-capacity limit. TZP adds valence radial columns, not the angular structure of cc-pVTZ.'),
    dict(id='11_training_loss_snapshot', title='Historical 500-system training loss and logarithmic energy-change trends',
         sources=['nnao-mace500-loss-20261019'], cohort='500-system-training/through-epoch-596', status='historical_time_limit',
         result='History reaches epoch 596; the saved update 5969 is a partial epoch 597. Epochs 5-596 supply 592 complete online means. Epoch 4 is missing, and epochs 0-3 use the earlier full-evaluation policy.',
         limitation='Online means combine pre-update energies at ten parameter versions. They are not final-vector full-set scores, best-model criteria or CBS errors. The evaluation checkpoint used in recent examples is update 3351, not update 5969.'),
    dict(id='12_legacy_methane_design', title='Early methane contraction optimization and conventional-basis comparisons',
         sources=['methane-nnao-20260916','methane-basis-comparison-20260916',
                  'methane-basis-comparison-expanded-20260916','methane-core6-basis-comparison-20260916'],
         cohort='legacy-2026-09-prototypes', status='historical_examples',
         result='Original optimization records and fixed-geometry AO/Hamiltonian-size/energy comparisons are preserved as separate historical source runs.',
         limitation='Prototype basis/core/pool and network definitions differ from the later 500-system model. These rows must not be relabelled as update-3351 results.'),
    dict(id='13_legacy_aniline_basis_comparison', title='Earlier aniline GPU4PySCF conventional-basis comparison',
         sources=['aniline-gpu4pyscf-df-basis-final-20260923','aniline-basis-comparison-20260920'],
         cohort='legacy-2026-09-aniline-model', status='historical_examples',
         result='Original single-geometry basis-size and total-energy tables; the final comparison includes a 98-AO trained NNAO basis and conventional minimal-to-triple-zeta controls.',
         limitation='This older model and cc-pVQZ-JKFIT auxiliary basis differ from the update-3351/def2-universal-jkfit cohort. The single aniline example does not establish generalization.'),
]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--downloads',type=Path,required=True)
    args=parser.parse_args()
    source=args.source_root.resolve(); out=args.output.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new empty output directory.')
    out.mkdir(parents=True,exist_ok=True)
    imported=[]; excluded=[]; geometries=[]; case_rows=[]

    def copy(src,dest,*,case,origin=None):
        target=out/dest
        if target.exists():
            if sha(src)!=sha(target):raise ValueError('Archive name collision: '+str(dest))
            return target
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,target)
        digest=sha(src)
        if digest!=sha(target):raise ValueError('Copy checksum mismatch: '+str(src))
        imported.append(dict(path=str(dest),source=origin or str(src),sha256=digest,
                             bytes=target.stat().st_size,case=case))
        return target

    def ingest(root,prefix,case,*,predicate=None):
        allowed={'.json','.jsonl','.csv','.npz','.npy','.msgpack','.xyz','.sdf',
                 '.png','.pdf','.svg','.html','.md','.txt','.py','.sh','.slurm','.log','.patch'}
        for src in sorted(root.rglob('*')):
            if not src.is_file():continue
            rel=src.relative_to(root)
            reason=None
            if src.is_symlink():reason='symlink not followed'
            elif any(p in {'__pycache__','.mpl-cache','mpl-cache','jit-cache','build','.git'} for p in rel.parts):reason='generated cache/build metadata'
            elif src.name.startswith(('._','.DS_Store')):reason='macOS metadata'
            elif any(p in {'cache','caches'} for p in rel.parts) and src.suffix not in {'.json','.md'}:reason='integral cache; not needed to inspect results or frozen bases'
            elif src.suffix not in allowed:reason='opaque archive or unsupported duplicate format'
            elif src.suffix=='.log' and src.stat().st_size>1024**2:reason='large log omitted; structured results and failures retained'
            elif predicate is not None and not predicate(rel):reason='superseded or pilot subtree outside this case'
            if reason:
                excluded.append(dict(source=str(src),bytes=src.stat().st_size,reason=reason,case=case));continue
            copy(src,Path(prefix)/rel,case=case)

    for case in CASES:
        case=dict(case,method={
            '07_radial_bsse_polarizability':'DF-HF (RHF/UHF according to spin), def2-universal-jkfit',
            '08_distance_curves_and_barriers':'DF-HF (RHF/UHF according to spin), def2-universal-jkfit',
            '12_legacy_methane_design':'Legacy HF/NNAO prototype settings; retain each source run separately',
            '13_legacy_aniline_basis_comparison':'DF-RHF; final GPU4PySCF comparison uses cc-pVQZ-JKFIT',
        }.get(case['id'],'All-electron DF-RHF, def2-universal-jkfit'))
        prefix=Path('cases')/case['id']
        for name in case['sources']:
            root=source/'artifacts'/name
            if not root.is_dir():raise FileNotFoundError(root)
            predicate=(lambda rel: rel.parts[0]!='smoke') if case['id']=='07_radial_bsse_polarizability' else None
            ingest(root,prefix/'artifacts'/name,case['id'],predicate=predicate)
        if case['id'] in {'03_ethane_second_p','04_ethane_radial_response','06_ethane_fixed_polarization'}:
            copy(source/'artifacts/nnao-ethane-torsion-20261018/structures-dense.json',
                 prefix/'geometry-input/structures-dense.json',case=case['id'])
        if case['id']=='11_training_loss_snapshot':
            copy(source/'artifacts/mace500-20261014/inputs/references.json',
                 prefix/'geometry-input/training-references.json',case=case['id'])
        if case['id']=='08_distance_curves_and_barriers':
            original=source/'artifacts/nnao-curves-barriers-20261020'
            copy(original/'dataset.json',prefix/'inputs/dataset.json',case=case['id'])
            copy(original/'RUN_SPEC.md',prefix/'records/original-RUN_SPEC.md',case=case['id'])
            copy(original/'representative/results.json',prefix/'superseded/original-results.json',case=case['id'])
            for p in sorted((original/'representative').glob('*-basis.json')):
                copy(p,prefix/'frozen-bases'/p.name,case=case['id'])
            for name in ('retry.py','rescue.py','merge.py','join.py','protocols.py','plot.py','run.py'):
                copy(source/'tests/comparisons/nnao_reactivity'/name,prefix/'implementation'/name,case=case['id'])
            for name in ('native.py','run.py','protocols.py'):
                copy(source/'tests/comparisons/nnao_core'/name,prefix/'implementation/nnao_core'/name,case=case['id'])
            for p in sorted((args.downloads/'checked_states').glob('*.npz')):
                copy(p,prefix/'independent-checked-states'/p.name,case=case['id'],origin=dict(
                    host='jy_gpu_2',path='/home/user_yuanjiao/private/homefile/GradSCF_mace500_20261014/artifacts/nnao-curves-barriers-repair-20261021/combined-retry/'+p.name))
        if case['id']=='07_radial_bsse_polarizability':
            for p in sorted((source/'tests/comparisons/nnao_core').glob('*.py')):
                copy(p,prefix/'implementation'/p.name,case=case['id'])
        case_rows.append(dict(case,archive_path=str(prefix)))

    models=Path('models/update-3351')
    model_files={'snapshot.json':'artifacts/nnao-relative500-20261018/snapshot.json',
                 'fixed-candidate.json':'artifacts/nnao-family500-20261021/inputs/l-candidate.json',
                 'model-config.json':'artifacts/mace500-20261014/inputs/model-config.json',
                 'training-references.json':'artifacts/mace500-20261014/inputs/references.json',
                 'calibration-references.json':'artifacts/mace500-20261014/inputs/pilot.json'}
    for name,path in model_files.items():copy(source/path,models/name,case='shared-model')
    snapshot=json.loads((out/models/'snapshot.json').read_text())
    weights=args.downloads/'models/checkpoint-current.npz'
    if sha(weights)!=snapshot['checkpoint_sha256']:raise ValueError('Frozen checkpoint checksum differs.')
    with np.load(weights,allow_pickle=False) as checkpoint:
        parameters=checkpoint['parameters']
    parameter_hash=hashlib.sha256(parameters.tobytes()).hexdigest()
    if parameter_hash!=snapshot['parameter_sha256'] or parameters.size!=snapshot['parameter_count']:
        raise ValueError('Frozen inference parameters differ from snapshot.')
    np.savez_compressed(out/models/'parameters.npz',parameters=parameters)
    exported=dict(snapshot,checkpoint_sha256=sha(out/models/'parameters.npz'),
                  export_kind='inference-only parameters; optimizer and saved gradients omitted',
                  original_checkpoint_sha256=snapshot['checkpoint_sha256'])
    dump(out/models/'inference-snapshot.json',exported)
    dump(out/models/'parameter-export.json',dict(original_checkpoint_sha256=sha(weights),
        original_checkpoint_bytes=weights.stat().st_size,parameter_sha256=parameter_hash,
        inference_checkpoint_sha256=exported['checkpoint_sha256'],dtype=str(parameters.dtype),
        shape=list(parameters.shape),parameter_bytes=parameters.nbytes,
        source_host='jy_gpu_2',source_path='/home/user_yuanjiao/private/homefile/GradSCF_mace500_20261014/artifacts/nnao-relative500-20261018/checkpoint-current.npz',
        parameters_bit_identical=True,optimizer_and_saved_gradient_excluded=True))
    excluded.append(dict(source=str(weights),bytes=weights.stat().st_size,sha256=sha(weights),
        reason='Full training state omitted; exact inference parameters exported and independently hashed.',case='shared-model'))
    for name in ('BASIS_GRID.md','ELEMENTS.md','POOL_CALIBRATION.md'):
        copy(source/'src/gradscf/model/nnao'/name,Path('contexts/basis-design')/name,case='context')
    ingest(source/'artifacts/nnao-family500-20261021','contexts/ongoing-training','context',
           predicate=lambda p: p.parts[0] not in {'cache','runs','native-build','m-calibration','__pycache__'})

    # Export only explicitly Angstrom-labelled coordinates, retaining exact states.
    def walk(value,location,inherited=None):
        inherited=dict(inherited or {})
        if isinstance(value,dict):
            inherited.update({k:value[k] for k in ('charge','spin') if k in value and isinstance(value[k],(int,float))})
            symbols=value.get('symbols'); coords=value.get('coords_angstrom')
            atoms=value.get('atom_angstrom')
            if coords is None and isinstance(atoms,list) and atoms:
                if all(isinstance(a,list) and len(a)==4 and isinstance(a[0],str) for a in atoms):
                    symbols=[a[0] for a in atoms]; coords=[a[1:] for a in atoms]
                elif all(isinstance(a,list) and len(a)==2 and isinstance(a[0],str) for a in atoms):
                    symbols=[a[0] for a in atoms]; coords=[a[1] for a in atoms]
            if isinstance(symbols,list) and coords is not None and symbols:
                if (len(symbols)!=len(coords) or not all(isinstance(s,str) for s in symbols)
                        or not all(isinstance(x,list) and len(x)==3 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in x) for x in coords)):
                    raise ValueError('Invalid explicit coordinate record: '+location)
                label=str(value.get('structure_id',value.get('identifier',value.get('label',location))))
                yield dict(symbols=symbols,coords_angstrom=coords,charge=inherited.get('charge'),
                           spin=inherited.get('spin'),label=label,source_record=location)
            for key,child in value.items():
                yield from walk(child,location+'/'+key,inherited)
        elif isinstance(value,list):
            for i,child in enumerate(value):yield from walk(child,location+'/'+str(i),inherited)

    scopes=[(r['id'],Path(r['archive_path'])) for r in case_rows]+[('shared-model',models)]
    for case_id,prefix in scopes:
        found={}
        for path in sorted((out/prefix).rglob('*')):
            if path.suffix not in {'.json','.jsonl'} or not path.is_file():continue
            rel=str(path.relative_to(out))
            records=[json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.suffix=='.jsonl' else [json.loads(path.read_text())]
            for index,record in enumerate(records):
                for geom in walk(record,rel+(':'+str(index+1) if path.suffix=='.jsonl' else '')):
                    identity={k:geom[k] for k in ('symbols','coords_angstrom','charge','spin')}
                    digest=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                    if digest in found:
                        found[digest]['source_records'].append(geom['source_record']);continue
                    name=re.sub(r'[^A-Za-z0-9_.-]+','_',geom['label']).strip('_.')[-65:] or 'structure'
                    dest=prefix/'structures'/f'{name}-{digest[:12]}.xyz'
                    target=out/dest;target.parent.mkdir(parents=True,exist_ok=True)
                    comment=f'id={geom["label"]} charge={geom["charge"]} spin_2S={geom["spin"]} units=angstrom'
                    target.write_text(str(len(geom['symbols']))+'\n'+comment+'\n'+''.join(
                        s+' '+' '.join(format(float(v),'.17g') for v in xyz)+'\n' for s,xyz in zip(geom['symbols'],geom['coords_angstrom'])))
                    # Ensure exports preserve every parsed numeric coordinate exactly.
                    parsed=[[float(v) for v in line.split()[1:]] for line in target.read_text().splitlines()[2:]]
                    if parsed!=geom['coords_angstrom']:raise ValueError('XYZ coordinate roundtrip changed: '+str(dest))
                    found[digest]=dict(case=case_id,path=str(dest),geometry_sha256=digest,natoms=len(geom['symbols']),
                        charge=geom['charge'],spin_2S=geom['spin'],label=geom['label'],units='angstrom',source_records=[geom['source_record']])
        geometries.extend(found.values())
        if case_id!='shared-model' and not found:raise ValueError('No exact structures recovered: '+case_id)
        if case_id!='shared-model':next(r for r in case_rows if r['id']==case_id)['exported_structures']=len(found)

    for case in case_rows:
        prefix=Path(case['archive_path'])
        figures=sorted(p.relative_to(out/prefix) for p in (out/prefix).rglob('*') if p.suffix in {'.png','.pdf','.html'})
        reports=sorted(p.relative_to(out/prefix) for p in (out/prefix).rglob('*') if p.name in {
            'summary.json','results.json','scan-summary.json','comparison.json','verification-summary.json','metrics.json'})
        lines=[f'# {case["title"]}','','## Methods','',
            f'Method: {case["method"]}. Cohort: `{case["cohort"]}`. Exact settings, commands, hardware and original status fields remain in the imported source records. Numerical source files are copied without editing.',
            '', '## Results','',case['result'],'',f'Archive status: `{case["status"]}`. Exported geometry/state records: **{case["exported_structures"]}**.',
            '', '## Interpretation boundary','',case['limitation'],'', '## Files','',
            '- `structures/`: generated standard XYZ with Angstrom coordinates, charge and 2S in the comment. Missing state information is explicitly `None`, not assigned a guessed state.',
            '- `artifacts/`: source data, coefficients, plots, scripts and protocol records with their original relative paths. Original-language notes are retained as source records.',
            '- `../../structure-index.json`: coordinate/state provenance and aliases; `../../FILE_MANIFEST.json`: original paths and SHA256 checksums.',
            '', '### Reports','']
        lines.extend(f'- [{p}]({p.as_posix()})' for p in reports[:18])
        lines+=['','### Figures','']+[f'- [{p}]({p.as_posix()})' for p in figures[:18]]
        lines+=['','## Provenance','', 'Original GradSCF artifact directories: '+', '.join('`'+s+'`' for s in case['sources'])+'.',
            'Copied drivers are evidence snapshots and may retain original checkout paths. Running a numerical driver requires the corresponding GradSCF environment; this import did not rerun SCF, optimization or training.','']
        (out/prefix/'README.md').write_text('\n'.join(lines))

    dump(out/'catalog.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_root=str(source),cases=case_rows,notes=[
        'Data-only handoff; no manuscript files are edited.',
        'Cases retain their actual checkpoint/core/pool/auxiliary identities; do not pool unlike cohorts.',
        'The repaired merged results in case 08 are canonical; superseded reports remain explicitly separate.',
        'Ongoing-training context is a dated configuration/status snapshot, not completed study results.']))
    dump(out/'structure-index.json',geometries)
    fields=['case','path','geometry_sha256','natoms','charge','spin_2S','label','units','source_records']
    with (out/'structure-index.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for g in geometries:writer.writerow(dict(g,source_records=json.dumps(g['source_records'])))
    dump(out/'EXCLUDED_FILES.json',excluded)
    lines=['# NNAO computational examples — 2026-10-21','','Data-only handoff for the manuscript Codex. No manuscript text was edited or numerical calculation rerun.',
        '', '## Case index','', '| Case | Cohort | Evidence status | XYZ records |','| --- | --- | --- | ---: |']
    lines += [f'| [{c["title"]}]({c["archive_path"]}/README.md) | {c["cohort"]} | {c["status"]} | {c["exported_structures"]} |' for c in case_rows]
    lines += ['', '## Reading order and evidence boundaries','',
        'For recent checkpoint-3351 examples, start with cases 01-09. Case 10 is a historical nine-family free-contraction diagnostic, case 11 is the training-history snapshot, and cases 12-13 are earlier methane/aniline prototypes.',
        'Case 08 contains the corrected 852-case view and 31 independently checked density/orbital states. Case 07 retains all 12 QZ memory failures. Case 05 retains all five nonconverged free endpoints. Do not infer global variational minima, complete QZ coverage or physical RHF homolytic dissociation.',
        '', '## Structures, units and model','',
        f'There are {len(geometries)} case-local geometry/state XYZ exports, with {len({g["geometry_sha256"] for g in geometries})} distinct exact coordinate/charge/spin signatures across cases. These are not counts of chemically distinct molecules. All are exported from explicitly Angstrom-labelled source fields with exact float roundtrips; original XYZ/SDF files are also preserved where present.',
        'Absolute energies are normally Hartree and relative energies kcal/mol; raw columns retain their source conventions. Radial grids/moments use Bohr and polarizability uses Bohr^3. Positive historical binding energies and negative interaction energies must not be mixed without an explicit sign conversion.',
        'The shared `models/update-3351/` contains an inference-only `parameters.npz` whose parameter bytes exactly match the frozen checkpoint, plus its original parameter/source identity, network configuration, fixed exponent candidate and 500 training / 26 calibration reference structures. `snapshot.json` is the original record; use the derived `inference-snapshot.json` with the smaller export. The original 390-MB training state stays at its recorded source; optimizer moments and saved training gradients are not needed for inference. This is evaluated update 3351, not a claimed final or best epoch-596 model.',
        '', '## Reproducibility','',
        '`catalog.json` maps cases and evidence boundaries. `structure-index.csv`/`.json` maps every generated XYZ to original records, including molecular charge and 2S where available. `FILE_MANIFEST.json` maps imported files to original local/remote paths and hashes. `SHA256SUMS` covers imported and generated payloads.',
        'Frozen per-atom basis exports allow independent same-basis recomputation. Full MACE inference or GradSCF driver replay additionally requires the recorded source/environment; original absolute paths are intentionally retained in raw records. The import itself performs file/schema/checksum checks only.',
        'Large primitive-integral caches, JIT caches, opaque duplicate archives and oversized logs are omitted and listed in `EXCLUDED_FILES.json`. Coefficient checkpoints, optimization histories, radial arrays, explicit failure records and independent-check reports are retained. No full four-index ERI cache is added.',
        '', '## Ongoing work','',
        '`contexts/ongoing-training/` is a dated snapshot of the remaining eight-family campaign and its source/configuration hashes. Its queued preparation/training jobs are not completed calculations. `contexts/basis-design/` preserves current structural definitions separately from the historical cases.',
        '', 'Original-language Markdown/log notes under `artifacts/` are source evidence, not manuscript prose. English case READMEs provide the handoff summaries.','']
    (out/'README.md').write_text('\n'.join(lines))
    copy(Path(__file__).resolve(),Path('_tools/assemble_examples.py'),case='archive-tool')
    payload=[]
    origins={r['path']:r for r in imported}
    for p in sorted(out.rglob('*')):
        if not p.is_file():continue
        rel=str(p.relative_to(out)); entry=origins.get(rel,dict(path=rel,source='generated during data-only import'))
        payload.append(dict(entry,sha256=sha(p),bytes=p.stat().st_size))
    dump(out/'FILE_MANIFEST.json',dict(created_at=datetime.now(timezone.utc).isoformat(),files=payload))
    (out/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.relative_to(out).as_posix()+'\n'
        for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    print(json.dumps(dict(cases=len(case_rows),structures=len(geometries),imported_files=len(imported),
        payload_files=len(payload)+2,bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())),indent=2))


if __name__=='__main__':main()
