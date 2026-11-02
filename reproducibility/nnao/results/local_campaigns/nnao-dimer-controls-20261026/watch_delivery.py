"""Finite monitor: fetch verified controls and add a data-only paper supplement."""
import csv
import fcntl
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import zipfile
import shutil

from pypdf import PdfReader

HERE=Path(__file__).resolve().parent
REMOTE='/home/user_yuanjiao/private/homefile/GradSCF_mace500_20261014/artifacts/nnao-dimer-controls-20261026'
PYTHON='/home/user_yuanjiao/private/homefile/conda-envs/jax_scf/bin/python'
PAPER=Path('/Users/jiaoyuan/Documents/GitHub/NNAO_paper')
DEST=PAPER/'data/computational_examples/2026-10-26/cases/15_dimer_cp_output_controls'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def record(value):
    value['checked_at']=datetime.now(timezone.utc).isoformat()
    (HERE/'delivery-progress.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value),flush=True)


def remote(code):
    command=PYTHON+' -c '+shlex.quote(code)
    answer=subprocess.run(['ssh','-o','ConnectTimeout=10','-o','ConnectionAttempts=1',
        '-o','BatchMode=yes','jy_gpu_2',command],text=True,capture_output=True,timeout=30)
    if answer.returncode:raise RuntimeError(answer.stderr[-1000:])
    return json.loads(answer.stdout)


def state():
    return remote('''import json,subprocess
from pathlib import Path
p=Path('''+repr(REMOTE)+''')
q=p/'verification.json';r=p/'run/results.json'
report=json.loads(r.read_text()) if r.exists() else {}
v=json.loads(q.read_text()) if q.exists() else {}
submission=json.loads((p/'submitted.json').read_text())
ids={str(job['job_id']) for job in submission['jobs'] if job['profile'] in ('full','continuation','figures')}
live=subprocess.run(['squeue','-u','user_yuanjiao','-h','-o','%i|%T|%R'],capture_output=True,text=True)
matched='\\n'.join(line.strip() for line in live.stdout.splitlines() if line.split('|')[0].strip() in ids)
print(json.dumps(dict(verified=v.get('status')=='passed',status=report.get('status'),
    completed=len(report.get('results',[])),passed=report.get('passed',0),failed=report.get('failed',0),
    live_jobs=matched)))
''')


def archive():
    return remote('''import json,zipfile,hashlib
from pathlib import Path
p=Path('''+repr(REMOTE)+''')
assert json.loads((p/'verification.json').read_text())['status']=='passed'
target=p/'verified-controls.zip'
paths=[]
for name in ('inputs','combined','figures','representative-structures'):
 paths += [x for x in (p/name).rglob('*') if x.is_file()]
paths += [x for x in (p/'run').iterdir() if x.is_file() and x.suffix=='.json']
paths += [p/name for name in ('verification.json','submitted.json','RUN_SPEC.md','assemble.py',
    'plot_controls.py','verify_results.py','measure.py','smoke.slurm','full.slurm','finalize.slurm')]
if (p/'continuation.slurm').exists():paths.append(p/'continuation.slurm')
if (p/'finish.slurm').exists():paths.append(p/'finish.slurm')
with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as out:
 for path in paths:out.write(path,path.relative_to(p).as_posix())
 for name in ('__init__.py','protocols.py','run.py'):
  path=p.parents[1]/'tests/comparisons/nnao_dimer_controls'/name
  out.write(path,'drivers/'+name)
print(json.dumps(dict(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),files=len(paths)+3)))
''')


def install(folder):
    assert not DEST.exists(),'Do not overwrite an existing paper supplement.'
    report=json.loads((folder/'verification.json').read_text());assert report['status']=='passed'
    subprocess.run([sys.executable,str(folder/'verify_results.py')],check=True)
    for mode in ('no-cp','cp'):
        assert len(PdfReader(folder/f'figures/dimers-{mode}-with-structures.pdf').pages)==12
        assert len(list((folder/'figures'/mode).glob('*.png')))==12
    before={name:digest(PAPER/name) for name in ('manuscript/main.tex',)}
    shutil.copytree(folder,DEST)
    (DEST/'README.md').write_text('''# Dimer interaction controls: CP × MACE output

[Two-group figure gallery](figures/index.html).
[No-CP 12-page PDF](figures/dimers-no-cp-with-structures.pdf).
[CP-corrected 12-page PDF](figures/dimers-cp-with-structures.pdf).

Historical checkpoint update 3351; DF-RHF / def2-universal-jkfit, spherical AOs,
float64. All 12 representative dimers and 108 geometries are unchanged.
Fixed MACE output uses the coefficients predicted at distance scale 1.00 for
the entire scan; adaptive output uses the archived prediction at each geometry.
Network weights and primitive exponents remain fixed in both policies.
At each geometry the selected coefficients are identical in all five CP
calculations, sharing the dimer auxiliary metric. Electronic SCF relaxes anew.

The no-CP panel uses E_AB - E_A(own) - E_B(own); the CP panel uses
E_AB - E_A(ghost) - E_B(ghost). Independent-monomer model interaction energies
are excluded from this matched comparison. Axes use dimensionless source
distance factors and kcal/mol, with equal y limits for both panels of a dimer.

Controls: STO-6G, STO-6G+P, 6-31G(d,p), def2-SVP, cc-pVDZ and cc-pVTZ.
STO-6G+P adds the most diffuse original uncontracted def2-SVP p(H)/d(heavy)
shell. Explicit assets preserve reference exponents, including Br d = 0.389.
Correlated-source reference curves also contain electronic-method differences.
The source benzene TS label denotes the T-shaped dimer, not a transition state.

There are 204 new five-energy cases and 660 reused records, including twelve
identical fixed/adaptive anchors: 864 records, 1728 calculated curve values,
plus 216 displayed reference values. Four new cases were checked first on CPU;
the continuation records its hardware per case. Native SCF/stationarity and
real internal stability were checked for all five energies of each new case.
Reused records retain their original validation scope; global HF minima and
external spin/complex stability are not certified.

- `combined/curves.csv`: exact CP/no-CP values and source fields.
- `combined/results.json`: all merged records with original diagnostics.
- `run/results.json`: unchanged new calculations, with per-case hardware/time.
- `inputs/cases.json`: full input geometries, frozen GTOs, anchor identities and
  source/checkpoint hashes. `inputs/polarization-assets.json`: STO-6G+P assets.
- `representative-structures/`: existing reference-geometry molecular renders.
- `verification.json`: exact arithmetic, anchor and plot-value checks.
- `RUN_SPEC.md`, `drivers/`, resource reports and Slurm scripts: reproduction
  evidence; numerical drivers require the recorded frozen GradSCF environment.

No finite differences or full four-index ERI were used. Original benchmark
archives, existing paper figures/scripts, manuscript text and training jobs
were not modified by this handoff. Generated figures received numerical and
layout-bound checks; a separate model visual review is not claimed here.
''')
    manifest={p.relative_to(DEST).as_posix():digest(p) for p in sorted(DEST.rglob('*')) if p.is_file()}
    (DEST/'FILE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    index=PAPER/'data/computational_examples/README.md'
    text=index.read_text();anchor='|---|---|---|\n';assert text.count(anchor)==1
    row='| Dimer CP × fixed/adaptive MACE output, Case 15 | [New controls](2026-10-26/cases/15_dimer_cp_output_controls/README.md) · [two-group figures](2026-10-26/cases/15_dimer_cp_output_controls/figures/index.html) | Matched own/ghost energies; STO-6G and STO-6G+P included; original independent-monomer curves preserved |\n'
    assert '15_dimer_cp_output_controls' not in text
    index.write_text(text.replace(anchor,anchor+row))
    unchanged={name:digest(PAPER/name)==value for name,value in before.items()}
    return dict(status='installed',destination=str(DEST),verification=report,
        copied_files=len(manifest)+1,manuscript_unchanged_during_install=unchanged,
        training_state_unchanged=True,visual_review_by_assistant=False)


def main():
    lock=(HERE/'delivery.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    (HERE/'watcher.pid').write_text(str(os.getpid())+'\n')
    deadline=time.monotonic()+8*3600
    while time.monotonic()<deadline:
        try:current=state()
        except Exception as error:
            record(dict(status='connection_retry',error=str(error)));time.sleep(45);continue
        record(dict(status='waiting_for_verified_figures',remote=current))
        if current['verified']:break
        live=current['live_jobs'].splitlines()
        if not live or all('DependencyNeverSatisfied' in line for line in live):
            record(dict(status='needs_review',remote=current));return
        time.sleep(45)
    else:
        record(dict(status='monitor_deadline_reached',remote=state()));return
    package=archive();download=HERE/'download';download.mkdir(exist_ok=True)
    target=download/'verified-controls.zip'
    subprocess.run(['scp','jy_gpu_2:'+package['path'],str(target)],check=True,timeout=180)
    assert digest(target)==package['sha256']
    folder=download/('unpacked-'+str(int(time.time())));folder.mkdir()
    with zipfile.ZipFile(target) as source:
        for member in source.infolist():
            path=Path(member.filename)
            assert not path.is_absolute() and '..' not in path.parts
        source.extractall(folder)
    result=install(folder);result['archive_sha256']=package['sha256']
    (HERE/'INSTALL_REPORT.json').write_text(json.dumps(result,indent=2)+'\n');record(result)


if __name__=='__main__':
    try:main()
    except Exception as error:
        record(dict(status='needs_review',error=str(error)));raise
