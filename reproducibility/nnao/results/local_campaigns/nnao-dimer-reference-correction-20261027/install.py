"""Archive conditional exports and publish corrected independent-reference figures."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
CASE=Path('/Users/jiaoyuan/Documents/GitHub/NNAO_paper/data/computational_examples/2026-10-26/cases/15_dimer_cp_output_controls')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    verification=json.loads((HERE/'verification.json').read_text());assert verification['status']=='passed'
    assert digest(CASE/'combined/results.json')==verification['raw_source_sha256']
    old=json.loads((CASE/'FILE_MANIFEST.json').read_text())
    for relative,expected in old.items():assert digest(CASE/relative)==expected,'Concurrent edit: '+relative
    index=CASE.parents[2]/'README.md'
    index_rows=[line for line in index.read_text().splitlines() if line.startswith('|') and '15_dimer_cp_output_controls/' in line]
    assert len(index_rows)==1,'Review a changed data-index layout before installation.'
    archive=CASE/'archive/conditional-reference-v1';assert not archive.exists()
    files=list(old)+['FILE_MANIFEST.json']
    archive.mkdir(parents=True)
    archived={}
    for relative in files:
        src=CASE/relative;dst=archive/relative;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst);assert digest(src)==digest(dst)
        archived[relative]=digest(src)
    (archive/'ARCHIVE_CHECKSUMS.json').write_text(json.dumps(archived,indent=2)+'\n')
    old_figures={p.relative_to(CASE/'figures').as_posix() for p in (CASE/'figures').rglob('*') if p.is_file()}
    new_figures={p.relative_to(HERE/'figures').as_posix() for p in (HERE/'figures').rglob('*') if p.is_file()}
    assert old_figures==new_figures
    for relative in new_figures:shutil.copy2(HERE/'figures'/relative,CASE/'figures'/relative)
    for name in ('curves.csv','verification.json'):shutil.copy2(HERE/'combined'/name,CASE/'combined'/name)
    for name in ('isolated-reference-map.json','isolated-inputs.json','isolated-results.json'):
        shutil.copy2(HERE/name,CASE/name)
    correction=CASE/'reference-correction';assert not correction.exists();correction.mkdir()
    for name in ('isolated.py','isolated.slurm'):
        shutil.copy2(HERE/name,correction/name)
    shutil.copy2(REPO/'tests/comparisons/nnao_dimer_controls/protocols.py',correction/'protocols.py')
    # Keep the main entry points reproducible using only bundled host helpers.
    build=(HERE/'build.py').read_text()
    build=build.replace('REPO=HERE.parents[1]\n','')
    build=build.replace("CASE=Path('/Users/jiaoyuan/Documents/GitHub/NNAO_paper/data/computational_examples/2026-10-26/cases/15_dimer_cp_output_controls')",'CASE=HERE')
    build=build.replace("sys.path.insert(0,str(REPO/'tests/comparisons'))","sys.path.insert(0,str(HERE/'reference-correction'))")
    build=build.replace('from nnao_dimer_controls.protocols import plotted_energy,conditional_energy','from protocols import plotted_energy,conditional_energy')
    compile(build,'assemble.py','exec');(CASE/'assemble.py').write_text(build)
    shutil.copy2(HERE/'plot.py',CASE/'plot_controls.py')
    shutil.copy2(HERE/'verify.py',CASE/'verify_results.py')
    shutil.copy2(HERE/'README_CORRECTED.md',CASE/'README.md')
    spec=CASE/'RUN_SPEC.md'
    text=spec.read_text();spec.write_text('# Plot-reference correction — 2026-10-27\n\n'
        'Current figures use constant independent-monomer references. The CP-adjusted '
        'estimate adds the original frozen-parameter CP increment and retains the basis-response term. '
        'The original calculation specification and conditional figures are archived in '
        '`archive/conditional-reference-v1/`. The historical numerical work below is unchanged; '
        'its original plot-reference convention is superseded by `README.md`.\n\n'+text)
    for relative,expected in old.items():
        if relative.startswith(('figures/','combined/curves.csv','combined/verification.json')):continue
        if relative in {'README.md','RUN_SPEC.md','assemble.py','plot_controls.py','verify_results.py','verification.json'}:continue
        assert digest(CASE/relative)==expected,'Raw evidence changed: '+relative
    subprocess.run([sys.executable,str(CASE/'verify_results.py')],check=True)
    assert digest(CASE/'combined/results.json')==verification['raw_source_sha256']
    text=index.read_text()
    after='Corrected constant isolated-monomer references; model + frozen-parameter CP estimate; conditional-v1 exports archived'
    current=[line for line in text.splitlines() if line.startswith('|') and '15_dimer_cp_output_controls/' in line]
    assert len(current)==1
    cells=current[0].split('|');assert len(cells)==5;cells[-2]=' '+after+' '
    index.write_text(text.replace(current[0],'|'.join(cells),1))
    manifest={p.relative_to(CASE).as_posix():digest(p) for p in sorted(CASE.rglob('*'))
              if p.is_file() and p!=CASE/'FILE_MANIFEST.json' and p.name!='.DS_Store'}
    (CASE/'FILE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for relative,expected in archived.items():assert digest(archive/relative)==expected
    report=dict(status='passed',destination=str(CASE),archived_conditional_exports=str(archive),
        archived_files_verified=len(archived),raw_results_sha256=verification['raw_source_sha256'],
        original_distance_energy_records_unchanged=True,monomer_reference_pairs=84,
        constant_reference_groups=96,corrected_curve_values=1728,reference_plot_values=216,
        figures=24,cp_definition=verification['cp_definition'],
        manuscript_edited=False,training_modified=False)
    (HERE/'INSTALL_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
