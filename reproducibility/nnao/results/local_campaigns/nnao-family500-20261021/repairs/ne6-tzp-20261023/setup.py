"""Reuse the validated stability source and preserve the failed TZP state."""
import importlib.util
import json
from pathlib import Path


ROOT=Path('/home/user_yuanjiao/private/homefile/GradSCF_family500_20261021/artifacts/nnao-family500-20261021')
HERE=ROOT/'repairs/ne6-tzp-20261023'
PREVIOUS=ROOT/'repairs/ne6-20261023'


def main():
    assert not (HERE/'deployment.json').exists()
    previous=json.loads((PREVIOUS/'deployment.json').read_text())
    spec=importlib.util.spec_from_file_location('ne6_deploy',PREVIOUS/'deploy.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.HERE=HERE
    assert module.source_hash(module.ROOT)==previous['old_source_sha256']
    source=Path(previous['source_directory'])
    assert module.source_hash(source)==previous['new_source_sha256']
    (HERE/'source').symlink_to(source,target_is_directory=True)
    migrated=module.migrate_checkpoint('nnao_tzp_s',previous['old_source_sha256'],
                                       previous['new_source_sha256'],435,5)
    report=dict(status='prepared',old_source_sha256=previous['old_source_sha256'],
        new_source_sha256=previous['new_source_sha256'],source_directory=str(source),
        changed_files=previous['changed_files'],checkpoints={'nnao_tzp_s':migrated},
        stability_tolerance=1e-7,finite_differences=False,
        original_failure=dict(structure_id='GMTKN55-v2/RG18/ne6',
            energy_hartree=-769.7227147810721,minimum_curvature=9.803122649688527,
            eigen_residual=7.37848996556231e-7))
    (HERE/'deployment.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
