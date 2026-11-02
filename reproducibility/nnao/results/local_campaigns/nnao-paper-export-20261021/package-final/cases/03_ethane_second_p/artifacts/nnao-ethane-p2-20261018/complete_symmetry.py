"""Complete equivalent ethane angles by exact atom mapping and checked SCF/AD."""
from pathlib import Path
from datetime import datetime
import gc
import json
import os
import time
import jax
import numpy as np

from gradscf.model.nnao.basis import prepare_direct_basis
from gradscf.model.nnao.initialization import bind_pool_candidate
from gradscf.model.nnao.integrals.fixed_pool import prepare_fixed_pool
from gradscf.model.nnao.training.initial_guess import atomic_guess_data
from gradscf.model.nnao.training.pool import FreeContractionRHF
from gradscf.model.nnao.runtime import atomic_json,file_hash,repository_root,scientific_source_hash
from expansion import expand_carbon_p
from symmetry import mirror_atom_map

OUT=Path(__file__).resolve().parent
BASE=OUT.parent/'nnao-ethane-torsion-20261018'
CHECKPOINT=OUT.parent/'nnao-relative500-20261018'


def main():
    assert jax.config.jax_enable_x64 and jax.default_backend()=='gpu'
    config=json.loads((OUT/'configuration.json').read_text())
    snapshot=config['checkpoint']
    assert scientific_source_hash(repository_root())==snapshot['source_sha256']
    assert file_hash(CHECKPOINT/'checkpoint-current.npz')==snapshot['checkpoint_sha256']
    with np.load(CHECKPOINT/'checkpoint-current.npz',allow_pickle=False) as saved:
        identity=json.loads(saved['metadata'].item())['identity']
    rows=json.loads((BASE/'structures.json').read_text())
    lookup={int(r['angle_deg']):r for r in rows}
    baseline={r['angle_deg']:r for r in json.loads((BASE/'scan-summary.json').read_text())['points']}
    transition=json.loads((OUT/'symmetry-transition.json').read_text())
    config.update(completion_strategy='Independent optimization through 60 degrees; exact mirror atom '
        'mapping for unfinished equivalent angles, followed by checked SCF and full coefficient gradient.',
        completion_script_sha256=file_hash(__file__),symmetry_sha256=file_hash(OUT/'symmetry.py'))
    atomic_json(OUT/'configuration-completed.json',config)
    cpu,gpu=jax.devices('cpu')[0],jax.devices('gpu')[0]
    started=time.perf_counter()
    scan=dict(status='running',job_id=os.environ.get('SLURM_JOB_ID'),previous_job_id=transition['job_id'],
        device=gpu.device_kind,cpus=os.environ.get('SLURM_CPUS_PER_TASK'),configuration=config,points=[])
    for row in rows:
        angle=int(row['angle_deg']);folder=OUT/f'angle-{angle:03d}'
        scan.update(phase=angle,elapsed_seconds=transition['elapsed_seconds']+time.perf_counter()-started)
        atomic_json(OUT/'progress.json',scan)
        existing=folder/'summary.json'
        if existing.exists():
            report=json.loads(existing.read_text())
            if report['status']!='complete' or not report['best_gradient_converged']:
                raise RuntimeError(f'Existing completed endpoint is invalid: {angle}.')
            report['optimization_mode']='independent_two_start_optimization'
        else:
            if angle<=60:raise RuntimeError(f'Missing independently optimized source angle: {angle}.')
            if folder.exists():
                archive=OUT/'interrupted-at-symmetry-switch';archive.mkdir(exist_ok=True)
                folder.rename(archive/folder.name)
            folder.mkdir()
            atomic_json(folder/'structure.json',row)
            source_angle=120-angle;source_folder=OUT/f'angle-{source_angle:03d}'
            source=json.loads((source_folder/'summary.json').read_text())
            assert source['status']=='complete' and source['best_gradient_converged']
            permutation,geometry_error=mirror_atom_map(lookup[source_angle],row)
            with np.load(source_folder/'best-coefficients.npz',allow_pickle=False) as saved:
                coefficients=np.asarray(saved['coefficients'])[permutation]
                expected_gradient=np.asarray(saved['gradient'])[permutation]
            tick=time.perf_counter()
            with jax.default_device(cpu):
                lower=bind_pool_candidate(prepare_direct_basis(list(zip(row['symbols'],row['coords_angstrom'])),
                    basis_family=identity['references']['basis_family'],
                    core_primitives=identity['references']['core_primitives'],cart=False),identity['candidate'])
                expanded=expand_carbon_p(lower,lower.reference_outputs())
            problem,manifest=prepare_fixed_pool(row,lower,BASE/folder.name/'cache'/'pool.npz',
                storage='device',auxbasis=identity['auxbasis'],implicit_tolerance=1e-8)
            free=FreeContractionRHF(expanded,problem.raw,problem.enuc,problem.value,problem.gradient,
                row['structure_id'],problem.diagnostics,atomic_guess=atomic_guess_data(expanded))
            energy,gradient,info=free.evaluate_device(jax.device_put(coefficients,gpu),gradient=True)
            gradient=np.asarray(gradient)
            energy_error=abs(energy-source['best_energy_hartree'])
            gradient_error=float(np.max(np.abs(gradient-expected_gradient)))
            norm=float(np.linalg.norm(gradient))
            assert energy_error<=1e-8 and gradient_error<=2e-7 and norm<=1e-5,(angle,energy_error,gradient_error,norm)
            assert energy<=baseline[angle]['free_energy_hartree']+1e-8
            np.savez(folder/'best-coefficients.npz',coefficients=coefficients,gradient=gradient)
            with jax.default_device(cpu):
                shells=expanded.atom_shells(expanded.bind(jax.device_put(coefficients,cpu)))
            atomic_json(folder/'frozen-basis.json',dict(structure_id=row['structure_id'],
                basis='NNAO / second C p / free',symbols=row['symbols'],coords_angstrom=row['coords_angstrom'],
                atom_labels=[f'{s}{i}' for i,s in enumerate(row['symbols'])],atom_shells=shells,nao=50,
                native_energy_hartree=energy))
            report=dict(status='complete',optimization_mode='exact_symmetry_transfer_with_scf_and_gradient_check',
                angle_deg=row['angle_deg'],structure_id=row['structure_id'],nao=50,pool_nao=152,
                source_angle_deg=source_angle,atom_permutation=permutation.tolist(),
                mapping_geometry_error_angstrom=geometry_error,mapped_energy_error_hartree=energy_error,
                mapped_gradient_max_error=gradient_error,old_energy_hartree=baseline[angle]['free_energy_hartree'],
                best_energy_hartree=energy,best_gradient_norm=norm,best_gradient_converged=True,best_scf=info,
                energy_lowering_hartree=baseline[angle]['free_energy_hartree']-energy,
                energy_lowering_kcal_mol=627.5094740631*(baseline[angle]['free_energy_hartree']-energy),
                converged_starts=0,starts=[],source_converged_starts=source['converged_starts'],
                endpoint_spread_hartree=None,inherited_source_start_spread_hartree=source['endpoint_spread_hartree'],
                elapsed_seconds=time.perf_counter()-tick,primitive_cache_reused=True,cache_signature=manifest['signature'])
            atomic_json(folder/'summary.json',report)
            atomic_json(folder/'progress.json',report)
        record={key:report[key] for key in ['angle_deg','structure_id','nao','old_energy_hartree',
            'best_energy_hartree','best_gradient_norm','best_gradient_converged','energy_lowering_hartree',
            'energy_lowering_kcal_mol','converged_starts','endpoint_spread_hartree','status','elapsed_seconds',
            'optimization_mode']}
        if 'source_angle_deg' in report:record['source_angle_deg']=report['source_angle_deg']
        scan['points'].append(record)
        scan['elapsed_seconds']=transition['elapsed_seconds']+time.perf_counter()-started
        atomic_json(OUT/'progress.json',scan)
        print('COMPLETED',json.dumps(record),flush=True)
        jax.clear_caches();gc.collect()
    scan.update(status='complete',phase='complete',elapsed_seconds=transition['elapsed_seconds']+time.perf_counter()-started)
    atomic_json(OUT/'summary.json',scan);atomic_json(OUT/'progress.json',scan)
    assert file_hash(CHECKPOINT/'checkpoint-current.npz')==snapshot['checkpoint_sha256']


if __name__=='__main__':main()
