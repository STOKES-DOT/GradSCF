"""CPU abstract memory analysis of cached-integral RHF forward only.

Compiles the real evaluator and its SCF diagnostics without value_and_grad.
Never executes SCF or creates S/H/DF numerical input arrays.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import runpy
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
helper_path = ROOT / 'artifacts/nnao-capacity-20261008/estimate_capacity.py'
spec = importlib.util.spec_from_file_location('nnao_capacity_helper', helper_path)
helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
jax, jnp, np = helper.jax, helper.jnp, helper.np


def inputs(selection):
    rows = {row['structure_id']: row for row in helper.cases(selection)}
    symbols = ['C'] * 33 + ['H'] * 68
    rows['fixture_C33H68_101atoms'] = dict(structure_id='fixture_C33H68_101atoms',
        symbols=symbols, charge=0, spin=0, kind='synthetic_separated_coordinate_size_fixture',
        coords_angstrom=[[3.*(i%10), 3.*((i//10)%10), 3.*(i//100)] for i in range(101)])
    # Start with two real baselines and the most relevant matched larger cases.
    return [rows[name] for name in ('S22/07','ADIM6/AD6','fixture_C26H54_80atoms',
        'fixture_C17H18N12O3_50atoms','fixture_C19H40_59atoms',
        'fixture_2xS22_07_60atoms','fixture_C33H68_101atoms')]


def estimate(row, experiment_class):
    tick=time.perf_counter()
    record=dict(structure_id=row['structure_id'], kind=row['kind'], natoms=len(row['symbols']),
        composition=dict(Counter(row['symbols'])), status='compiling',
        native_integral_evaluations=0, scf_executions=0, backward_executions=0)
    try:
        atom=list(zip(row['symbols'],row['coords_angstrom']))
        ex=experiment_class.__new__(experiment_class)
        ex.layout=helper.prepare_direct_basis(atom,unit='Angstrom',basis_family='szp663_direct',core_primitives=6)
        primitive,_=helper.primitive_basis(ex.layout.topology,ex.layout.parameters)
        auxiliary,_=helper.integrals.prepare_basis(atom,'def2-universal-jkfit',cart=False)
        ex.jk_backend='df';ex.nelectron=sum(ex.layout.topology.nuclear_charges)
        ex.implicit_tolerance=1e-8;ex.scf_rescue_level_shift=.2
        ex.enuc=jnp.asarray(0.,dtype=jnp.float64)
        p,n,q=primitive.nao,ex.layout.topology.nao,auxiliary.nao
        pairs=p*(p+1)//2
        shapes=(jax.ShapeDtypeStruct(ex.layout.reference_outputs().shape,jnp.float64),
                jax.ShapeDtypeStruct((p,p),jnp.float64),jax.ShapeDtypeStruct((p,p),jnp.float64),
                jax.ShapeDtypeStruct((q,pairs),jnp.float64))
        evaluator=jax.jit(ex._implicit_value)
        compiled=evaluator.lower(*shapes).compile()
        stats=compiled.memory_analysis()
        memory={key:int(getattr(stats,key)) for key in ('argument_size_in_bytes',
            'temp_size_in_bytes','output_size_in_bytes','alias_size_in_bytes','generated_code_size_in_bytes')}
        total=memory['argument_size_in_bytes']+memory['temp_size_in_bytes']+memory['output_size_in_bytes']-memory['alias_size_in_bytes']
        record.update(status='compiled',primitive_nao=p,contracted_nao=n,
            auxiliary_rank_assuming_full_metric_rank=q,nelectron=ex.nelectron,
            packed_df_bytes=8*q*pairs,metric_bytes=8*q*q,
            first_df_generation_at_least_two_three_center_payloads_bytes=16*q*pairs,
            memory_analysis_bytes=memory,arguments_plus_temporary_minus_alias_bytes=total,
            estimated_arguments_plus_temporary_gib=total/2**30)
        del compiled,evaluator,ex,shapes
    except Exception as error:
        record.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally:
        jax.clear_caches();gc.collect()
        record['compile_elapsed_seconds']=time.perf_counter()-tick
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record['process_peak_rss_gib']=rss/(2**30 if platform.system()=='Darwin' else 2**20)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,default=ROOT/'artifacts/nnao-stage2-20261007/selection/all.jsonl')
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('results.json'))
    parser.add_argument('--cases',nargs='+')
    parser.add_argument('--append',action='store_true')
    args=parser.parse_args();rows=inputs(args.selection)
    if args.cases:
        unknown=set(args.cases)-{row['structure_id'] for row in rows}
        if unknown:parser.error(f'Unknown cases: {sorted(unknown)}')
        rows=[row for row in rows if row['structure_id'] in args.cases]
    source=ROOT/'tools/optimize_methane_nnao.py'
    experiment_class=runpy.run_path(str(source))['MethaneRHF']
    report=dict(started_utc=datetime.now(timezone.utc).isoformat(),method='RHF-DF cached-integral forward only',
        jax_version=jax.__version__,backend=jax.default_backend(),dtype='float64',
        platform=platform.platform(),python_version=platform.python_version(),
        basis='szp663_direct/core6 spherical',implicit_tolerance=1e-8,scf_rescue_level_shift=.2,
        source_sha256={str(path.relative_to(ROOT)):helper.digest(path) for path in (
            source,ROOT/'src/gradscf/integrals/density_fitting.py',ROOT/'src/gradscf/solvers/nonlinear/fixed_point.py')},
        script_sha256=helper.digest(__file__),selection_sha256=helper.digest(args.selection),
        limitations=['CPU abstract compilation, not CPU runtime RSS or GPU peak.',
            'No native integrals, SCF execution, backward execution or giant numerical input arrays.',
            'Fixed S/H/whitened packed DF already available; first cache generation is excluded.',
            'Full auxiliary metric rank assumed.',
            'Synthetic fixture coordinates are not chemical geometries or physical results.',
            'mmap and CPU zero-copy/alias behavior do not guarantee that actual RSS equals this estimate.'],results=[])
    if args.append and args.output.exists():
        previous=json.loads(args.output.read_text())
        for key in ('method','jax_version','backend','dtype','source_sha256','selection_sha256'):
            if previous[key]!=report[key]:raise ValueError(f'Cannot append after {key} changed.')
        report=previous
        done={row['structure_id'] for row in report['results'] if row['status']=='compiled'}
        rows=[row for row in rows if row['structure_id'] not in done]
    backward=json.loads((ROOT/'artifacts/nnao-capacity-20261008/results.json').read_text())
    matched={row['structure_id']:row for row in backward['results'] if row['status']=='compiled'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for row in rows:
        print(f"Compiling abstract forward {row['structure_id']}",flush=True)
        result=estimate(row,experiment_class)
        if result['status']=='compiled' and row['structure_id'] in matched:
            previous=matched[row['structure_id']]
            result['matched_cpu_value_and_grad_gib']=previous['estimated_arguments_plus_temporary_gib']
            result['matched_backward_to_forward_ratio']=previous['estimated_arguments_plus_temporary_gib']/result['estimated_arguments_plus_temporary_gib']
        report['results'].append(result);report['updated_utc']=datetime.now(timezone.utc).isoformat()
        helper.atomic_json(args.output,report)
        print(json.dumps({key:result.get(key) for key in ('structure_id','status','primitive_nao',
            'contracted_nao','estimated_arguments_plus_temporary_gib','compile_elapsed_seconds','error')}),flush=True)
    return int(any(row['status']=='failed' for row in report['results']))


if __name__=='__main__':
    raise SystemExit(main())
