"""Real RHF/RKS direct SCF stage profiling plus same-snapshot J/K diagnostics.

Use profile_followup's CLI, e.g. CONFIG OUT --mode diagnostic --fixed-cycles 5
--budget-seconds 1200. Kernel timings are instrumented, shared-input work, not
an SCF baseline or additive SCF stages. No tensor is serialized.
"""
from functools import partial
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
import profile_followup
import profile_kernels


def run_kernels(inputs, result, cfg, output, budget_seconds):
    import jax
    import numpy as np
    from gradscf.dft.xc import hybrid_coeff, xc_type
    from gradscf.integrals.grids.blocks import AOGrid, _materialize_scf_grid
    from gradscf.integrals.backends.native.jk import _selected_jk
    from gradscf.scf.rsh import rsh_preset, _direct_rsh_jk

    deadline, rows, output = time.perf_counter()+max(0., budget_seconds), [], Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    def emit(**fields):
        row = dict(scope='shared-input/kernel; diagnostic snapshot, not complete SCF',
                   instrumented=True, remaining_seconds=max(0., deadline-time.perf_counter()), **fields)
        with output.open('a') as stream: stream.write(json.dumps(row, allow_nan=False)+'\n')
        rows.append(row)
    provider = getattr(inputs, 'direct_basis', None)
    if cfg['representation'] != 'direct' or cfg['method'] not in ('RHF', 'RKS') or provider is None:
        emit(event='kernels', status='not_applicable', reason='Requires captured RHF/RKS direct inputs')
        return rows
    density = result.density_matrix
    if density.shape != inputs.overlap.shape or provider.nao != density.shape[0]:
        emit(event='kernels', status='error', reason='Captured molecular input/result dimensions differ')
        return rows
    xc = 'hf' if cfg['method'] == 'RHF' else cfg['xc']
    preset, alpha = rsh_preset(xc), float(hybrid_coeff(xc))
    need_k = preset is not None or abs(alpha) > 1e-14
    controls = dict(screening_threshold=float(cfg['direct_scf_tol']), workers=int(cfg['threads']))
    runtime_path = output.parent/'runtime.json'
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    emit(event='kernel_scope', status='ok', xc=xc, nao=provider.nao,
         snapshot_converged=bool(result.converged), snapshot_cycles=int(result.cycles),
         controls=controls, runtime=str(runtime_path), adapter_sha256=profile_followup.profile_stages.digest(__file__),
         provenance={key:runtime.get(key) for key in ('module','native','native_sha256','source_snapshot','probe_sha256')},
         exchange_convention='Hybrid/RSH weights included in K; restricted Fock factor 1/2 not applied',
         grid_use='No grids in these J/K products; full SCF remains unchanged')
    def action(bound, dm, *, with_j, with_k):
        if preset is not None:
            return _direct_rsh_jk(bound, dm, preset.to_range_separated_hybrid_coefficients(),
                                  with_j=with_j, with_k=with_k, **controls)
        j, k = _selected_jk(bound, dm, with_j=with_j, with_k=with_k, **controls)
        return j, alpha*k
    values = {}
    for name, with_j, with_k in [('j_only',True,False),('k_only',False,True),('combined',True,need_k)]:
        if name == 'k_only' and not need_k:
            emit(event='kernel', name=name, status='not_applicable', reason='No exact exchange in this functional')
            continue
        if deadline-time.perf_counter() < (45. if not with_k else 90.):
            emit(event='kernel', name=name, status='skipped_budget')
            continue
        row = dict(event='kernel', name=name, all_array_arguments_dynamic=True,
                   inputs=[dict(shape=list(a.shape),dtype=str(a.dtype),bytes=int(a.nbytes))
                           for a in jax.tree_util.tree_leaves((provider,density)) if hasattr(a,'shape')])
        phase = 'lower'
        emit(**row, status='started')
        try:
            jax.block_until_ready((provider, density))
            t = time.perf_counter(); lowered = jax.jit(partial(action,with_j=with_j,with_k=with_k)).lower(provider,density)
            row['lower_seconds'] = time.perf_counter()-t
            phase = 'compile'; t = time.perf_counter(); executable = lowered.compile()
            row['compile_seconds'] = time.perf_counter()-t
            try:
                analysis = executable.memory_analysis()
                row['compiler_memory_not_rss'] = None if analysis is None else {key:getattr(analysis,key,None)
                    for key in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')}
            except (AttributeError,RuntimeError,NotImplementedError): row['compiler_memory_not_rss'] = None
            row['memory_note'] = 'XLA buffer estimate only; native opaque scratch and process RSS are separate'
            if time.perf_counter() >= deadline:
                row.update(status='skipped_budget', phase='first_execute'); emit(**row); continue
            phase = 'first_execute'; t = time.perf_counter(); value = jax.block_until_ready(executable(provider,density))
            row['first_execute_seconds'] = time.perf_counter()-t
            samples, phase = [], 'warm_execute'
            for _ in range(3):
                if deadline-time.perf_counter() < max(1., row['first_execute_seconds']*1.1): break
                t = time.perf_counter(); value = jax.block_until_ready(executable(provider,density))
                samples.append(time.perf_counter()-t)
            values[name] = value
            row.update(warm_seconds=samples, warm_median_seconds=float(np.median(samples)) if samples else None,
                       status='ok' if len(samples)==3 else 'partial_budget',
                       matrix_norms=[float(np.linalg.norm(np.asarray(a))) for a in value])
            emit(**row)
        except Exception as error:
            row.update(status='error', phase=phase, error=repr(error)); emit(**row)
    checks = {}
    if 'combined' in values:
        for name, index in [('j_only',0),('k_only',1)]:
            if name in values:
                delta = np.asarray(values[name][index])-np.asarray(values['combined'][index])
                finite = bool(np.isfinite(delta).all())
                checks[name] = dict(max_abs=float(np.max(np.abs(delta))) if finite else None,
                                    passed=bool(finite and np.allclose(values[name][index], values['combined'][index], atol=1e-10, rtol=1e-10)),
                                    atol=1e-10, rtol=1e-10)
    expected = 2 if need_k else 1
    emit(event='kernel_cross_consistency', checks=checks,
         status='not_fully_checked' if len(checks)!=expected else 'ok' if all(c['passed'] for c in checks.values()) else 'consistency_failed')
    raw, kind = getattr(inputs, 'grid_ao', None), xc_type(xc)
    if kind == 'HF' or not isinstance(raw, AOGrid) or not raw.adaptive:
        emit(event='grid_materializer_compiler_estimate', status='not_applicable', nested=True)
    elif deadline-time.perf_counter() < 30.:
        emit(event='grid_materializer_compiler_estimate', status='skipped_budget', nested=True)
    else:
        try:
            t = time.perf_counter()
            compiled = _materialize_scf_grid.lower(raw, block_size=raw.block_size, deriv=0 if kind=='LDA' else 1).compile()
            stats = compiled.memory_analysis()
            estimate = None if stats is None else {key:getattr(stats,key,None) for key in
                ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')}
            emit(event='grid_materializer_compiler_estimate', status='ok', nested=True, executed=False,
                 lower_compile_seconds=time.perf_counter()-t, compiler_memory_not_rss=estimate,
                 ngrid=raw.ngrids, block_size=raw.block_size, note='Compile only; no AO output executed; not additive to SCF')
        except Exception as error:
            emit(event='grid_materializer_compiler_estimate', status='error', nested=True, error=repr(error))
    return rows


if __name__ == '__main__':
    profile_followup.profile_stages.Stages.install = profile_followup.install
    profile_kernels.run_kernels = run_kernels
    raise SystemExit(profile_followup.profile_stages.main())
