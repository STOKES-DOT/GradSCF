"""Thin S9 consumer catalogue and profiling adapter for the existing runner.

Input/output arguments match worker.py. Numerical work stays in that worker,
nnao_step.py, or the registered tests; this module owns no solver or formula.
Native-count runs are diagnostic runs and cannot enter the clean speed table.
"""
from contextlib import contextmanager
import argparse
import cProfile
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CASES = {
    'scf_pbe': dict(group='scf', driver='worker', method='pbe', native_applicable=True,
        scope='fresh molecular RKS value', outputs=('E', 'D/F/reference.npz', 'SCF residual/cycles')),
    'scf_direct': dict(group='scf', driver='worker', method='hf', native_applicable=True,
        required_options={'jk_backend': 'direct'}, scope='fresh native-direct RHF value',
        outputs=('E', 'D/F/reference.npz', 'SCF residual/cycles')),
    'scf_geometry': dict(group='scf', driver='pytest', native_applicable=True,
        scope='complete moving-integral/grid/SCF response', outputs=('E', 'R gradient/FD', 'HF R3/R4'),
        nodes=('tests/test_molecular_scf_forward.py::test_scf_forward_matches_minimum_state_and_complete_geometry_response',
               'tests/test_molecular_geometry_higher.py::test_h2_relaxed_cubic_quartic_against_independent_hessian_differences',
               'tests/test_scf_orbital_direct.py::test_direct_geometry_gradient_and_hvp_match_full')),
    'spin_generalized': dict(group='spin', driver='pytest', native_applicable=True,
        scope='UHF/UKS relaxed geometry; GKS fixed-input partial and value',
        outputs=('spin density/particle count', 'E/R gradient', 'complex spinor value', 'GKS input response'),
        nodes=('tests/test_molecular_unrestricted_response.py',
               'tests/test_scf_ghf_gks.py::test_gks_facade_collinear_pbe',
               'tests/test_scf_orbital_dft_gradients.py::test_dft_energy_and_density_response_to_integrals_and_grid[implicit-gks_ncol]',
               'tests/test_scf_ghf_gks.py::test_generalized_response_is_explicitly_unsupported')),
    'tda': dict(group='excitation', driver='worker', method='pbe_tda', native_applicable=True,
        scope='fresh RKS/TDA value', outputs=('excitation energies', 'oscillator strengths', 'convergence')),
    'tddft': dict(group='excitation', driver='worker', method='pbe_td', native_applicable=True,
        scope='fresh RKS/full-TDDFT value', outputs=('excitation energies', 'oscillator strengths', 'convergence')),
    'excitation_geometry': dict(group='excitation', driver='pytest', native_applicable=True,
        scope='relaxed R/U response and CIS transition-amplitude geometry',
        outputs=('excitation R gradient', 'JVP/VJP/FD', 'transition moment/strength response'),
        nodes=('tests/test_direct_excitation_geometry.py::test_relaxed_direct_excitation_geometry_matches_full_and_finite_difference[rks-pbe-tda]',
               'tests/test_direct_excitation_geometry.py::test_restricted_pbe_full_tddft_relaxed_geometry_matches_full_and_finite_difference',
               'tests/test_direct_excitation_geometry.py::test_heh_both_spin_channels_relaxed_geometry_matches_full_and_finite_difference[hf-utdhf]',
               'tests/test_cis_transition_response.py::test_cis_transition_geometry_response_matches_independent_energy_state_fd')),
    'ci_fci': dict(group='ci_fci', driver='pytest', native_applicable=True,
        scope='selected relaxed CI/FCI; supplied-frame UCI and FCI eigenpair partials',
        outputs=('CI/FCI E/R gradient/HVP', 'core/active selected tensors', 'FCI density response'),
        nodes=('tests/ci/test_correlated_blocks.py::test_selected_relaxed_ci_fci_matches_full_mo[h2-cisd-None-0]',
               'tests/ci/test_correlated_blocks.py::test_selected_relaxed_ci_fci_matches_full_mo[lih-fci-None-1]',
               'tests/ci/test_selected_integrals.py::test_prepared_raw_c_f_energy_ao_chart_jvp_vjp_hvp[True]',
               'tests/fci/test_response.py::test_energy_and_density_jvp_vjp[dense]')),
    'mp2': dict(group='mp2_ri', driver='worker', method='mp2', native_applicable=True,
        scope='fresh conventional RHF/MP2 value', outputs=('E_HF/E_corr/E_total', 't2 shape')),
    'ri_mp2': dict(group='mp2_ri', driver='worker', method='dfmp2', native_applicable=True,
        scope='fresh true-weigend-RI RHF/MP2 value; no RI nuclear AD claim',
        outputs=('E_HF/E_corr/E_total', 'naux', 't2 shape')),
    'mp2_cuda_geometry': dict(group='mp2_ri', driver='pytest', native_applicable=True,
        requires_gpu=True, scope='complete direct MP2 CPU/CUDA geometry placement and response',
        outputs=('E', 'R gradient/HVP', 'actual array devices'),
        nodes=('tests/mp/test_native_gpu_geometry.py::test_nitrogen_direct_mp2_geometry_cpu_cuda',)),
    'recursive_mp': dict(group='recursive_mp', driver='pytest', native_applicable=True,
        scope='complete selected R/U MP4 geometry; recursive order distinct from AD order',
        outputs=('MP4 E', 'R gradient/JVP/HVP', 'raw/selected parity'),
        nodes=('tests/mp/test_selected_series_geometry.py::test_selected_series_relaxed_geometry_matches_raw[h2-series]',
               'tests/mp/test_selected_series_geometry.py::test_selected_series_relaxed_geometry_matches_raw[h3-tensor]')),
    'ccsdt': dict(group='cc_eom', driver='worker', method='ccsdt', native_applicable=True,
        scope='fresh RHF/CCSD/(T) value with original amplitudes',
        outputs=('E_corr/E_total/(T)', 'CC residual/iterations', 'synchronized t1/t2 shapes')),
    'cc_eom_response': dict(group='cc_eom', driver='pytest', native_applicable=True,
        scope='selected relaxed UCCSD(T) R2; native EOM supplied-MO/ground response',
        outputs=('nonzero (T)', 'E/R gradient/HVP', 'EE/IP/EA actions/roots', 'EA C partial'),
        nodes=('tests/cc/test_scf_unrestricted_geometry.py::test_selected_uhf_relaxed_gradient_hvp_and_nonzero_triples[ccsd(t)]',
               'tests/cc/test_native_eom_action.py::test_native_eom_right_left_actions_and_roots_match_dense',
               'tests/cc/test_native_eom_action.py::test_native_ea_root_keeps_ground_and_orbital_response')),
    'gwbse': dict(group='gw_bse', driver='worker', method='gwbse', native_applicable=True,
        scope='original formal QP/G/W/BSE windows from input JSON, unchanged',
        outputs=('QP energies/residuals/weights/masks/candidates', 'TDA/full BSE roots/residuals')),
    'gw_bse_response': dict(group='gw_bse', driver='pytest', native_applicable=True,
        scope='RHF relaxed geometry/optical R1; UGW/qsGW/scGW fixed-input partials',
        outputs=('QP/weight/screening R response', 'BSE strength/polarizability', 'outer/factor response'),
        nodes=('tests/test_molecular_gw_response.py::test_relaxed_qp_and_weight_geometry_response[g0w0]',
               'tests/test_molecular_gw_response.py::test_complete_bse_geometry_chain_and_transition_operator[False]',
               'tests/gw/test_gw_outer_response.py::test_qsgw_orbital_and_energy_response_matches_reconverged_fd',
               'tests/gw/test_gw_scgw_response.py::test_h2_df_response_matches_reconverged_finite_difference',
               'tests/bse/test_unrestricted.py::test_open_shell_optical_jvp_vjp_includes_both_screening_channels[False-davidson]')),
    'gaussian_ofdft': dict(group='ofdft', driver='pytest', native_applicable=True,
        scope='fixed Gaussian S/T/V/s4 inputs, self-consistent potential response',
        outputs=('E/density/N/mu', 'potential response/Taylor2'),
        nodes=('tests/dfpt/test_ofdft.py::test_molecular_gaussian_ofdft_response',)),
    'nnao': dict(group='nnao', driver='nnao_step', native_applicable=True,
        scope='unchanged real-MACE fixture; native fixed-aux c/log-alpha energy gradient',
        outputs=('loss/gradient/JointAdam step', 'transfer/component times', 'checkpoint roundtrip')),
    'periodic_na': dict(group='periodic_na', driver='pytest', native_applicable=False,
        scope='FFT/GTH periodic control; molecular native 4c counts N/A',
        outputs=('periodic E/force', 'GTH first/second geometry response'),
        nodes=('tests/pbc/test_gradients.py::test_nonlocal_gth_geometry_gradient_and_second_derivative',)),
}


class CounterUnavailable(RuntimeError):
    pass


class CounterAdapter:
    """Inject reset/read callbacks; numerical consumers never import this tool."""
    def __init__(self, reset, snapshot):
        self._reset, self._snapshot = reset, snapshot

    def reset(self):
        self._reset()

    def snapshot(self):
        snapshot = self._snapshot()
        if not isinstance(snapshot, dict) or snapshot.get('schema') != 'libcint-call-counts-v1':
            raise CounterUnavailable('Unknown libcint counter schema')
        if not snapshot.get('enabled', False):
            raise CounterUnavailable('Native counters are not enabled; use the separate instrumented build')
        return snapshot


def counted_call(function, *, ready, counter):
    counter.reset()
    value = ready(function())
    return value, counter.snapshot()


def _native_counter():
    try:
        from gradscf.integrals._native.diagnostics import reset_counts, get_counts
    except ImportError as error:
        raise CounterUnavailable('Native counter instrumentation is unavailable') from error
    adapter = CounterAdapter(reset_counts, get_counts)
    adapter.snapshot()  # Fail before expensive warmup if this is a production build.
    return adapter


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def _script(name):
    spec = importlib.util.spec_from_file_location('s9_'+name, HERE/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def _arguments(values):
    previous, sys.argv = sys.argv, values
    try:
        yield
    finally:
        sys.argv = previous


def _build_metadata(applicable):
    if not applicable:
        return dict(native_applicable=False)
    try:
        from gradscf.integrals._native.diagnostics import get_counts
    except ImportError:
        return dict(enabled=None, build_kind='unobserved')
    return {key: value for key, value in get_counts().items() if key != 'families'}


def _worker(cfg, output, case, counter, profile):
    module = _script('worker')
    method = case['method']
    if cfg.get('method', method) != method or cfg.get('engine', 'gradscf') != 'gradscf':
        raise ValueError('Probe worker method/engine differs from its registered case')
    if not cfg.get('atom') or not cfg.get('basis'):
        raise ValueError('Worker probes require explicit original atom and basis inputs')
    for key, value in case.get('required_options', {}).items():
        if cfg.get(key) != value:
            raise ValueError(f'This registered case requires {key}={value!r}')
    resolved = dict(cfg, engine='gradscf', method=method)
    path = output/'resolved-input.json'
    _save(path, resolved)
    if counter is not None:
        if cfg.get('repeats', 4) < 2:
            raise ValueError('Native-count mode needs the existing cold call plus at least one warm call')
        original = module.calculate
        def calculate(config, folder, repeat):
            if repeat == 0:
                return original(config, folder, repeat)
            result, counts = counted_call(lambda: original(config, folder, repeat),
                                         ready=module.ready, counter=counter)
            _save(folder/f'native-counts-r{repeat}.json', dict(
                scope='fresh-object task after existing first invocation', repeat=repeat, snapshot=counts))
            return result
        module.calculate = calculate
    with _arguments([str(HERE/'worker.py'), str(path), str(output)]+(['--profile'] if profile else [])):
        module.main()
    return json.loads((output/'records.json').read_text())


def _pytest_once(nodes, folder):
    import pytest
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    report = folder/'junit.xml'
    previous = Path.cwd()
    try:
        os.chdir(ROOT)  # Existing fixtures also open repository-relative inputs.
        code = int(pytest.main(['-q', *nodes, '--junitxml='+str(report)]))
    finally:
        os.chdir(previous)
    counts = dict(tests=0, failures=0, errors=0, skipped=0)
    if report.exists():
        for suite in ET.parse(report).getroot().iter('testsuite'):
            if not suite.findall('testsuite'):
                for key in counts:
                    counts[key] += int(suite.get(key, '0'))
    status = ('error' if code else 'skipped' if counts['tests'] == counts['skipped'] else
              'partial' if counts['skipped'] else 'ok')
    return dict(status=status, pytest_exitcode=code, junit=str(report), checks=counts)


def _fixture_ready(value):
    # Requested physical outputs are synchronized by the registered tests'
    # NumPy/host assertions; effects_barrier also completes runtime callbacks.
    import jax
    jax.effects_barrier()
    return value


def _validation(case, output, counter):
    nodes = case['nodes']
    if counter is None:
        return _fixture_ready(_pytest_once(nodes, output))
    warmup = _fixture_ready(_pytest_once(nodes, output/'warmup'))
    if warmup['status'] != 'ok':
        return dict(warmup, native_counts_status='warmup_not_complete')
    result, counts = counted_call(lambda: _pytest_once(nodes, output/'measured'),
                                 ready=_fixture_ready, counter=counter)
    _save(output/'native-counts-validation.json', dict(
        scope='prewarmed validation-fixture rerun including fixtures, FD and transpose checks', snapshot=counts))
    return result


def _nnao(cfg, output, counter):
    mode = cfg.get('mode', 'full-step')
    if mode not in ('full-step', 'components'):
        raise ValueError('Use the existing NNAO full-step or components mode')
    if counter is not None and mode != 'full-step':
        raise ValueError('Use full-step for batch native counts; component timing stays a separate unchanged process')
    module = _script('nnao_step')
    original = None
    if counter is not None:
        from gradscf.model.nnao.training.native import NativeJointEngine
        ready = _script('worker').ready
        original, calls = NativeJointEngine.evaluate, 0
        def evaluate(self, *args, **kwargs):
            nonlocal calls
            index, calls = calls, calls+1
            if index == 0:
                return original(self, *args, **kwargs)
            result, counts = counted_call(lambda: original(self, *args, **kwargs), ready=ready, counter=counter)
            _save(output/f'native-counts-batch-{index}.json', dict(
                scope='same-engine loss/gradient batch; optimizer has no Gaussian calls',
                phase='warm_step' if index < 4 else 'final_validation', batch=index, snapshot=counts))
            return result
        NativeJointEngine.evaluate = evaluate
    report = output/'nnao.json'
    try:
        with _arguments([str(HERE/'nnao_step.py'), '--device', cfg.get('device', 'cpu'),
                         '--mode', mode, '--seed', str(cfg.get('seed', 3)), '--output', str(report)]):
            module.main()
    finally:
        if original is not None:
            NativeJointEngine.evaluate = original
    values = json.loads(report.read_text())
    return dict(status='ok' if values['status'] == 'complete' else values['status'], values=values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?', type=Path)
    parser.add_argument('output', nargs='?', type=Path)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--profile', action='store_true', help='existing cProfile diagnostic; never a clean timing')
    args = parser.parse_args()
    if args.list:
        print(json.dumps(CASES, indent=2))
        return
    if args.config is None or args.output is None:
        parser.error('Supply the existing input JSON and output directory, or --list')
    cfg = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    case = CASES[cfg['probe']]
    started = time.perf_counter()
    rows = []
    try:
        if case.get('requires_gpu'):
            if cfg.get('device') != 'gpu':
                raise NotImplementedError('This placement gate requires the explicitly allocated GPU')
            os.environ['GRADSCF_TEST_GPU'] = '1'
        counter = _native_counter() if cfg.get('native_counts') and case['native_applicable'] else None
        if case['driver'] == 'worker':
            rows = _worker(cfg, args.output, case, counter, args.profile)
        else:
            function = (lambda: _validation(case, args.output, counter)) if case['driver'] == 'pytest' else (
                lambda: _nnao(cfg, args.output, counter))
            if args.profile:
                profiler = cProfile.Profile()
                result = profiler.runcall(function)
                profiler.dump_stats(args.output/'validation-profile.pstats')
            else:
                result = function()
            scope = ('dedicated NNAO driver owns first/three-warm step or component timings; not this outer wall'
                     if case['driver'] == 'nnao_step' else 'validation fixture; not a clean single-method timer')
            rows = [dict(result, repeat=0, timing_scope=scope)]
        build = _build_metadata(case['native_applicable'])
        for row in rows:
            row.update(consumer_probe=cfg['probe'], consumer_group=case['group'], derivative_scope=case['scope'],
                requested_outputs=case['outputs'], probe_input_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), native_build=build,
                native_counts_requested=bool(cfg.get('native_counts')), cprofile=args.profile,
                timing_eligible=(case['driver'] in {'worker', 'nnao_step'} and not args.profile and counter is None and
                                 build.get('enabled') is False),
                native_applicable=case['native_applicable'])
    except Exception as error:
        status = ('counter_unavailable' if isinstance(error, CounterUnavailable) else
                  'unsupported' if isinstance(error, (NotImplementedError, ImportError)) else 'error')
        rows.append(dict(status=status, exception=repr(error),
                         traceback=traceback.format_exc(), timing_eligible=False))
        traceback.print_exc()
    for row in rows:
        row.update(probe_process_seconds=time.perf_counter()-started,
            peak_rss_bytes=1024*resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            memory_method='Linux process high-water RSS including compilation, preparation and prior repeats',
            affinity=sorted(os.sched_getaffinity(0)))
        row.setdefault('config', cfg)
    _save(args.output/'records.json', rows)
    print(json.dumps(dict(probe=cfg['probe'], statuses=[row['status'] for row in rows])))
    # Match worker.py: the existing controller classifies records separately
    # as nonconverged/skipped/unsupported/error, rather than erasing that state
    # behind a generic nonzero-process-returncode classification.


if __name__ == '__main__':
    main()
