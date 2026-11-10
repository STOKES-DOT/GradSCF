"""Metadata-only tests: no imports of JAX or an electronic-structure engine."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_registered_matrix_preserves_the_requested_physical_tasks():
    path = Path(__file__).with_name('make_plan.py')
    assert path.exists(), 'The fixed 96-atom protocol generator is missing'
    spec = importlib.util.spec_from_file_location('scf_100_plan', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases, manifest = module.build_cases(ROOT)
    assert len(cases) == 60 and manifest['geometry']['natom'] == 96
    assert len({c['case_id'] for c in cases}) == 60
    for cfg in cases:
        assert cfg['basis'] == 'def2-svp' and cfg['cart']
        assert cfg['grid_level'] == 3 and cfg['init_guess'] == 'atom'
        assert (cfg['conv_tol'], cfg['density_tol'], cfg['gradient_tol']) == (1e-10, 1e-8, 1e-7)
        assert cfg['direct_scf_tol'] == cfg['df_tol'] == 1e-12
        assert cfg['max_cycle'] == 200 and cfg['threads'] == 64
        assert cfg['representation'] in ('direct', 'df')
        assert cfg['auxbasis'] == ('weigend' if cfg['representation'] == 'df' else None)
        expected = ('ri_j_direct_k' if cfg['xc_family'] == 'RSH' else
                    'ri_j' if cfg['xc_family'] in ('LDA', 'GGA') else 'ri_jk')
        assert cfg['ri_strategy'] == (expected if cfg['representation'] == 'df' else None)
        assert cfg['only_dfj'] == (cfg['representation'] == 'df' and cfg['xc_family'] == 'RSH')
        warm = (cfg['method'], cfg['xc']) in (('RKS', 'pbe'), ('UKS', 'pbe0'))
        assert cfg['repeats'] == (4 if warm else 1)
    assert len({c['geometry_sha256'] for c in cases}) == 1
    assert len({c['protocol_sha256'] for c in cases}) == 1


def worker_module():
    path = Path(__file__).with_name('worker.py')
    assert path.exists(), 'The shared SCF benchmark worker is missing'
    spec = importlib.util.spec_from_file_location('scf_100_worker', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_tampering_is_rejected_before_numerical_imports():
    module = worker_module()
    plan = Path(__file__).with_name('make_plan.py')
    spec = importlib.util.spec_from_file_location('scf_100_plan_check', plan)
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    cases, _ = producer.build_cases(ROOT)
    cfg = cases[0].copy()
    cfg['conv_tol'] = 1e-6
    try:
        module.validate_config(cfg)
    except ValueError as error:
        assert 'conv_tol' in str(error)
    else:
        raise AssertionError('Changing a frozen convergence gate was accepted')


def test_missing_only_dfj_capability_cannot_be_reported_as_ri_jk():
    module = worker_module()

    class OldFacade:
        def density_fit(self, auxbasis):
            raise AssertionError('An incompatible public entry point must not be called')

    cfg = dict(representation='df', auxbasis='weigend', only_dfj=True, df_tol=1e-12, threads=64, direct_scf_tol=1e-12)
    try:
        module.configure_gradscf(OldFacade(), cfg)
    except NotImplementedError as error:
        assert 'only_dfj' in str(error)
    else:
        raise AssertionError('Missing RI-J/direct-K support was accepted')


def test_generated_configuration_files_reproduce_registered_fingerprints(tmp_path):
    import hashlib
    import subprocess
    import sys
    script = Path(__file__).with_name('make_plan.py')
    output = tmp_path/'plan'
    subprocess.run([sys.executable, '-S', str(script), str(output)], check=True, capture_output=True, text=True)
    registered = json.loads((ROOT/'reports/scf_100/20261105-40a569bf/plan-v2/input_manifest.json').read_text())
    generated = json.loads((output/'input_manifest.json').read_text())
    assert generated == registered
    for name, sha in generated['config_file_sha256'].items():
        assert hashlib.sha256((output/'configs'/name).read_bytes()).hexdigest() == sha


def test_metadata_inspection_does_not_retain_numerical_payloads():
    import gc
    from types import SimpleNamespace
    import weakref
    module = worker_module()
    assert callable(getattr(module, 'gradscf_dimensions', None)), 'Payload inspection must have a bounded local lifetime'

    class Payload:
        df_factors = SimpleNamespace(shape=(7, 5, 5))
        grid_weights = SimpleNamespace(size=44)

    payload = Payload()
    reference = weakref.ref(payload)
    mf = SimpleNamespace(_scf_inputs=payload, reference=None)
    result = module.gradscf_dimensions(mf, SimpleNamespace())
    assert result == (7, 44), 'Observed grids, including HF preparation grids, must be reported'
    del payload, mf
    gc.collect()
    assert reference() is None


def test_pyscf_ri_j_with_retained_exchange_eris_is_not_direct():
    module = worker_module()
    assert callable(getattr(module, 'reference_strategy', None)), 'Requested and observed integral strategies must be distinct'
    cfg = dict(representation='df', only_dfj=True, ri_strategy='ri_j_direct_k')
    assert module.reference_strategy(cfg, retained_eri=True) == 'ri_j_packed_k'
    assert not module.strategy_matches(cfg, 'ri_j_packed_k')
    assert module.strategy_matches(cfg, module.reference_strategy(cfg, retained_eri=False))


def test_measurement_uses_observed_grid_size_without_hf_relabeling():
    import ast
    source = ast.parse(Path(__file__).with_name('worker.py').read_text())
    grid_fields = [keyword.value for call in ast.walk(source) if isinstance(call, ast.Call)
                   and isinstance(call.func, ast.Name) and call.func.id == 'dict'
                   for keyword in call.keywords if keyword.arg == 'ngrid']
    assert grid_fields and all(isinstance(value, ast.Name) and value.id == 'ngrid' for value in grid_fields)


def test_pure_ri_j_metadata_never_builds_missing_df_factors():
    from types import SimpleNamespace
    module = worker_module()

    class DirectJDF:
        _cderi = None
        auxmol = SimpleNamespace(nao_nr=lambda: 37)

        def get_naoaux(self):
            raise AssertionError('get_naoaux would call build() after the timed SCF')

        def build(self):
            raise AssertionError('Observing dimensions must not build integrals')

    # This is the old worker expression until the bounded collector is added.
    collect = getattr(module, 'pyscf_auxiliary_dimensions', lambda df: (int(df.get_naoaux()), None))
    assert collect(DirectJDF()) == (37, None)
    import ast
    tree = ast.parse(Path(__file__).with_name('worker.py').read_text())
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr == 'get_naoaux' for node in ast.walk(tree))


def test_existing_df_shape_is_metadata_not_a_build_request():
    from types import SimpleNamespace
    module = worker_module()
    assert callable(getattr(module, 'pyscf_auxiliary_dimensions', None))
    df = SimpleNamespace(_cderi=SimpleNamespace(shape=(31, 15)),
                         auxmol=SimpleNamespace(nao_nr=lambda: 37))
    assert module.pyscf_auxiliary_dimensions(df) == (37, 31)
    df._cderi = 'existing-factors.h5'
    assert module.pyscf_auxiliary_dimensions(df) == (37, None)


class FakeReferenceMF:
    """Kernel-control fixture; no arrays or electronic-structure imports."""
    def __init__(self, history, attempts, max_cycle=5):
        self.history, self.attempts = history, iter(attempts)
        self.max_cycle, self.init_guess, self.conv_check = max_cycle, 'atom', True
        self.calls, self.densities = [], []
        self.converged, self.cycles, self.finite = False, 0, True

    def _execute(self, kind, dm0):
        self.calls.append((kind, self, dm0, self.max_cycle))
        main, extra, self.converged, self.finite = next(self.attempts)
        assert main <= self.max_cycle
        self.cycles = main
        for index in range(main + extra):
            passed = self.converged if index >= main else bool(extra and index == main - 1)
            self.history.append(dict(energy_change=1e-12, density_rms=1e-10,
                                     gradient=5e-8 if passed else 2e-7, passed=passed))
        self.densities.append(object())
        return self

    def run(self):
        return self._execute('run', None)

    def kernel(self, *, dm0):
        return self._execute('kernel', dm0)

    def make_rdm1(self):
        return self.densities[-1]


def reference_control(module, mf, history, continuation=False):
    helper = getattr(module, 'run_reference', None)
    assert callable(helper), 'The explicit bounded reference policy is missing'
    return helper(mf, history, max_cycle=5, continuation=continuation,
                  finite_state=lambda current: current.finite)


def test_reference_continuation_reuses_mf_current_density_and_main_budget():
    history = []
    mf = FakeReferenceMF(history, [(2, 1, False, True), (3, 1, True, True)])
    result = reference_control(worker_module(), mf, history, continuation=True)
    assert len(mf.calls) == 2 and all(call[1] is mf for call in mf.calls)
    assert mf.calls[0][0] == 'run' and mf.calls[0][2] is None and mf.init_guess == 'atom'
    assert mf.calls[1][0] == 'kernel' and mf.calls[1][2] is mf.densities[0]
    assert [call[3] for call in mf.calls] == [5, 3]
    assert mf.max_cycle == 5 and mf.conv_check is True
    assert result['kernel_calls'] == 2 and result['total_main_cycles'] == 5
    assert result['total_extra_checks'] == 2 and result['total_density_updates'] == 7
    assert result['terminal_kernel_cycles'] == 3 and result['reference_accepted']
    assert result['reference_policy'] == 'bounded_final_check_restart_v1'
    assert [row['phase'] for row in history] == ['main', 'main', 'extra', 'main', 'main', 'main', 'extra']


def test_reference_default_keeps_original_single_kernel_policy():
    history = []
    mf = FakeReferenceMF(history, [(2, 1, False, True), (3, 1, True, True)])
    result = reference_control(worker_module(), mf, history)
    assert len(mf.calls) == 1 and not result['reference_accepted']
    assert result['reference_policy'] == 'original_single_kernel'
    assert result['total_main_cycles'] == 2 and result['total_extra_checks'] == 1
    assert mf.max_cycle == 5


def test_reference_nonfinite_terminal_state_is_not_restarted():
    history = []
    mf = FakeReferenceMF(history, [(2, 1, False, False)])
    result = reference_control(worker_module(), mf, history, continuation=True)
    assert len(mf.calls) == 1 and not result['reference_accepted']
    assert not result['reference_finite_terminal'] and mf.max_cycle == 5


def test_reference_main_budget_exhaustion_stops_without_reset():
    history = []
    mf = FakeReferenceMF(history, [(5, 1, False, True)])
    result = reference_control(worker_module(), mf, history, continuation=True)
    assert len(mf.calls) == 1 and result['total_main_cycles'] == 5
    assert result['total_extra_checks'] == 1 and result['total_density_updates'] == 6
    assert not result['reference_accepted'] and mf.max_cycle == 5


def test_reference_no_extra_or_no_progress_cannot_trigger_restart():
    module = worker_module()
    for attempt in [(2, 0, False, True), (0, 1, False, True)]:
        history = []
        mf = FakeReferenceMF(history, [attempt])
        result = reference_control(module, mf, history, continuation=True)
        assert len(mf.calls) == 1 and not result['reference_accepted']
        assert mf.max_cycle == 5
