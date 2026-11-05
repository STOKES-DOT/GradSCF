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
