"""Freeze a 96-atom SCF matrix without importing numerical libraries."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
METHODS = ('RHF', 'UHF', 'GHF', 'RKS', 'UKS', 'GKS')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build_cases(root=ROOT, system_id='beta-carotene', basis='def2-svp', engines=('gradscf', 'pyscf')):
    catalog_path = Path(root)/'benchmarks/probes/scf_solver/systems.json'
    catalog = json.loads(catalog_path.read_text())
    system = next(item for item in catalog['systems'] if item['id'] == system_id)
    geometry = {key: system[key] for key in ('atom', 'unit', 'cart', 'charge', 'spin', 'natom')}
    protocol = dict(basis=basis, grid_level=3, init_guess='atom', threads=64,
                    max_memory=180000., rss_guard_bytes=200*1024**3,
                    max_cycle=200, conv_tol=1e-10, density_tol=1e-8, gradient_tol=1e-7,
                    df_tol=1e-12, direct_scf_tol=1e-12,
                    single_call_limit_seconds=6000, dtype='float64', device='cpu',
                    functionals=catalog['protocol']['functionals'],
                    ri_policy=dict(HF='ri_jk', global_hybrid='ri_jk', LDA='ri_j', GGA='ri_j', RSH='ri_j_direct_k'),
                    only_dfj_policy='True for RSH RI-J/direct-K; False otherwise',
                    warm_policy='One cold call plus three new-object warm calls for RKS/PBE and UKS/PBE0; no saved SCF state')
    geometry_sha, protocol_sha = fingerprint(geometry), fingerprint(protocol)
    rows = []
    for method in METHODS:
        functionals = (catalog['protocol']['functionals'] if method.endswith('KS') else
                       [dict(gradscf='hf', pyscf='hf', family='HF')])
        for functional in functionals:
            for representation in ('df', 'direct'):
                for engine in engines:
                    xc, family = functional['gradscf'], functional['family']
                    case_id = '-'.join((system_id, basis, method.lower(), xc, representation, engine))
                    repeats = 4 if (method, xc) in (('RKS', 'pbe'), ('UKS', 'pbe0')) else 1
                    rows.append(dict(geometry, id=system_id, case_id=case_id, engine=engine,
                        method=method, basis=basis, xc=xc, pyscf_xc=functional['pyscf'], xc_family=family,
                        grid_level=3, init_guess='atom', threads=64, max_memory=180000.,
                        max_cycle=200, conv_tol=1e-10, density_tol=1e-8, gradient_tol=1e-7,
                        direct_scf_tol=1e-12, df_tol=1e-12, representation=representation,
                        auxbasis='weigend' if representation == 'df' else None,
                        ri_strategy=protocol['ri_policy'][family] if representation == 'df' else None,
                        only_dfj=representation == 'df' and family == 'RSH',
                        repeats=repeats, scopes=['scf_100'], single_call_limit_seconds=6000,
                        process_limit_seconds=6000*repeats+120, energy_comparison_tolerance_hartree=1e-7,
                        geometry_sha256=geometry_sha, protocol_sha256=protocol_sha))
    manifest = dict(schema_version=1, geometry=dict(geometry, sha256=geometry_sha, source=system['source']),
                    protocol=dict(protocol, sha256=protocol_sha),
                    catalog_sha256=hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
                    row_count=len(rows), calls_if_supported=sum(row['repeats'] for row in rows),
                    initial_guess_note='GradSCF atom uses non-spherical isolated native atomic UHF; PySCF atom uses its own atomic-HF guess. These are distinct initial densities, not shared input.',
                    capability_note='Every requested method is retained. Missing production capability is unsupported, never silently replaced by packed ERIs, another spin method or another RI policy.')
    return rows, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--system', default='beta-carotene', help='methane provides a separate smoke input')
    parser.add_argument('--basis', default='def2-svp')
    parser.add_argument('--engines', nargs='+', choices=('gradscf', 'pyscf'), default=['gradscf', 'pyscf'])
    args = parser.parse_args()
    rows, manifest = build_cases(system_id=args.system, basis=args.basis, engines=args.engines)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'configs').mkdir()
    for row in rows:
        (args.output/'configs'/(row['case_id']+'.json')).write_text(json.dumps(row, indent=2)+'\n')
    manifest['config_file_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in sorted((args.output/'configs').glob('*.json'))}
    (args.output/'input_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    plan = dict(independent_worker_count=len(rows), run_order=[row['case_id'] for row in rows],
                rss_guard_bytes=200*1024**3, max_logical_cpus=64, max_memory_bytes=256*1024**3,
                geometry_sha256=manifest['geometry']['sha256'], protocol_sha256=manifest['protocol']['sha256'])
    (args.output/'run_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), configurations=len(rows), calls_if_supported=manifest['calls_if_supported'],
                          geometry_sha256=plan['geometry_sha256'], protocol_sha256=plan['protocol_sha256'])))


if __name__ == '__main__':
    main()
