"""Physical-only benchmark workflow over the unchanged frozen energy engine.

The numerical engine is preserved byte-for-byte so existing physical energies
retain their exact identity and are reused. No ghost or conditioned cases run.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

from run_gpu_engine import ROOT,ROWS,SETTINGS,DRIVER_SHA,solve,write

LANES=dict(
    small=['t%02d'%i for i in [0,1,2,3,4,5,8,9,14,15]]+
          [f'nnao_{z}p_{s}' for z in ['sz','dz'] for s in ['s','m','l']],
    tz=['t%02d'%i for i in [6,7,10,11,16]]+[f'nnao_tzp_{s}' for s in ['s','m','l']],
    qz=['t12','t13','t17','reference'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--lane',required=True,choices=LANES)
    parser.add_argument('--direct',action='store_true')
    args=parser.parse_args()
    gate=json.loads((ROOT/'export-gate.json').read_text());assert gate['status']=='passed'
    for name,sha in gate['files'].items():
        assert hashlib.sha256((ROOT/'frozen'/name).read_bytes()).hexdigest()==sha,name
    failures=[];started=time.perf_counter()
    for key in LANES[args.lane]:
        for identifier in ROWS:
            for method in SETTINGS['methods']:
                try:solve(key,identifier,method)
                except Exception as error:failures.append(str(error))
    write(ROOT/f'lane-{args.lane}.json',dict(status='passed' if not failures else 'failed',
        failures=failures,seconds=time.perf_counter()-started,bases=LANES[args.lane],
        execution_mode='direct',calculation_scope='physical_only',cpu_comparisons_run=False,
        cp_bsse_calculations=False,expected_physical_cases=len(LANES[args.lane])*len(ROWS)*len(SETTINGS['methods']),
        numerical_engine_sha256=DRIVER_SHA,
        workflow_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    if failures:raise RuntimeError(f'{len(failures)} explicit failed physical cases; no score will omit them')


if __name__=='__main__':main()
