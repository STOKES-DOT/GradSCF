import sys,runpy
from pathlib import Path
root=Path('/tmp/gradscf-szp663-20260916')
sys.path[:0]=[str(root/'src'),str(root/'src/nnao'),'/tmp/gradscf-nnao-20260916/deps']
sys.path.append('/home/yjiao/opt/miniconda3/envs/grad_dft/lib/python3.11/site-packages')
mode=sys.argv.pop(1)
if mode=='test':
 import pytest
 raise SystemExit(pytest.main(sys.argv[1:]))
if mode=='optimize':
 sys.argv[0]='tools/optimize_methane_nnao.py'
 runpy.run_path(sys.argv[0],run_name='__main__')
else:raise ValueError(mode)
