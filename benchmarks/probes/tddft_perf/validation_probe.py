"""Narrow existing non-symmetric, neural-response and NNAO regressions."""
import json
from pathlib import Path
import subprocess
import sys
import time
from worker import Recorder


cfg=json.loads(Path(sys.argv[1]).read_text())
output=Path(sys.argv[2]).resolve();output.mkdir(parents=True,exist_ok=True)
log=Recorder(cfg,output)
tests=[
    'tests/test_direct_tdhf_fusion.py',
    'tests/test_streamed_response_grid.py::test_streamed_custom_hvp_retains_complete_cross_grid_coupling',
    'tests/test_streamed_response_grid.py::test_streamed_legacy_neural_scf_projection_and_parameter_vjp_match_dense',
    'tests/nnao/test_libcint_consumers.py::test_two_geometry_projection_log_exponents_jvp_vjp_and_fd',
]
start=time.perf_counter()
done=subprocess.run([sys.executable,'-m','pytest','-q',*tests,'--junitxml='+str(output/'regression.xml')],check=False)
log.add('focused_regressions',time.perf_counter()-start,status='ok' if done.returncode==0 else 'error',tests=tests,returncode=done.returncode)
if done.returncode:sys.exit(done.returncode)
start=time.perf_counter()
done=subprocess.run([sys.executable,'tools/libcint_migration/nnao_step.py','--device','cpu','--mode','full-step','--output',str(output/'nnao-step.json')],check=False)
log.add('nnao_training_fixture',time.perf_counter()-start,status='ok' if done.returncode==0 else 'error',returncode=done.returncode,
        notes='small random MACE fixture; no production checkpoint; exact existing training entry')
sys.exit(done.returncode)
