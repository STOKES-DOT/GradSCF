"""Bounded independent CuPy compiler probe; no PySCF or model imports."""
import faulthandler
import os
from pathlib import Path
import cupy as cp
faulthandler.dump_traceback_later(20,repeat=False)
print('CUDA_CONFIG',flush=True)
cp.show_config()
print('CUDA_ENV',{k:v for k,v in os.environ.items() if k in ['LD_LIBRARY_PATH','LD_PRELOAD','CUDA_PATH','CUDA_HOME','CUDA_VISIBLE_DEVICES','CUPY_CACHE_DIR']},flush=True)
for line in Path('/proc/self/maps').read_text().splitlines():
    if any(k in line for k in ['nvrtc','cudart','libcuda.so']):print('LIBRARY',line.split()[-1],flush=True)
print('BEFORE_ARANGE',flush=True)
print(cp.arange(2).get(),flush=True)
print('CUPY_COMPILER_PASSED',flush=True)
