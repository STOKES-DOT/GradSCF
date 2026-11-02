"""Validate official package pins and the complete CUDA toolkit before SCF."""
from pathlib import Path
import argparse
import ast
import contextlib
import ctypes
from ctypes.util import find_library
import importlib.metadata as metadata
import io
import json
import os
import subprocess
import sys
import time

from packaging.requirements import Requirement

ROOT=Path(__file__).resolve().parent.parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',action='store_true')
    args=parser.parse_args();start=time.perf_counter()
    mode='gpu' if args.gpu else 'cpu'
    report={'status':'running','mode':mode,'job_id':os.environ.get('SLURM_JOB_ID')}
    output=ROOT/'runtime'/f'toolkit-validation-{mode}.json'
    try:
        pins={}
        for line in (ROOT/'runtime/requirements-official-v1.7.0.txt').read_text().splitlines():
            if not line.strip() or line.startswith('#'):continue
            requirement=Requirement(line);version=metadata.version(requirement.name)
            assert version in requirement.specifier,(requirement,version)
            pins[requirement.name]=version
        cupy_names=[d.metadata['Name'] for d in metadata.distributions()
                    if d.metadata['Name'].lower().startswith('cupy')]
        assert cupy_names==['cupy-cuda12x'],cupy_names
        report['packages']=pins
        toolkit=Path(os.environ['CUDA_PATH'])
        report['nvcc']=subprocess.check_output([str(toolkit/'bin/nvcc'),'--version'],text=True)
        assert 'release 12.8' in report['nvcc']
        assert (toolkit/'include/cuda_runtime.h').is_file()
        report['libraries']={}
        for name,symbol in [('cusolver','cusolverDnDsygvd_bufferSize'),
                            ('cublas','cublasCreate_v2'),('nvrtc','nvrtcVersion'),
                            ('cudart','cudaRuntimeGetVersion'),('cutensor','cutensorGetVersion')]:
            found=find_library(name);assert found is not None,(name,found)
            handle=ctypes.CDLL(found);assert getattr(handle,symbol)
            report['libraries'][name]=found
        # Test every direct cuSOLVER symbol used by this installed GPU4PySCF.
        package=metadata.distribution('gpu4pyscf-cuda12x')
        source=package.locate_file('gpu4pyscf/lib/cusolver.py').read_text()
        symbols=sorted({n.attr for n in ast.walk(ast.parse(source))
                        if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name)
                        and n.value.id=='libcusolver'})
        handle=ctypes.CDLL(find_library('cusolver'))
        for symbol in symbols:assert getattr(handle,symbol)
        report['cusolver_symbols']=symbols
        runtime=ctypes.CDLL(find_library('cudart'));runtime_version=ctypes.c_int()
        assert runtime.cudaRuntimeGetVersion(ctypes.byref(runtime_version))==0
        assert runtime_version.value==12080,runtime_version.value
        report['cuda_runtime_version']=runtime_version.value
        nvrtc=ctypes.CDLL(find_library('nvrtc'));major,minor=ctypes.c_int(),ctypes.c_int()
        assert nvrtc.nvrtcVersion(ctypes.byref(major),ctypes.byref(minor))==0
        assert (major.value,minor.value)==(12,8)
        report['nvrtc_version']=[major.value,minor.value]
        cutensor_library=ctypes.CDLL(find_library('cutensor'))
        cutensor_library.cutensorGetVersion.restype=ctypes.c_size_t
        report['cutensor_version']=cutensor_library.cutensorGetVersion()
        assert report['cutensor_version']==20200,report['cutensor_version']
        maps=Path('/proc/self/maps').read_text().splitlines()
        loaded=sorted({line.split()[-1] for line in maps if any(name in line
                       for name in ['libcusolver.so','libcublas.so','libnvrtc.so',
                                    'libcudart.so','libcutensor.so'])})
        for path in loaded:
            if 'libcutensor.so' in path:assert 'gpu4pyscf_bench/' in path,path
            else:assert '/25.3/' in path and '/12.8/' in path,path
        report['loaded_library_paths']=loaded
        if args.gpu:
            import numpy as np
            import cupy as cp
            from scipy.linalg import eigh
            from gpu4pyscf.lib import cutensor,cusolver
            from gpu4pyscf.scf import hf
            from gpu4pyscf.dft import rks
            from gpu4pyscf.mp import dfmp2
            assert cp.cuda.runtime.getDeviceCount()==1
            report['device']=cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
            config=io.StringIO()
            with contextlib.redirect_stdout(config):cp.show_config()
            report['cupy_config']=config.getvalue()
            assert cutensor.cutensor is not None and cutensor.contract_engine is None
            a=np.array([[2.,.3],[.3,1.]])
            s=np.array([[1.,.1],[.1,1.2]])
            np.testing.assert_array_equal(cp.arange(4).get(),np.arange(4))
            np.testing.assert_allclose(cutensor.contract('ij,jk->ik',cp.asarray(a),cp.asarray(s)).get(),a@s,atol=1e-12)
            w,v=cusolver.eigh(cp.asarray(a),cp.asarray(s))
            np.testing.assert_allclose(w.get(),eigh(a,s)[0],rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(a@v.get(),s@v.get()*w.get(),atol=1e-12)
            report['gpu_imports_and_kernels']='passed'
        report['status']='passed'
    except Exception as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-start
        output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
