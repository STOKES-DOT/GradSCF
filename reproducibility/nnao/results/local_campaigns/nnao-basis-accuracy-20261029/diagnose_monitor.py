"""Bounded test of telemetry interference, on the existing GPU allocation."""
import faulthandler
import run_gpu
from pyscf.scf.hf import init_guess_by_minao

faulthandler.dump_traceback_later(30,repeat=False)


class WithoutMemoryThread:
    def __enter__(self):
        self.baseline=0;self.peak=0
        return self
    def __exit__(self,*args):pass


run_gpu.MemorySample=WithoutMemoryThread
def cpu_minao(self,mol=None):
    return run_gpu.cp.asarray(init_guess_by_minao(self.mol if mol is None else mol))
run_gpu.hf.RHF.init_guess_by_minao=cpu_minao
for key in ['t01','nnao_szp_s']:
    run_gpu.solve(key,'S22/02/A','mp2',variant='diagnostic-cpu-minao-no-memory-thread')
faulthandler.cancel_dump_traceback_later()
