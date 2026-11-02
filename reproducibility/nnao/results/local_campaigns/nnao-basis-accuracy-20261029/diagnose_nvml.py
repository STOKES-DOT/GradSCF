"""Same first scientific case with a monitor that does not touch CUDA."""
import faulthandler
import run_gpu
from telemetry import MemorySample

faulthandler.dump_traceback_later(45,repeat=True)
run_gpu.MemorySample=lambda:MemorySample(run_gpu.cp.cuda.Device(0).pci_bus_id)
run_gpu.solve('nnao_szp_s','S22/02/A','mp2',variant='diagnostic-nvml-monitor')
faulthandler.cancel_dump_traceback_later()
