"""Non-intrusive NVML sampling; the monitor thread never calls CUDA APIs."""
import threading
import pynvml


class MemorySample:
    def __init__(self,pci_bus_id):self.pci_bus_id=pci_bus_id
    def __enter__(self):
        pynvml.nvmlInit()
        self.handle=pynvml.nvmlDeviceGetHandleByPciBusId(self.pci_bus_id)
        self.baseline=int(pynvml.nvmlDeviceGetMemoryInfo(self.handle).used)
        self.peak=self.baseline;self.error=None;self.stop=threading.Event()
        def sample():
            while not self.stop.is_set():
                try:self.peak=max(self.peak,int(pynvml.nvmlDeviceGetMemoryInfo(self.handle).used))
                except Exception as e:self.error=e;break
                self.stop.wait(.1)
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start()
        return self
    def __exit__(self,*args):
        self.stop.set();self.thread.join()
        self.peak=max(self.peak,int(pynvml.nvmlDeviceGetMemoryInfo(self.handle).used))
        pynvml.nvmlShutdown()
        if self.error is not None:raise self.error
