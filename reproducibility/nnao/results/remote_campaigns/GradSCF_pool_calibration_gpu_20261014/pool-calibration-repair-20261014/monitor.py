"""External /proc sampling survives a child LLVM abort."""
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path('artifacts/pool-calibration-repair-20261014')
root.mkdir(exist_ok=True,parents=True)
suffix='-gpu' if '--gpu' in sys.argv else ''
with (root/f'reproduce{suffix}.log').open('w') as log,(root/f'resources{suffix}.jsonl').open('w') as samples:
    child=subprocess.Popen([sys.executable,'-u',str(root/'reproduce.py'),*sys.argv[1:]],stdout=log,stderr=subprocess.STDOUT)
    while child.poll() is None:
        try:
            proc=Path('/proc')/str(child.pid)
            record={'time':time.time(),'maps':len((proc/'maps').read_text().splitlines())}
            for line in (proc/'status').read_text().splitlines():
                if line.startswith(('VmRSS:','VmHWM:','RssAnon:','Threads:')):
                    k,v=line.split(':',1);record[k]=v.strip()
            samples.write(json.dumps(record)+'\n');samples.flush()
        except OSError:
            pass
        time.sleep(.25)
    (root/f'returncode{suffix}.json').write_text(json.dumps({'returncode':child.returncode})+'\n')
    print('REPRODUCER_EXIT',child.returncode,flush=True)
