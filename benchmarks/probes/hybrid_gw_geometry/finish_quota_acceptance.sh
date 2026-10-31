#!/bin/bash
# Last controller only: release this allocation after success or bounded failure.
set -eu
allocation=$1
gradient_controller=$2
finish() {
    code=$?
    printf '%s\n' "$code" > "results/finish-quota-$allocation.exit"
    touch release.spin
}
trap finish EXIT
while kill -0 "$gradient_controller" 2>/dev/null; do sleep 5; done
python3 - "$allocation" <<'PY'
import json, sys
from pathlib import Path
root=Path('results')
name=f'phi-quota-gradient-{sys.argv[1]}'
process=json.loads((root/f'{name}.process.json').read_text())
assert process['status'] == 'ok' and process['returncode'] == 0
rows=[json.loads(line) for line in (root/f'{name}.jsonl').read_text().splitlines()]
assert any(r['stage']=='derivative' and r['scientific_pass'] for r in rows)
PY
for step in 0.0003 0.0001; do
    bash harness/cpu_geometry.sh "$allocation" "fd-$step"
    python3 - "$allocation" "$step" <<'PY'
import json, sys
from pathlib import Path
path=Path('results')/f'phi-quota-fd-{sys.argv[2]}-{sys.argv[1]}.jsonl'
rows=[json.loads(line) for line in path.read_text().splitlines()]
assert any(r['stage']=='directional_fd' and r['scientific_pass'] for r in rows)
PY
done
