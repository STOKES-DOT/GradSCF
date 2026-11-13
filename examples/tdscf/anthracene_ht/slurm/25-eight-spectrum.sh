set -euo pipefail
[ "$(cat queue/24-eight-mode-response.sh.exit)" = 0 ]
export JAX_PLATFORMS=cpu JAX_ENABLE_COMPILATION_CACHE=false
export GRADSCF_ANTHRACENE_INPUT="$RUN/examples/tdscf/anthracene_ht/input_eight_states.json"
export GRADSCF_ANTHRACENE_OUTPUT="$RUN/reports/anthracene_ht/20261110-b3lyp-def2svp/eight_states/raw"
srun -n1 -c8 --cpu-bind=cores "$PY" -m benchmarks.probes.response_grid3.guard --seconds 300 --rss-bytes 21474836480 results/eight-spectrum.process.json -- "$PY" examples/tdscf/anthracene_ht/04_spectrum.py > results/eight-spectrum.log 2>&1
