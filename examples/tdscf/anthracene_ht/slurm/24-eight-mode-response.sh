set -euo pipefail
[ "$(cat queue/23-eight-green.sh.exit)" = 0 ]
export JAX_PLATFORMS=cuda,cpu JAX_ENABLE_COMPILATION_CACHE=false
export GRADSCF_ANTHRACENE_INPUT="$RUN/examples/tdscf/anthracene_ht/input_eight_states.json"
export GRADSCF_ANTHRACENE_OUTPUT="$RUN/reports/anthracene_ht/20261110-b3lyp-def2svp/eight_states/raw"
srun -n1 -c64 --cpu-bind=cores "$PY" -m benchmarks.probes.response_grid3.guard --seconds 60000 --rss-bytes 236223201280 results/eight-mode-response.process.json -- "$PY" examples/tdscf/anthracene_ht/03_mode_response.py > results/eight-mode-response.log 2>&1
