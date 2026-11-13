set -euo pipefail
[ "$(cat queue/15-harmonics.sh.exit)" = 0 ]
export JAX_ENABLE_COMPILATION_CACHE=false
export JAX_COMPILATION_CACHE_MAX_SIZE=2147483648
srun -n1 -c64 --cpu-bind=cores "$PY" -m benchmarks.probes.response_grid3.guard --seconds 86400 --rss-bytes 236223201280 results/mode-response.process.json -- "$PY" examples/tdscf/anthracene_ht/03_mode_response.py > results/mode-response.log 2>&1
