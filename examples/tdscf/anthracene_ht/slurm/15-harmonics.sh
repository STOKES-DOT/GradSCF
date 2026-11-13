set -euo pipefail
[ "$(cat queue/14-optimize.sh.exit)" = 0 ]
export JAX_ENABLE_COMPILATION_CACHE=false
export JAX_COMPILATION_CACHE_MAX_SIZE=2147483648
srun -n1 -c64 --cpu-bind=cores "$PY" -m benchmarks.probes.response_grid3.guard --seconds 43200 --rss-bytes 236223201280 results/harmonics.process.json -- "$PY" examples/tdscf/anthracene_ht/02_harmonics.py > results/harmonics.log 2>&1
