set -euo pipefail
export JAX_COMPILATION_CACHE_MAX_SIZE=2147483648
[ "$(cat queue/02-factorized-green.sh.exit)" = 0 ]
srun -n1 -c64 --cpu-bind=cores "$PY" -m benchmarks.probes.response_grid3.guard --seconds 21600 --rss-bytes 236223201280 results/optimization.process.json -- "$PY" examples/tdscf/anthracene_ht/01_optimize.py > results/optimization.log 2>&1
