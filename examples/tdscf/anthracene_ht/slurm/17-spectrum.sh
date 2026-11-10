set -euo pipefail
[ "$(cat queue/16-mode-response.sh.exit)" = 0 ]
export JAX_ENABLE_COMPILATION_CACHE=false
export JAX_COMPILATION_CACHE_MAX_SIZE=2147483648
srun -n1 -c8 --cpu-bind=cores "$PY" -m benchmarks.probes.response_grid3.guard --seconds 300 --rss-bytes 21474836480 results/spectrum.process.json -- "$PY" examples/tdscf/anthracene_ht/04_spectrum.py > results/spectrum.log 2>&1
