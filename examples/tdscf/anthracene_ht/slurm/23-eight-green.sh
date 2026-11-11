set -euo pipefail
export JAX_PLATFORMS=cpu JAX_ENABLE_COMPILATION_CACHE=false
srun -n1 -c8 --cpu-bind=cores "$PY" -m benchmarks.probes.response_grid3.guard --seconds 300 --rss-bytes 21474836480 results/eight-green.process.json -- "$PY" -m pytest -q tests/test_multistate_transition_tracking.py tests/test_excited_geometry_tracking.py tests/test_factorized_vibronic.py tests/test_anthracene_multistate_spectrum.py --tb=short --junitxml=results/eight-green.xml > results/eight-green.log 2>&1
