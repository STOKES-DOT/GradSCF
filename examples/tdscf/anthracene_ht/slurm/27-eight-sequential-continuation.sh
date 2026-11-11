set -euo pipefail
# No extra concurrent allocation: submission happens only after a timeout,
# and the new job cannot start until the original allocation has ended.
if [ "$(cat queue/24-eight-mode-response.sh.exit)" = 124 ]; then
 test ! -e "$RUN/reports/anthracene_ht/20261110-b3lyp-def2svp/eight_states/raw/mode_response_failure.json"
 if ! test -e results/eight-continuation-job.txt; then
  job=$(sbatch --parsable --dependency="afterany:$SLURM_JOB_ID" examples/tdscf/anthracene_ht/slurm/run_eight_resume.sbatch)
  printf '%s\n' "$job" > results/eight-continuation-job.txt
 fi
fi
# Scientific failures are retained rather than automatically retried.
touch STOP
