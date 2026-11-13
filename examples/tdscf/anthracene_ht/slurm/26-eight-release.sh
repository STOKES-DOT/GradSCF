set -euo pipefail
[ "$(cat queue/25-eight-spectrum.sh.exit)" = 0 ]
touch STOP
