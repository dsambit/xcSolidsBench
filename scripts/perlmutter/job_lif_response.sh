#!/bin/bash
# LiF lattice constant: frozen-density response to the model parameters (xcsb.studies.lif_response; diagnostic). Inside ONE
# 1-node allocation of 2 h, from the clone this script lives in: (1) the norms of the norm-dependent variants, re-solved side by
# side (serial processes, one core each); (2) SCH-SI-A:v3.2 at 6 lattice constants (10^3 k, 32 ranks x 1 thread, four side by
# side), each followed by the E_xc of every variant on its converged density. Finished points are skipped on a relaunch.
#   bash scripts/perlmutter/job_lif_response.sh <run dir> <xchole pin (40 hex)>
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
source "$REPO/scripts/perlmutter/env_xcsb.sh"
RUN=${1:?run dir}; PIN=${2:?xchole pin}
export XCSB_XCHOLE_SRC=$W/src/xchole-v3.2/src XCSB_XCHOLE_LINE=v3.2
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN/logs" "$RUN/prep"
cd "$REPO"
JOB=${SLURM_JOB_ID:-nojob}
END=$(( $(date +%s) + ${XCSB_ALLOC_S:-7200} - 150 ))
SUM="$RUN/summary.txt"
echo "job $JOB account ${SLURM_JOB_ACCOUNT:-unknown} resp start $(date -Is)" >> "$SUM"
XHEAD=$(git -C "$(dirname "$XCSB_XCHOLE_SRC")" rev-parse HEAD 2>/dev/null)
echo "$JOB $(git rev-parse HEAD) xchole $XHEAD line v3.2 resp $(date -Is)" >> "$RUN/COMMITS"
[ "$XHEAD" = "$PIN" ] || { echo "STOP: xchole clone is at '$XHEAD', not the pin $PIN" >> "$SUM"; exit 0; }
python -c "import os, xcsb, sys; f = os.path.abspath(xcsb.__file__); sys.exit(0 if f.startswith('$REPO/') else 1)" \
  || { echo "STOP: xcsb is not imported from $REPO" >> "$SUM"; exit 0; }
export OMP_PLACES=cores OMP_PROC_BIND=close OMP_NUM_THREADS=1
python -c "from xcsb.studies import lif_response as R; print('\n'.join(R.frozen_names()))" > "$RUN/frozen.txt"
python -c "from xcsb.studies import lif_response as R; print('\n'.join(R.variant_names()))" > "$RUN/variants.txt"
i=0
while read -r v; do
  i=$(( i + 1 ))
  srun -N 1 -n 1 -c 4 --exact python -m xcsb.cli prep-variant "$v" < /dev/null > "$RUN/prep/$i.log" 2>&1 &
done < "$RUN/variants.txt"
wait
cat "$RUN"/prep/*.log | grep -E '^\{' > "$RUN/prep.log"
echo "prep done $(date -Is): $(grep -c cache "$RUN/prep.log") cached, $(grep -c refused "$RUN/prep.log") refused" >> "$SUM"
n=0
for step in $(python -c "from xcsb.studies import lif_response as R; print(' '.join(map(str, R.STEPS)))"); do
  python - "$RUN" "$step" <<'PYDONE' && continue
import sys
from xcsb.studies import lif_response as R
r = R.record(sys.argv[1], int(sys.argv[2]))
sys.exit(0 if r is not None and r.get("status") == "ok" and len(r.get("frozen", {})) == len(R.frozen_names()) else 1)
PYDONE
  args=$(python -c "from xcsb.studies import lif_response as R; print(' '.join(R.command('$RUN', $step)))")
  lim=$(( END - $(date +%s) )); [ "$lim" -ge 300 ] || { echo "step $step: skipped, $lim s left" >> "$SUM"; continue; }
  ( timeout --signal=TERM --kill-after=30 "$lim" srun -N 1 -n 32 -c 2 --exact --cpu-bind=cores python -m xcsb.cli $args \
      < /dev/null > "$RUN/logs/step_$step.log" 2>&1; echo "  step $step rc=$? $(date -Is)" >> "$SUM" ) &
  n=$(( n + 1 ))
  [ $(( n % 4 )) -eq 0 ] && wait
done
wait
python -m xcsb.cli lc resp-summary --run "$RUN" > "$RUN/resp_summary.md" 2>> "$SUM"
echo "job $JOB done $(date -Is)" >> "$SUM"
