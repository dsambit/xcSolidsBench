#!/bin/bash
# LiF: the exchange outer-switch floor lever (xcsb.studies.lif_response, xfloor_*; diagnostic). Inside ONE 1-node allocation of
# 30 min: SCH-SI-A:v3.2 at 6 lattice constants (10^3 k, 32 ranks x 1 thread, four side by side), each followed by the frozen
# integral I = sum_sigma int (1 - w)(e_iso - e_UEG) on its converged density (serial on the root).
#   bash scripts/perlmutter/job_lif_xfloor.sh <run dir> <xchole pin (40 hex)>
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
source "$REPO/scripts/perlmutter/env_xcsb.sh"
RUN=${1:?run dir}; PIN=${2:?xchole pin}
export XCSB_XCHOLE_SRC=$W/src/xchole-v3.2/src XCSB_XCHOLE_LINE=v3.2
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN/logs"
cd "$REPO"
JOB=${SLURM_JOB_ID:-nojob}
SUM="$RUN/summary.txt"
echo "job $JOB account ${SLURM_JOB_ACCOUNT:-unknown} xfloor start $(date -Is)" >> "$SUM"
XHEAD=$(git -C "$(dirname "$XCSB_XCHOLE_SRC")" rev-parse HEAD 2>/dev/null)
echo "$JOB $(git rev-parse HEAD) xchole $XHEAD line v3.2 xfloor $(date -Is)" >> "$RUN/COMMITS"
[ "$XHEAD" = "$PIN" ] || { echo "STOP: xchole clone is at '$XHEAD', not the pin $PIN" >> "$SUM"; exit 0; }
python -c "import os, xcsb, sys; f = os.path.abspath(xcsb.__file__); sys.exit(0 if f.startswith('$REPO/') else 1)" \
  || { echo "STOP: xcsb is not imported from $REPO" >> "$SUM"; exit 0; }
export OMP_PLACES=cores OMP_PROC_BIND=close OMP_NUM_THREADS=1
n=0
for step in $(python -c "from xcsb.studies import lif_response as R; print(' '.join(map(str, R.STEPS)))"); do
  args=$(python -c "from xcsb.studies import lif_response as R; print(' '.join(R.xfloor_command('$RUN', $step)))")
  ( timeout --signal=TERM --kill-after=30 1200 srun -N 1 -n 32 -c 2 --exact --cpu-bind=cores python -m xcsb.cli $args \
      < /dev/null > "$RUN/logs/xfloor_$step.log" 2>&1; echo "  xfloor step $step rc=$? $(date -Is)" >> "$SUM" ) &
  n=$(( n + 1 ))
  [ $(( n % 4 )) -eq 0 ] && wait
done
wait
python -m xcsb.cli lc xfloor-summary --run "$RUN" > "$RUN/xfloor_summary.md" 2>> "$SUM"
echo "job $JOB xfloor done $(date -Is)" >> "$SUM"
