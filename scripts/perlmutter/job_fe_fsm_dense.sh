#!/bin/bash
# v4 Fe item 1 (xcsb.studies.fe_fsm_dense): dense FSM scan m = 0.6-0.9 (A, B, A#ueg-end, PBE; cold) and the hysteresis starts (A, B
# from the m = 0.5 and 1.0 WAVEFUNCTIONS of the FSM run), 4 quarter-node lanes (campaign runner, stage D), then the table; then,
# if >= 50 min remain, item 2 (the frozen spin-balance evaluation) on the FSM run and these dense points.
#   bash scripts/perlmutter/job_fe_fsm_dense.sh <run dir> <item-2 run dir>
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
source "$REPO/scripts/perlmutter/env_xcsb.sh"
source "$REPO/scripts/perlmutter/env_xcsb_v4.sh"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
RUN=${1:?run dir}; SBRUN=${2:?item-2 run dir}
mkdir -p "$RUN"
cd "$REPO"
JOB=${SLURM_JOB_ID:-nojob}
T0=$(date +%s); DEADLINE=$(( T0 + ${XCSB_ALLOC_S:-7200} - 300 ))
SUM="$RUN/summary.txt"
python -c "import os, xcsb, sys; sys.exit(0 if os.path.abspath(xcsb.__file__).startswith('$REPO/') else 1)" \
  || { echo "STOP: xcsb is not imported from $REPO" >> "$SUM"; exit 0; }
echo "$JOB $(git rev-parse HEAD) xchole $(git -C "$(dirname "$XCSB_XCHOLE_SRC")" rev-parse HEAD) line v4 $(date -Is)" >> "$RUN/COMMITS"
export OMP_NUM_THREADS=4 OMP_PLACES=cores OMP_PROC_BIND=close
python -m xcsb.campaign --study fe_fsm_dense --run "$RUN" --deadline "$DEADLINE" --slots 4 --stages D >> "$RUN/campaign_$JOB.out" 2>&1
echo "campaign rc=$? $(date -Is)" >> "$SUM"
python -c "
from xcsb.studies import fe_fsm_dense as D
import json
t = D.table('$RUN'); json.dump(t, open('$RUN/dense_table.json', 'w'), indent=1, default=float); print(D.render(t))
" > "$RUN/dense_table.md" 2>> "$SUM"
if [ $(( DEADLINE - $(date +%s) )) -ge 3000 ]; then
  mkdir -p "$SBRUN"
  timeout $(( DEADLINE - $(date +%s) )) srun -N 1 -n 1 -c 256 --cpu-bind=none python -m xcsb.studies.fe_sb_frozen \
      --fsm "${XCSB_FSM_RUN:-/pscratch/sd/d/dsambit/xc_functional_dev/runs/xcsb/2026-11-11_fe_fsm}" --fsm "$RUN" --out "$SBRUN" \
      > "$SBRUN/sb_$JOB.log" 2>&1
  echo "item 2 rc=$? $(date -Is)" >> "$SUM"
else
  echo "item 2 not started (less than 50 min left)" >> "$SUM"
fi
echo "done $(date -Is)" >> "$SUM"
