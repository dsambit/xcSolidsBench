#!/bin/bash
# One 1-node allocation for two v4 items (user 2026-11-12):
#   phase 1  r2SCAN-SB stage R (xchole line v4-sb): the Ni rerun with the carrier beta 0.1 (recorded deviation) and free Fe at
#            omega 0 / 10 / 30 (half, kin) with the term diagnostics, in the r2SCAN-SB run dir (its phase-0 FD PASS is reused);
#   phase 2  the low-m cold FSM starts (xcsb.studies.fe_fsm_lowm; xchole line v3.1 as the FSM run): A m 0.1-0.5, B 0.2-0.5,
#            starting_magnetization 0.2 / 0.5 / 0.8;
#   then the tables and the lower-branch analysis (FSM run + dense run + low-m run).
#   bash scripts/perlmutter/job_v4_combo.sh <r2SCAN-SB run dir> <low-m run dir>
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
source "$REPO/scripts/perlmutter/env_xcsb.sh"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
SBRUN=${1:?r2SCAN-SB run dir}; LOWM=${2:?low-m run dir}
FSM=$W/runs/xcsb/2026-11-11_fe_fsm; DENSE=$W/runs/xcsb/v4_2026-11-12_fe_fsm_dense
mkdir -p "$LOWM"
cd "$REPO"
JOB=${SLURM_JOB_ID:-nojob}
DEADLINE=$(( $(date +%s) + ${XCSB_ALLOC_S:-7200} - 300 ))
SUM="$LOWM/summary.txt"
python -c "import os, xcsb, sys; sys.exit(0 if os.path.abspath(xcsb.__file__).startswith('$REPO/') else 1)" \
  || { echo "STOP: xcsb is not imported from $REPO" >> "$SUM"; exit 0; }
export OMP_NUM_THREADS=4 OMP_PLACES=cores OMP_PROC_BIND=close XCSB_LINE=v4
# ---- phase 1: r2SCAN-SB stage R ----------------------------------------------------------------------------------------------
if [ -f "$SBRUN/phase0_PASS" ]; then
  ( export XCSB_XCHOLE_SRC=$W/src/xchole-v4-sb/src XCSB_XCHOLE_LINE=v4-sb
    echo "$JOB $(git rev-parse HEAD) xchole $(git -C "$(dirname "$XCSB_XCHOLE_SRC")" rev-parse HEAD) line v4-sb stage R $(date -Is)" >> "$SBRUN/COMMITS"
    python -m xcsb.campaign --study r2scan_sb --run "$SBRUN" --deadline "$DEADLINE" --slots 4 --stages R >> "$SBRUN/campaign_$JOB.out" 2>&1
    python -c "
import json
from xcsb.studies import r2scan_sb as S
t = S.table('$SBRUN'); json.dump(t, open('$SBRUN/sb_table.json', 'w'), indent=1, default=float)
" )
  echo "phase 1 (r2SCAN-SB stage R) done $(date -Is)" >> "$SUM"
else
  echo "phase 1 skipped: no phase-0 FD PASS in $SBRUN" >> "$SUM"
fi
# ---- phase 2: the low-m cold FSM starts ----------------------------------------------------------------------------------------
echo "$JOB $(git rev-parse HEAD) xchole $(git -C "$(dirname "$XCSB_XCHOLE_SRC")" rev-parse HEAD) line v3.1 $(date -Is)" >> "$LOWM/COMMITS"
python -m xcsb.campaign --study fe_fsm_lowm --run "$LOWM" --deadline "$DEADLINE" --slots 4 --stages L >> "$LOWM/campaign_$JOB.out" 2>&1
echo "phase 2 campaign rc=$? $(date -Is)" >> "$SUM"
python scripts/fe_fsm_lower_branch.py "$FSM" "$DENSE" "$LOWM/lower_branch.json" "$LOWM" > "$LOWM/lower_branch.txt" 2>> "$SUM"
echo "done $(date -Is)" >> "$SUM"
