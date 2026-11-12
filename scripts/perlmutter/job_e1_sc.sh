#!/bin/bash
# Self-consistent E1 on free Fe + r2SCAN-SB at omega 10 for Ni and the Fe FSM points (user go 2026-11-12). One allocation:
#   phase 0  FD potential check (scripts/fd_check.py) of SCH-SI-A/B:v3.1 + E1(omega=10) (R0 half, the model form) and
#            + E1kin(omega=10) (comparison); a failure STOPS the E1 phase (xchole line v4-e1);
#   phase 1  r2SCAN-SB / -kin at omega 10: Ni (carrier beta 0.1, stage R: only its SB tasks are pending) and Fe at m = 2.2, 2.4,
#            2.6 (stage T), in the r2SCAN-SB run dir (xchole line v4-sb);
#   phase 2  xcsb.studies.fe_e1_sc: installed A/B and A/B + E1 / E1kin at omega 10, 30 on SB-Fe (a 2.858 A); tables.
#   bash scripts/perlmutter/job_e1_sc.sh <E1 run dir> <r2SCAN-SB run dir>
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
source "$REPO/scripts/perlmutter/env_xcsb.sh"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
RUN=${1:?E1 run dir}; SBRUN=${2:?r2SCAN-SB run dir}
mkdir -p "$RUN"
cd "$REPO"
JOB=${SLURM_JOB_ID:-nojob}
DEADLINE=$(( $(date +%s) + ${XCSB_ALLOC_S:-7200} - 300 ))
SUM="$RUN/summary.txt"
python -c "import os, xcsb, sys; sys.exit(0 if os.path.abspath(xcsb.__file__).startswith('$REPO/') else 1)" \
  || { echo "STOP: xcsb is not imported from $REPO" >> "$SUM"; exit 0; }
export OMP_NUM_THREADS=4 OMP_PLACES=cores OMP_PROC_BIND=close XCSB_LINE=v4
E1ENV="XCSB_XCHOLE_SRC=$W/src/xchole-v4-e1/src XCSB_XCHOLE_LINE=v4-e1"
SBENV="XCSB_XCHOLE_SRC=$W/src/xchole-v4-sb/src XCSB_XCHOLE_LINE=v4-sb"
echo "$JOB $(git rev-parse HEAD) xchole e1 $(git -C $W/src/xchole-v4-e1 rev-parse HEAD) sb $(git -C $W/src/xchole-v4-sb rev-parse HEAD) $(date -Is)" >> "$RUN/COMMITS"
# ---- phase 0 -----------------------------------------------------------------------------------------------------------------
if [ ! -f "$RUN/phase0_PASS" ]; then
  for s in A B; do for v in E1 E1kin; do
    env $E1ENV timeout 1500 srun -N 1 -n 1 -c 64 --exact --cpu-bind=cores python scripts/fd_check.py "SCH-SI-$s:v3.1+$v(omega=10)" \
        > "$RUN/phase0_fd_${s}_$v.log" 2>&1 &
  done; done
  wait
  ok=1
  for f in "$RUN"/phase0_fd_*.log; do [ "$(grep -c 'FD pass' "$f")" = 2 ] || ok=0; echo "$(basename $f): $(grep -c 'FD pass' "$f") pass" >> "$SUM"; done
  [ $ok = 1 ] && touch "$RUN/phase0_PASS" || echo "phase 0 FAILED: the E1 phase will not run" >> "$SUM"
fi
# ---- phase 1: r2SCAN-SB at omega 10 (Ni, Fe FSM points) ------------------------------------------------------------------------
if [ -f "$SBRUN/phase0_PASS" ]; then
  env $SBENV python -m xcsb.campaign --study r2scan_sb --run "$SBRUN" --deadline "$DEADLINE" --slots 4 --stages R,T \
      >> "$SBRUN/campaign_$JOB.out" 2>&1
  env $SBENV python -c "
import json
from xcsb.studies import r2scan_sb as S
t = S.table('$SBRUN'); json.dump(t, open('$SBRUN/sb_table.json', 'w'), indent=1, default=float)
" 2>> "$SUM"
  echo "phase 1 done $(date -Is)" >> "$SUM"
fi
# ---- phase 2: SC E1 ----------------------------------------------------------------------------------------------------------
if [ -f "$RUN/phase0_PASS" ]; then
  env $E1ENV python -m xcsb.campaign --study fe_e1_sc --run "$RUN" --deadline "$DEADLINE" --slots 4 --stages E \
      >> "$RUN/campaign_$JOB.out" 2>&1
  env $E1ENV python -c "
import json
from xcsb.studies import fe_e1_sc as E
t = E.table('$RUN'); json.dump(t, open('$RUN/e1_table.json', 'w'), indent=1, default=float); print(json.dumps(t, indent=1, default=float))
" > "$RUN/e1_table.txt" 2>> "$SUM"
  echo "phase 2 done $(date -Is)" >> "$SUM"
fi
echo "done $(date -Is)" >> "$SUM"
