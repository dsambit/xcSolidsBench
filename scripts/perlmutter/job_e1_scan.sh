#!/bin/bash
# The SC omega scan of E1 on free Fe (user go 2026-11-12): SCH-SI-A/B:v3.1 + E1(omega=w), half split (the model form),
# w in 60, 90, 150, 300 (xcsb.studies.fe_e1_sc stage S, the e1sc run dir, xchole line v4-e1); phase 0 = the E1 FD check at
# omega 10 unless the run dir already has its PASS; then the table and m(omega) (fe_e1_sc.m_of_omega: target = the experimental
# TOTAL moment 2.22 (user decision 2026-11-12; spin vs total by the user's choice); our PBE 2.359 as the reference column).
#   bash scripts/perlmutter/job_e1_scan.sh <e1sc run dir>
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
source "$REPO/scripts/perlmutter/env_xcsb.sh"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export XCSB_XCHOLE_SRC=$W/src/xchole-v4-e1/src XCSB_XCHOLE_LINE=v4-e1 XCSB_LINE=v4
RUN=${1:?run dir}
mkdir -p "$RUN"
cd "$REPO"
JOB=${SLURM_JOB_ID:-nojob}
DEADLINE=$(( $(date +%s) + ${XCSB_ALLOC_S:-7200} - 300 ))
SUM="$RUN/summary_scan.txt"
python -c "import os, xcsb, sys; sys.exit(0 if os.path.abspath(xcsb.__file__).startswith('$REPO/') else 1)" \
  || { echo "STOP: xcsb is not imported from $REPO" >> "$SUM"; exit 0; }
echo "$JOB $(git rev-parse HEAD) xchole $(git -C $W/src/xchole-v4-e1 rev-parse HEAD) scan $(date -Is)" >> "$RUN/COMMITS"
export OMP_NUM_THREADS=4 OMP_PLACES=cores OMP_PROC_BIND=close
if [ ! -f "$RUN/phase0_PASS" ]; then
  for s in A B; do
    timeout 1500 srun -N 1 -n 1 -c 64 --exact --cpu-bind=cores python scripts/fd_check.py "SCH-SI-$s:v3.1+E1(omega=10)" \
        > "$RUN/phase0_scan_fd_$s.log" 2>&1 &
  done
  wait
  ok=1
  for s in A B; do [ "$(grep -c 'FD pass' "$RUN/phase0_scan_fd_$s.log")" = 2 ] || ok=0; done
  [ $ok = 1 ] || { echo "STOP: phase 0 FD check failed" >> "$SUM"; exit 0; }
fi
python -m xcsb.campaign --study fe_e1_sc --run "$RUN" --deadline "$DEADLINE" --slots 4 --stages S >> "$RUN/campaign_scan_$JOB.out" 2>&1
python -c "
import json
from xcsb.studies import fe_e1_sc as E
t = E.table('$RUN'); json.dump(t, open('$RUN/e1_table.json', 'w'), indent=1, default=float)
r = E.m_of_omega('$RUN'); json.dump(r, open('$RUN/m_of_omega.json', 'w'), indent=1, default=float); print(json.dumps(r, indent=1, default=float))
" > "$RUN/m_of_omega.txt" 2>> "$SUM"
echo "done $(date -Is)" >> "$SUM"
