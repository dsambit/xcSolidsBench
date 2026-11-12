#!/bin/bash
# Fit of the E1 gate width omega to our PBE moment of bcc Fe (xcsb.studies.fe_omega_fit; v4, xchole line v4-sb). One allocation:
# phase 0 = FD potential check of SCH-SI-A/B:v3.1+E1 at omega 90 (scripts/fd_check.py; a failure STOPS); the plan (target from the
# r2SCAN-SB run's PBE Fe record, bracket from the frozen omega scan; STOP if none / below omega 10); stage F (3 omega per set);
# omega* + the H check (STOP if not monotone / not reached / H fails); stage C (confirm at omega*); the table.
#   bash scripts/perlmutter/job_fe_omega_fit.sh <run dir>
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
source "$REPO/scripts/perlmutter/env_xcsb.sh"
source "$REPO/scripts/perlmutter/env_xcsb_v4.sh"
export XCSB_XCHOLE_SRC=$W/src/xchole-v4sb/src XCSB_XCHOLE_LINE=v4-sb
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
RUN=${1:?run dir}
mkdir -p "$RUN"
cd "$REPO"
JOB=${SLURM_JOB_ID:-nojob}
DEADLINE=$(( $(date +%s) + ${XCSB_ALLOC_S:-5400} - 300 ))
SUM="$RUN/summary.txt"
python -c "import os, xcsb, sys; sys.exit(0 if os.path.abspath(xcsb.__file__).startswith('$REPO/') else 1)" \
  || { echo "STOP: xcsb is not imported from $REPO" >> "$SUM"; exit 0; }
echo "$JOB $(git rev-parse HEAD) xchole $(git -C "$(dirname "$XCSB_XCHOLE_SRC")" rev-parse HEAD) line v4-sb $(date -Is)" >> "$RUN/COMMITS"
export OMP_NUM_THREADS=4 OMP_PLACES=cores OMP_PROC_BIND=close
if [ ! -f "$RUN/phase0_PASS" ]; then
  ok=1
  for s in A B; do
    timeout 1500 srun -N 1 -n 1 -c 64 --cpu-bind=cores python scripts/fd_check.py "SCH-SI-$s:v3.1+E1@omega=90" > "$RUN/phase0_fd_$s.log" 2>&1 &
  done
  wait
  for s in A B; do grep -c "FD pass" "$RUN/phase0_fd_$s.log" | grep -q 2 || ok=0; tail -2 "$RUN/phase0_fd_$s.log" >> "$SUM"; done
  [ $ok = 1 ] || { echo "STOP: phase 0 FD check failed" >> "$SUM"; exit 0; }
  touch "$RUN/phase0_PASS"
fi
python -c "from xcsb.studies import fe_omega_fit as F; import json; print(json.dumps(F.plan('$RUN')))" >> "$SUM" 2>&1 \
  || { echo "STOP: no fit plan (target or bracket; see above)" >> "$SUM"; exit 0; }
python -m xcsb.campaign --study fe_omega_fit --run "$RUN" --deadline "$DEADLINE" --slots 4 --stages F >> "$RUN/campaign_$JOB.out" 2>&1
if [ ! -f "$RUN/OMEGA_STAR.json" ]; then
  timeout 1200 srun -N 1 -n 1 -c 64 --cpu-bind=cores python -m xcsb.studies.fe_omega_fit "$RUN" >> "$SUM" 2>&1 \
    || { echo "STOP: omega* not determined ($(cat $RUN/STOP.txt 2>/dev/null))" >> "$SUM"; exit 0; }
fi
python -m xcsb.campaign --study fe_omega_fit --run "$RUN" --deadline "$DEADLINE" --slots 4 --stages C >> "$RUN/campaign_$JOB.out" 2>&1
echo "campaign rc=$? $(date -Is)" >> "$SUM"
echo "done $(date -Is)" >> "$SUM"
