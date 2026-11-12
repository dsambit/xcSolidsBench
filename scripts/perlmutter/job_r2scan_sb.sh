#!/bin/bash
# Self-consistent spin-balanced r2SCAN (xcsb.studies.r2scan_sb; strongxc-v4 line, xchole line v4-sb). One 1-node allocation:
# phase 0 = the energy-vs-potential FD check of r2SCAN-SB and r2SCAN-SB-kin (tests/test_external_fd.py; a failure STOPS the job),
# then PBE, r2SCAN, r2SCAN-SB, r2SCAN-SB-kin on bcc Fe and fcc Ni (free moment) and Fe at m = 2.2, 2.4, 2.6 (4 lanes), then the table.
#   bash scripts/perlmutter/job_r2scan_sb.sh <run dir>
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
source "$REPO/scripts/perlmutter/env_xcsb.sh"
source "$REPO/scripts/perlmutter/env_xcsb_v4.sh"
export XCSB_XCHOLE_SRC=$W/src/xchole-v4-sb/src XCSB_XCHOLE_LINE=v4-sb
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
RUN=${1:?run dir}
mkdir -p "$RUN"
cd "$REPO"
JOB=${SLURM_JOB_ID:-nojob}
DEADLINE=$(( $(date +%s) + ${XCSB_ALLOC_S:-7200} - 300 ))
SUM="$RUN/summary.txt"
python -c "import os, xcsb, sys; sys.exit(0 if os.path.abspath(xcsb.__file__).startswith('$REPO/') else 1)" \
  || { echo "STOP: xcsb is not imported from $REPO" >> "$SUM"; exit 0; }
echo "$JOB $(git rev-parse HEAD) xchole $(git -C "$(dirname "$XCSB_XCHOLE_SRC")" rev-parse HEAD) line v4-sb $(date -Is)" >> "$RUN/COMMITS"
export OMP_NUM_THREADS=4 OMP_PLACES=cores OMP_PROC_BIND=close
if [ ! -f "$RUN/phase0_PASS" ]; then
  timeout 1800 srun -N 1 -n 1 -c 64 --cpu-bind=cores python -m pytest -q -m node tests/test_external_fd.py -p no:cacheprovider \
      > "$RUN/phase0_fd.log" 2>&1
  rc=$?
  tail -3 "$RUN/phase0_fd.log" >> "$SUM"
  if [ $rc -ne 0 ] || grep -q "skipped" "$RUN/phase0_fd.log"; then echo "STOP: phase 0 (FD potential check) failed rc=$rc" >> "$SUM"; exit 0; fi
  touch "$RUN/phase0_PASS"
fi
python -m xcsb.campaign --study r2scan_sb --run "$RUN" --deadline "$DEADLINE" --slots 4 --stages S >> "$RUN/campaign_$JOB.out" 2>&1
echo "campaign rc=$? $(date -Is)" >> "$SUM"
python -c "
import json
from xcsb.studies import r2scan_sb as S
t = S.table('$RUN'); json.dump(t, open('$RUN/sb_table.json', 'w'), indent=1, default=float); print(json.dumps(t, indent=1, default=float))
" > "$RUN/sb_table.txt" 2>> "$SUM"
echo "done $(date -Is)" >> "$SUM"
