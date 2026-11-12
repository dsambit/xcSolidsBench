#!/bin/bash
# AFM-II NiO / MnO GKS gaps (xcsb.studies.tmo_gap; v4, xchole line v4-sb = strongxc-v4 b2501727c). One allocation per call:
#   bash scripts/perlmutter/job_tmo_gap.sh <run dir> <stages: P,G | F>
# Phase 0 (first allocation): the r2SCAN-SB / -kin energy-vs-potential FD check (tests/test_external_fd.py); a failure STOPS.
# Cells: the r2SCAN-relaxed AFM cells of Swathilakshmi et al. (tmo_gap.CELLS; the same geometry for every functional).
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
source "$REPO/scripts/perlmutter/env_xcsb.sh"
source "$REPO/scripts/perlmutter/env_xcsb_v4.sh"
export XCSB_XCHOLE_SRC=$W/src/xchole-v4-sb/src XCSB_XCHOLE_LINE=v4-sb
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
RUN=${1:?run dir}; STAGES=${2:?stages}
mkdir -p "$RUN"
cd "$REPO"
JOB=${SLURM_JOB_ID:-nojob}
DEADLINE=$(( $(date +%s) + ${XCSB_ALLOC_S:-14400} - 420 ))
SUM="$RUN/summary.txt"
python -c "import os, xcsb, sys; sys.exit(0 if os.path.abspath(xcsb.__file__).startswith('$REPO/') else 1)" \
  || { echo "STOP: xcsb is not imported from $REPO" >> "$SUM"; exit 0; }
echo "$JOB $(git rev-parse HEAD) xchole $(git -C "$(dirname "$XCSB_XCHOLE_SRC")" rev-parse HEAD) line v4-sb stages $STAGES $(date -Is)" >> "$RUN/COMMITS"
export OMP_NUM_THREADS=4 OMP_PLACES=cores OMP_PROC_BIND=close
if [ ! -f "$RUN/phase0_PASS" ]; then
  timeout 1800 srun -N 1 -n 1 -c 64 --cpu-bind=cores python -m pytest -q -m node tests/test_external_fd.py -p no:cacheprovider \
      > "$RUN/phase0_fd.log" 2>&1
  rc=$?
  tail -3 "$RUN/phase0_fd.log" >> "$SUM"
  if [ $rc -ne 0 ] || grep -q "skipped" "$RUN/phase0_fd.log"; then echo "STOP: phase 0 (FD potential check) failed rc=$rc" >> "$SUM"; exit 0; fi
  touch "$RUN/phase0_PASS"
fi
python -m xcsb.campaign --study tmo_gap --run "$RUN" --deadline "$DEADLINE" --slots 4 --stages "$STAGES" >> "$RUN/campaign_$JOB.out" 2>&1
echo "campaign $STAGES rc=$? $(date -Is)" >> "$SUM"
python -c "
import json
from xcsb.studies import tmo_gap as T
t = T.table('$RUN'); json.dump(t, open('$RUN/tmo_table.json', 'w'), indent=1, default=float)
" 2>> "$SUM"
echo "done $(date -Is)" >> "$SUM"
