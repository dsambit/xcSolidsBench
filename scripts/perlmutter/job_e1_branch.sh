#!/bin/bash
# B + E1 (half) SCF branches between omega 90 and 150 on free Fe (xcsb.studies.fe_e1_branch; xchole line v4-e1). The E1 FD check
# passed in the e1sc run dir (its phase0_PASS is required). 4 quarter-node lanes (campaign stage B), then the analysis.
#   bash scripts/perlmutter/job_e1_branch.sh <run dir> <e1sc run dir>
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
source "$REPO/scripts/perlmutter/env_xcsb.sh"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export XCSB_XCHOLE_SRC=$W/src/xchole-v4-e1/src XCSB_XCHOLE_LINE=v4-e1 XCSB_LINE=v4
RUN=${1:?run dir}; E1SC=${2:?e1sc run dir}
export XCSB_E1SC_RUN=$E1SC
mkdir -p "$RUN"
cd "$REPO"
JOB=${SLURM_JOB_ID:-nojob}
DEADLINE=$(( $(date +%s) + ${XCSB_ALLOC_S:-7200} - 300 ))
SUM="$RUN/summary.txt"
python -c "import os, xcsb, sys; sys.exit(0 if os.path.abspath(xcsb.__file__).startswith('$REPO/') else 1)" \
  || { echo "STOP: xcsb is not imported from $REPO" >> "$SUM"; exit 0; }
[ -f "$E1SC/phase0_PASS" ] || { echo "STOP: no E1 FD PASS in $E1SC" >> "$SUM"; exit 0; }
echo "$JOB $(git rev-parse HEAD) xchole $(git -C $W/src/xchole-v4-e1 rev-parse HEAD) $(date -Is)" >> "$RUN/COMMITS"
export OMP_NUM_THREADS=4 OMP_PLACES=cores OMP_PROC_BIND=close
python -m xcsb.campaign --study fe_e1_branch --run "$RUN" --deadline "$DEADLINE" --slots 4 --stages B >> "$RUN/campaign_$JOB.out" 2>&1
echo "campaign rc=$? $(date -Is)" >> "$SUM"
python -c "
import json
from xcsb.studies import fe_e1_branch as Bm
a = Bm.analysis('$RUN'); json.dump(a, open('$RUN/branch_analysis.json', 'w'), indent=1, default=float); print(json.dumps({k: a[k] for k in ('ground_m_of_omega', 'target_crossing', 'E_of_m_lowest', 'local_minima_m')}, indent=1, default=float))
" > "$RUN/branch_analysis.txt" 2>> "$SUM"
echo "done $(date -Is)" >> "$SUM"
