#!/bin/bash
# v4 Fe item 2 (xcsb.studies.fe_sb_frozen): frozen spin-balance evaluation and the spin-gradient texture on the saved FSM states
# (no SCF), one process on the whole node. Resumable (per-point results cached in the run dir).
#   bash scripts/perlmutter/job_fe_sb.sh <run dir> <FSM run dir> [<dense FSM run dir>]
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
source "$REPO/scripts/perlmutter/env_xcsb.sh"
source "$REPO/scripts/perlmutter/env_xcsb_v4.sh"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
RUN=${1:?run dir}; F1=${2:?FSM run}; F2=${3:-}
mkdir -p "$RUN"
cd "$REPO"
JOB=${SLURM_JOB_ID:-nojob}
python -c "import os, xcsb, sys; sys.exit(0 if os.path.abspath(xcsb.__file__).startswith('$REPO/') else 1)" \
  || { echo "STOP: xcsb is not imported from $REPO" >> "$RUN/summary.txt"; exit 0; }
echo "$JOB $(git rev-parse HEAD) xchole $(git -C "$(dirname "$XCSB_XCHOLE_SRC")" rev-parse HEAD) line v4 $(date -Is)" >> "$RUN/COMMITS"
ARGS=(--fsm "$F1"); [ -n "$F2" ] && [ -d "$F2" ] && ARGS+=(--fsm "$F2")
timeout ${XCSB_STEP_S:-3300} srun -N 1 -n 1 -c 256 --cpu-bind=none python -m xcsb.studies.fe_sb_frozen "${ARGS[@]}" --out "$RUN" \
    > "$RUN/sb_$JOB.log" 2>&1
echo "fe_sb_frozen rc=$? $(date -Is)" >> "$RUN/summary.txt"
