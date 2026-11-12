#!/bin/bash
# One production warm-start rerun of an FSM point (user 2026-11-11: Fe80-fsm-2.8, SCH-SI-B:v3.1, from the converged m = 2.6
# point): QE reads that run's wavefunctions, rho and tau (startingwfc/startingpot 'file'), the hook starts directly; same
# settings as the FSM job (80/320 Ry, 8^3 k, Gaussian 0.02, beta 0.3, conv_thr 1e-8, 8 ranks, one pool, --fsm-parts). The
# failed cold record is kept (record_attempt1.json, summarized in the new record). Inside a 1-node allocation:
#   bash scripts/perlmutter/job_fe_fsm_warm.sh <FSM run dir> <case> <functional> <source case> "<label>"
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
source "$REPO/scripts/perlmutter/env_xcsb.sh"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
RUN=${1:?FSM run dir}; CASE=${2:?case}; F=${3:?functional}; SRC=${4:?source case}; LABEL=${5:?label}
cd "$REPO"
SUM="$RUN/summary_warm.txt"
python -c "import os, xcsb, sys; sys.exit(0 if os.path.abspath(xcsb.__file__).startswith('$REPO/') else 1)" \
  || { echo "STOP: xcsb is not imported from $REPO" >> "$SUM"; exit 0; }
SAFE=${F//:/_}
echo "job ${SLURM_JOB_ID:-nojob} $(git rev-parse HEAD) $CASE $F from $SRC ($LABEL) start $(date -Is)" >> "$SUM"
export OMP_NUM_THREADS=4 OMP_PLACES=cores OMP_PROC_BIND=close
t0=$(date +%s)
timeout 3000 srun -N 1 -n 8 -c 8 --exact --cpu-bind=cores python -m xcsb.cli scf --case "$CASE" --mode hooked --name "$F" \
    --root "$RUN" --fsm-parts --warm-from "$RUN/${SRC}__hooked__$SAFE" --warm-label "$LABEL" > "$RUN/${CASE}__${SAFE}__warm.log" 2>&1
echo "$CASE $F warm rc=$? $(( $(date +%s) - t0 ))s" >> "$SUM"
python - "$RUN/${CASE}__hooked__$SAFE/record.json" >> "$SUM" 2>&1 <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
h = r.get("history", [])
print(f"status {r.get('status')} start {r.get('start')!r} iterations {len(h)} E {r.get('energies_ry', {}).get('etot')} "
      f"m {r.get('magnetization')} error {r.get('error')}")
print("dr2 history:", [f"{x['dr2']:.2e}" for x in h])
print("previous attempts:", [(a["attempt"], a["status"], a["hook_iterations"]) for a in r.get("previous_attempts", [])])
PY
echo "done $(date -Is)" >> "$SUM"
