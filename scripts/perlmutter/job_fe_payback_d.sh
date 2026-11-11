#!/bin/bash
# Fe payback follow-up (user 2026-11-10): the alignment weight d. Same smoke settings as job_fe_payback.sh.
#   lanes 1-3: B with d = 0.5, 1.0, 0 at the installed r_s0 0.75, ups_p 1 (the d = 0 run with the frozen attribution);
#   lane 4: converge B with d = 0, r_s0 = 0: beta 0.3 with MAXITER 400; if it fails, one labelled diagnostic run at beta 0.1.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
source "$REPO/scripts/perlmutter/env_xcsb.sh"
RUN=${1:?run dir}
mkdir -p "$RUN"
cd "$REPO"
git rev-parse HEAD > "$RUN/COMMIT"
export OMP_NUM_THREADS=4 OMP_PLACES=cores OMP_PROC_BIND=close
B=SCH-SI-B:v3.1
run() {   # tag args...
  local tag=$1; shift; local t0=$(date +%s)
  timeout 7200 srun -N 1 -n 8 -c 8 --exact --cpu-bind=cores python -m xcsb.cli scf --case Fe100 --mode hooked --root "$RUN" "$@" \
      > "$RUN/$tag.log" 2>&1
  local rc=$?; echo "$tag rc=$rc $(( $(date +%s) - t0 ))s" >> "$RUN/summary.txt"; return $rc
}
run d0.5 --name "$B@d=0.5" &
run d1.0 --name "$B@d=1.0" &
run d0   --name "$B@d=0" --payback-diag &
( run d0_rs0_0_beta0.3_maxiter400 --name "$B@d=0,rs0_pb=0" --maxiter 400 --payback-diag \
  || run d0_rs0_0_beta0.1_maxiter400 --name "$B@d=0,rs0_pb=0" --maxiter 400 --mixing-beta 0.1 --tag beta0.1 --payback-diag ) &
wait
echo "done $(date -Is)" >> "$RUN/summary.txt"
