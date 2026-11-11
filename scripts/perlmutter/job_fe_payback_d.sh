#!/bin/bash
# Fe payback follow-up (user 2026-11-10): the alignment weight d. Same smoke settings as job_fe_payback.sh.
#   B with d = 0.5, 1.0, 0 at the installed r_s0 0.75, ups_p 1 (r_s0 = 0 is not a usable setting: user 2026-11-10);
#   d = 0 (with the frozen attribution): MAXITER 400 at beta 0.3; if it fails, one labelled diagnostic run at beta 0.1.
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
( run d0_beta0.3_maxiter400 --name "$B@d=0" --maxiter 400 --payback-diag \
  || run d0_beta0.1_maxiter400 --name "$B@d=0" --maxiter 400 --mixing-beta 0.1 --tag beta0.1 --payback-diag ) &
wait
echo "done $(date -Is)" >> "$RUN/summary.txt"
