#!/bin/bash
# Diagnostic (user 2026-11-10): set B's spin-polarization payback in bcc Fe. Smoke settings (xcsb.studies.fe_payback):
# a = 2.866 A, 100/400 Ry, 8^3 k, Gaussian 0.02 Ry, carrier start, conv_thr CONV_THR_BENCH. Admissible variants only.
#   bash scripts/perlmutter/job_fe_payback.sh <run dir>      (inside a 1-node allocation; resumable: finished runs are skipped)
# Four lanes in parallel (8 ranks x 4 cores each).
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
source "$REPO/scripts/perlmutter/env_xcsb.sh"
RUN=${1:?run dir}
mkdir -p "$RUN"
cd "$REPO"
git rev-parse HEAD > "$RUN/COMMIT"
cp "$XCSB_BUILD/BUILD_INFO.json" "$RUN/" 2>/dev/null
export OMP_NUM_THREADS=4 OMP_PLACES=cores OMP_PROC_BIND=close STEP_TIMEOUT=${STEP_TIMEOUT:-5400}
B=SCH-SI-B:v3.1
cat > "$RUN/queue.txt" <<EOF
Fe100 $B --payback-diag
Fe100 SCH-SI-A:v3.1
Fe100 $B@ups_p=0
Fe100 $B@rs0_pb=0
Fe100 $B@ups_p=0.5
Fe100 $B@rs0_pb=0.375
Fe100 $B@rs0_pb=1.5
Fe100 $B@d=0,rs0_pb=0
Fe100-NM $B
Fe100-NM SCH-SI-A:v3.1
Fe100-NM $B@rs0_pb=0
EOF
one() {   # case name [flags]
  local case=$1 name=$2; shift 2
  local tag="${case}__${name//[:@,=]/_}"
  local safe=${name//:/_}; safe=${safe//@/__}; safe=${safe//,/_}; safe=${safe//=/}     # = xcsb.tiers.run_dir
  local rec="$RUN/${case}__hooked__${safe}/record.json"
  if [ -f "$rec" ] && grep -q '"status": "ok"' "$rec"
  then echo "$tag skip (done)" >> "$RUN/summary.txt"; return; fi
  local t0=$(date +%s)
  timeout "$STEP_TIMEOUT" srun -N 1 -n 8 -c 8 --exact --cpu-bind=cores python -m xcsb.cli scf --case "$case" --mode hooked \
      --name "$name" --root "$RUN" "$@" > "$RUN/$tag.log" 2>&1
  echo "$tag rc=$? $(( $(date +%s) - t0 ))s" >> "$RUN/summary.txt"
}
# smoke of the frozen attribution on a synthetic polarized density (fails fast before the queue; result logged only)
timeout 900 srun -N 1 -n 1 -c 32 --exact --cpu-bind=cores env OMP_NUM_THREADS=16 python - > "$RUN/attr_smoke.log" 2>&1 <<'PY'
import json, sys
sys.path.insert(0, "tests")
from test_v2_potential_fd import setup
from xcsb.studies.fe_payback import attribution
g, rho, tau = setup("triclinic")
a = attribution("SCH-SI-B:v3.1", g, rho, tau)
print(json.dumps({k: a[k] for k in ("E_payback_ry", "d_Ex_ry", "m_weighted_mean")}, default=float))
print(json.dumps(a["factors"]["dw_switch_shift"], default=float))
PY
echo "attr_smoke rc=$?" >> "$RUN/summary.txt"
export -f one
export RUN STEP_TIMEOUT
xargs -P 4 -L 1 bash -c 'one "$@"' _ < "$RUN/queue.txt"
python -m xcsb.cli payback-table --root "$RUN" > "$RUN/table.txt" 2>&1
echo "done $(date -Is)" >> "$RUN/summary.txt"
