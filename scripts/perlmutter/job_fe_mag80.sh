#!/bin/bash
# Fe magnetization analysis at 80/320 Ry (user 2026-11-10): reference A, B; d scan; UEG-gauge end alone; iso-end parameter
# sensitivity at the admissible range ends (norm-dependent edits re-solved once by xchole's external.load). Smoke settings:
# a = 2.866 A, 8^3 k, Gaussian 0.02 Ry, carrier start, conv_thr CONV_THR_BENCH, beta 0.3.
#   bash scripts/perlmutter/job_fe_mag80.sh <run dir>      (inside a 1-node allocation; resumable)
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
source "$REPO/scripts/perlmutter/env_xcsb.sh"
RUN=${1:?run dir}
mkdir -p "$RUN"
cd "$REPO"
git rev-parse HEAD > "$RUN/COMMIT"
export XCSB_NORM_CACHE=$W/cache/xchole_norms
export OMP_NUM_THREADS=4 OMP_PLACES=cores OMP_PROC_BIND=close STEP_TIMEOUT=${STEP_TIMEOUT:-3600}
A=SCH-SI-A:v3.1
B=SCH-SI-B:v3.1
SUM="$RUN/summary.txt"

# ---- phase 0: checks (UEG-end potential vs finite differences; the frozen split sums to the model) ------------------------
timeout 1200 srun -N 1 -n 1 -c 32 --exact --cpu-bind=cores env OMP_NUM_THREADS=16 python - > "$RUN/checks.log" 2>&1 <<'PYCHECK'
import json, sys
sys.path.insert(0, "tests")
from test_v2_potential_fd import setup, fd_spin
from xcsb.functional import get
from xcsb.studies.fe_magnetism import split
g, rho, tau = setup("triclinic")
rep = fd_spin(get("SCH-SI-B:v3.1#ueg-end"), g, rho, tau)
print("ueg-end V2a", {f"{s}{w}": min(e.values()) for (s, w), e in rep.items()}, flush=True)
sp = split("SCH-SI-B:v3.1", g, rho, tau)
print("split sum check (Ry)", sp["sum_pieces_minus_model_ry"], flush=True)
PYCHECK
echo "checks rc=$? (see checks.log)" >> "$SUM"

# ---- phase 1: norm re-solves of the norm-dependent variants (one process per name, 8 in parallel) -------------------------
printf '%s\n' \
  "$B@lam=0.8" "$B@a_L=3.0" "$B@eps=0.001" "$B@nc_kd=2.0" "$B@nc_kd=15.0" "$B@nc_ke=2.0" "$B@nc_ke=10.0" \
  "$B@beta_g=0.3" "$B@r_c=5.0" "$B@a_c=0.3" "$B@a_c=4.0" \
  "$A@lam=0.8" "$A@a_L=3.0" "$A@eps=0.001" "$A@eps=0.01" "$A@nc_kd=2.0" "$A@nc_kd=15.0" "$A@nc_ke=2.0" \
  "$A@beta_g=0.3" "$A@r_c=5.0" "$A@a_c=0.3" > "$RUN/prep.txt"
t0=$(date +%s)
xargs -P 8 -I{} timeout 2400 srun -N 1 -n 1 -c 16 --exact --cpu-bind=cores env OMP_NUM_THREADS=8 \
    python -m xcsb.cli prep-variant "{}" < "$RUN/prep.txt" > "$RUN/prep.log" 2>&1
echo "prep rc=$? $(( $(date +%s) - t0 ))s (see prep.log)" >> "$SUM"

# ---- phase 2: the SCF queue (4 lanes) --------------------------------------------------------------------------------------
{
  echo "Fe80 $B --mag-split --payback-diag"
  echo "Fe80 $A --mag-split"
  echo "Fe80 $B#ueg-end"
  echo "Fe80 $A#ueg-end"
  echo "Fe80 $B@d=0.5"
  echo "Fe80 $B@d=1.0"
  echo "Fe80 $B@d=0 --maxiter 400 --payback-diag"
  for e in lam=0.8 a_L=3.0 eps=0.001 c_min=0.1 nc_kd=2.0 nc_kd=15.0 nc_ke=2.0 nc_ke=10.0 beta_g=0.3 r_c=5.0 a_c=0.3 \
           a_c=4.0 kappa_r=0.0059 t_H=0.01 t_H=1.0 y_w=1e-08 y_w=100000000.0 b_rise=1.0; do echo "Fe80 $B@$e"; done
  for e in lam=0.8 a_L=3.0 eps=0.001 eps=0.01 c_min=0.11 nc_kd=2.0 nc_kd=15.0 nc_ke=2.0 beta_g=0.3 r_c=5.0 a_c=0.3 \
           kappa_r=0.0059 t_H=0.01 t_H=1.0 y_w=1e-08 y_w=100000000.0 b_rise=0.0 b_rise=1.0; do echo "Fe80 $A@$e"; done
} > "$RUN/queue.txt"

one() {   # case name [flags]
  local case=$1 name=$2; shift 2
  local tag="${case}__${name//[:@,=#]/_}"
  local safe=${name//:/_}; safe=${safe//@/__}; safe=${safe//,/_}; safe=${safe//=/}     # = xcsb.tiers.run_dir
  local rec="$RUN/${case}__hooked__${safe}/record.json"
  if [ -f "$rec" ] && grep -q '"status": "ok"' "$rec"; then echo "$tag skip (done)" >> "$SUM"; return; fi
  local t0=$(date +%s)
  timeout "$STEP_TIMEOUT" srun -N 1 -n 8 -c 8 --exact --cpu-bind=cores python -m xcsb.cli scf --case "$case" --mode hooked \
      --name "$name" --root "$RUN" "$@" > "$RUN/$tag.log" 2>&1
  echo "$tag rc=$? $(( $(date +%s) - t0 ))s" >> "$SUM"
}
export -f one
export RUN SUM STEP_TIMEOUT XCSB_NORM_CACHE
xargs -P 4 -L 1 bash -c 'one "$@"' _ < "$RUN/queue.txt"

# the d = 0 fallback: one labelled beta 0.1 diagnostic run if the beta 0.3 / MAXITER 400 run did not converge
D0="$RUN/Fe80__hooked__SCH-SI-B_v3.1__d0/record.json"
if ! { [ -f "$D0" ] && grep -q '"status": "ok"' "$D0"; }; then
  t0=$(date +%s)
  timeout "$STEP_TIMEOUT" srun -N 1 -n 8 -c 8 --exact --cpu-bind=cores python -m xcsb.cli scf --case Fe80 --mode hooked \
      --name "$B@d=0" --root "$RUN" --maxiter 400 --mixing-beta 0.1 --tag beta0.1 --payback-diag > "$RUN/d0_beta0.1.log" 2>&1
  echo "Fe80__B_d0_beta0.1 rc=$? $(( $(date +%s) - t0 ))s" >> "$SUM"
fi
echo "done $(date -Is)" >> "$SUM"
