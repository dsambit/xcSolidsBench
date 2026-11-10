#!/bin/bash
# M0 verification tiers on one compute node (inside a 1-node interactive allocation; self-terminating).
#   bash scripts/perlmutter/job_m0_tiers.sh <run dir>
# Phases (each runs even if an earlier one failed; the summary lists every exit code):
#   node pytest (xchole: V1 sanity points, V2a, V2b), V3, V4, V5 (8 and 32 ranks), V6 (8 ranks), V6 Si/PBE at 1 rank.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
source "$REPO/scripts/perlmutter/env_xcsb.sh"
RUN=${1:?run dir}
mkdir -p "$RUN"
cd "$REPO"
git rev-parse HEAD > "$RUN/COMMIT"
cp "$XCSB_BUILD/BUILD_INFO.json" "$RUN/" 2>/dev/null
export OMP_NUM_THREADS=16 OMP_PLACES=threads OMP_PROC_BIND=spread
SUMMARY=$RUN/summary.txt
: > "$SUMMARY"
step() {   # name, command...
  local name=$1; shift
  local t0=$(date +%s)
  timeout ${STEP_TIMEOUT:-1800} "$@" > "$RUN/$name.log" 2>&1
  local rc=$?
  echo "$name rc=$rc $(( $(date +%s) - t0 ))s" | tee -a "$SUMMARY"
}
S8="srun -N 1 -n 8 -c 32 --cpu-bind=cores"
S32="srun -N 1 -n 32 -c 8 --cpu-bind=cores"
S1="srun -N 1 -n 1 -c 32 --cpu-bind=cores"
step build_check     $S1 bash scripts/perlmutter/build_env.sh check
cp "$XCSB_BUILD/BUILD_INFO.json" "$RUN/" 2>/dev/null
step pytest_laptop   $S1 python -m pytest -q -m "not node" tests
step pytest_node     env OMP_NUM_THREADS=32 $S1 python -m pytest -q -m node tests
step V3              $S8 python -m xcsb.cli tier v3 --out "$RUN/V3"
step V4              $S8 python -m xcsb.cli tier v4 --out "$RUN/V4"
step V5_np8          $S8 python -m xcsb.cli tier v5 --out "$RUN/V5"
step V5_np32         env OMP_NUM_THREADS=4 $S32 python -m xcsb.cli tier v5 --out "$RUN/V5"
step V6_np8          $S8 python -m xcsb.cli tier v6 --out "$RUN/V6"
step V6_np1_SiPBE    env OMP_NUM_THREADS=32 $S1 python -m xcsb.cli tier v6 --out "$RUN/V6" --functionals PBE --systems Si
echo "done $(date -Is)" >> "$SUMMARY"
