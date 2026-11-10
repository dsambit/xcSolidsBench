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
# STEPS: space-separated subset (default: all). Every SCF is its own srun (QE keeps functional state per process).
STEPS=${STEPS:-"build_check pytest_laptop pytest_node V3 V4 V5_np8 V5_np32 V6 V6_np1"}
has() { [[ " $STEPS " == *" $1 "* ]]; }
SCF="python -m xcsb.cli scf"
has build_check   && step build_check   $S1 bash scripts/perlmutter/build_env.sh check
cp "$XCSB_BUILD/BUILD_INFO.json" "$RUN/" 2>/dev/null
has debug_fd      && step debug_fd      env OMP_NUM_THREADS=32 $S1 python scripts/debug/xchole_point_fd.py "$RUN/debug_fd_A.json" SCH-SI-A:v3.1
has pytest_laptop && step pytest_laptop $S1 python -m pytest -q -m "not node" tests
has pytest_node   && step pytest_node   env OMP_NUM_THREADS=32 $S1 python -m pytest -q -m node -rA tests
if has V3; then
  for c in Si Fe; do for x in PBE TPSS; do
    step V3_${c}_${x}_native $S8 $SCF --case $c --mode native --name $x --host-check $x --root "$RUN/V3"
    step V3_${c}_${x}_hooked $S8 $SCF --case $c --mode hooked --name $x --root "$RUN/V3"
  done; done
  for x in PBE TPSS; do step V3_Si-lowsym_${x}_native $S8 $SCF --case Si-lowsym --mode native --name $x --host-check $x --root "$RUN/V3"; done
  step V3_compare python -m xcsb.cli compare v3 --root "$RUN/V3"
fi
if has V4; then
  for x in PBE r2SCAN; do
    step V4_Si_${x} $S8 $SCF --case Si --mode hooked --name $x --root "$RUN/V4"
    step V4_Si-nspin2-m0_${x} $S8 $SCF --case Si-nspin2-m0 --mode hooked --name $x --root "$RUN/V4"
  done
  step V4_compare python -m xcsb.cli compare v4 --root "$RUN/V4"
fi
has V5_np8        && step V5_np8        $S8 python -m xcsb.cli tier v5 --out "$RUN/V5"
has V5_np32       && step V5_np32       env OMP_NUM_THREADS=4 $S32 python -m xcsb.cli tier v5 --out "$RUN/V5"
if has V6; then
  for c in Si Fe; do for x in PBE r2SCAN SCH-SI-A:v3.1 SCH-SI-B:v3.1; do
    step V6_${c}_${x/:/_} $S8 $SCF --case $c --mode hooked --name $x --root "$RUN/V6"
  done; done
  step V6_compare python -m xcsb.cli compare v6 --root "$RUN/V6"
fi
if has V6_np1; then
  step V6_np1_Si_PBE env OMP_NUM_THREADS=32 $S1 $SCF --case Si --mode hooked --name PBE --root "$RUN/V6_np1"
  step V6_np1_compare python -m xcsb.cli compare v6 --root "$RUN/V6_np1"
fi
echo "done $(date -Is)" >> "$SUMMARY"
