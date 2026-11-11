#!/bin/bash
# Fixed-spin-moment E(m) curves of bcc Fe (user 2026-11-11), 80/320 Ry, 8^3 k, Gaussian 0.02 Ry, a = 2.866 A, carrier start,
# conv_thr CONV_THR_BENCH, beta 0.3; QE tot_magnetization (two Fermi levels). Every point with --fsm-parts (E_x, E_c, pieces,
# switch means, frozen pairing test). Phase 0 checks the constraint against the free SCF of A and B at their own moments
# and STOPS the job if the FSM point does not reproduce the free energy.
#   bash scripts/perlmutter/job_fe_fsm.sh <run dir>      (inside a 1-node allocation; resumable)
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
source "$REPO/scripts/perlmutter/env_xcsb.sh"
RUN=${1:?run dir}
mkdir -p "$RUN"
cd "$REPO"
git rev-parse HEAD > "$RUN/COMMIT"
git -C "$(dirname "$XCSB_XCHOLE_SRC")" rev-parse HEAD > "$RUN/XCHOLE_COMMIT"     # pinned >= f9a275025 (P1/P2 reference)
export OMP_NUM_THREADS=4 OMP_PLACES=cores OMP_PROC_BIND=close STEP_TIMEOUT=${STEP_TIMEOUT:-3600}
SUM="$RUN/summary.txt"

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
export RUN SUM STEP_TIMEOUT

# ---- phase 0: the constraint vs the free SCF (B 2.7400, A 2.7646 muB at 80 Ry, job 61128680) -----------------------------
one Fe80-fsm-2.7400 SCH-SI-B:v3.1 &
one Fe80-fsm-2.7646 SCH-SI-A:v3.1 &
wait
python - "$RUN" >> "$SUM" 2>&1 <<'PYCHK'
import json, os, sys
run = sys.argv[1]
free = {"SCH-SI-B_v3.1": (-236.80694134165267, "2.7400"), "SCH-SI-A_v3.1": (-236.85674818708483, "2.7646")}
ok = True
for name, (e_free, m) in free.items():
    p = os.path.join(run, f"Fe80-fsm-{m}__hooked__{name}", "record.json")
    r = json.load(open(p)) if os.path.exists(p) else {}
    if r.get("status") != "ok":
        print(f"FSM check {name}: not converged ({r.get('status')}, {r.get('error')})"); ok = False; continue
    d = r["energies_ry"]["etot"] - e_free
    print(f"FSM check {name} at m = {m}: E_fsm - E_free = {d:.3e} Ry, m_fsm = {r['magnetization']['total']:.4f}")
    ok = ok and abs(d) <= 1e-4
print("FSM check", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PYCHK
if [ $? -ne 0 ]; then echo "STOP: the FSM constraint does not reproduce the free SCF" >> "$SUM"; echo "done $(date -Is)" >> "$SUM"; exit 0; fi

# ---- phase 1: the curves -----------------------------------------------------------------------------------------------------
{
  for f in SCH-SI-B:v3.1 SCH-SI-A:v3.1 "SCH-SI-B:v3.1#ueg-end" "SCH-SI-A:v3.1#ueg-end" PBE; do
    for m in 0.0 0.5 1.0 1.5 2.0 2.2 2.4 2.6 2.8 3.0; do echo "Fe80-fsm-$m $f --fsm-parts"; done
  done
  echo "Fe80 PBE --fsm-parts"
  for m in 0.0 0.5 1.0 1.5 2.0 2.2 2.4 2.6 2.8 3.0; do echo "Fe80-fsm-$m r2SCAN --fsm-parts"; done
  echo "Fe80 r2SCAN --fsm-parts"
  # small-m points for the curvature at m = 0 (spin stiffness; user 2026-11-11), behind the main grid
  for f in SCH-SI-B:v3.1 SCH-SI-A:v3.1 "SCH-SI-B:v3.1#ueg-end" "SCH-SI-A:v3.1#ueg-end" PBE r2SCAN; do
    for m in 0.1 0.2 0.3; do echo "Fe80-fsm-$m $f --fsm-parts"; done
  done
} > "$RUN/queue.txt"
xargs -P 4 -L 1 bash -c 'one "$@"' _ < "$RUN/queue.txt"
echo "done $(date -Is)" >> "$SUM"
