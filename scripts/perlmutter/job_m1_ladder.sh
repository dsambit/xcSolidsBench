#!/bin/bash
# M1 step 1, the cutoff ladder of bcc Fe and V (DESIGN.md Sec. 6.2; xcsb.studies.m1_ladder). Inside ONE 1-node allocation;
# resumable, and several allocations (at most 2 at a time) may work on the same run directory (task claims).
#   bash scripts/perlmutter/job_m1_ladder.sh <run dir> P|L
# Stage P (user 2026-11-11: the first allocation runs this ONLY, then stops and reports): phase 0 (~10 min: QE k-point pools,
# npool 8 vs 1 for PBE and SCH-SI-B, Fe 40 Ry 8^3 k, must give the same SCF; the ripple-seeded restart must run, PBE, all six
# seeds; a failure STOPS the job), then the PBE scan 40-140 Ry (E_c^PBE -> EC.json) and the full-node probe, then the report
# stage_p_report.{json,md} (measured times with pools, revised stage-L node-hours). Stage L (only after the user's decision;
# needs phase 0 PASS and EC.json): the ladder, the d = 8 points and the ripple seeds, until the deadline. The runner never
# starts a task of the other stage.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
source "$REPO/scripts/perlmutter/env_xcsb.sh"
RUN=${1:?run dir}
STAGE=${2:?stage P or L}
case $STAGE in P|L) ;; *) echo "stage must be P or L, got $STAGE" >&2; exit 2;; esac
mkdir -p "$RUN"
cd "$REPO"
JOB=${SLURM_JOB_ID:-nojob}
T0=$(date +%s)
DEADLINE=$(( T0 + ${XCSB_ALLOC_S:-14400} - 420 ))      # 7 min before the end of a 4 h allocation
echo "$JOB $(git rev-parse HEAD) $(date -Is)" >> "$RUN/COMMITS"
git -C "$(dirname "$XCSB_XCHOLE_SRC")" rev-parse HEAD > "$RUN/XCHOLE_COMMIT"
export OMP_NUM_THREADS=4 OMP_PLACES=cores OMP_PROC_BIND=close
SUM="$RUN/summary.txt"
echo "job $JOB stage $STAGE start $(date -Is) deadline $(date -d @$DEADLINE -Is)" >> "$SUM"
if [ "$STAGE" = L ] && { [ ! -f "$RUN/phase0/PASS" ] || [ ! -f "$RUN/EC.json" ]; }; then
  echo "STOP: stage L needs phase 0 PASS and EC.json (run stage P first)" >> "$SUM"; exit 0
fi

P0="$RUN/phase0"
# phase 0 (stage P allocations only; a stage-L job has exited above without PASS) runs once: the allocation that takes the lock runs it; another one waits for its verdict (a lock of a job that
# no longer exists is taken over)
while [ ! -f "$P0/PASS" ] && [ ! -f "$P0/FAIL" ]; do
  if mkdir "$RUN/phase0.lock" 2>/dev/null; then echo "$JOB" > "$RUN/phase0.lock/job"; break; fi
  owner=$(cat "$RUN/phase0.lock/job" 2>/dev/null)
  if [ -n "$owner" ] && ! squeue -h -j "$owner" 2>/dev/null | grep -q .; then rm -rf "$RUN/phase0.lock"; continue; fi
  sleep 30
done
if [ -f "$P0/FAIL" ]; then echo "phase 0 failed ($P0/FAIL): stop" >> "$SUM"; exit 0; fi
if [ ! -f "$P0/PASS" ]; then
  mkdir -p "$P0"
  C=M1-Fe-FM-40-d4-k8
  p0() {   # functional npool tag [flags]
    local f=$1 np=$2 tag=$3; shift 3
    timeout 1500 srun -N 1 -n 8 -c 8 --exact --cpu-bind=cores python -m xcsb.cli scf --case $C --mode hooked --name "$f" \
        --root "$P0" --npool "$np" --tag "$tag" "$@" > "$P0/${f//:/_}__$tag.log" 2>&1
    echo "phase0 $f $tag rc=$?" >> "$SUM"
  }
  p0 PBE 1 np1 & p0 PBE 8 np8 & p0 SCH-SI-B:v3.1 1 np1 & p0 SCH-SI-B:v3.1 8 np8 &
  wait
  p0 PBE 8 rip --ripple all --m1-diag & p0 SCH-SI-B:v3.1 8 rip --ripple all --m1-diag &
  wait
  python - "$P0" >> "$SUM" 2>&1 <<'PYCHK'
import json, os, sys
p0 = sys.argv[1]
C = "M1-Fe-FM-40-d4-k8"
def rec(f, tag):
    p = os.path.join(p0, f"{C}__hooked__{f.replace(':', '_')}__{tag}", "record.json")
    return json.load(open(p)) if os.path.exists(p) else {"status": "missing"}
ok = True
for f in ("PBE", "SCH-SI-B:v3.1"):
    a, b = rec(f, "np1"), rec(f, "np8")
    if a.get("status") != "ok" or b.get("status") != "ok":
        print(f"phase0 pools {f}: status {a.get('status')} / {b.get('status')}"); ok = False; continue
    de = b["energies_ry"]["etot"] - a["energies_ry"]["etot"]
    dm = b["magnetization"]["total"] - a["magnetization"]["total"]
    print(f"phase0 pools {f}: E(np8) - E(np1) = {de:.2e} Ry, m diff {dm:.2e} muB, iterations {a['iterations']} / {b['iterations']}")
    ok = ok and abs(de) <= 1e-6 and abs(dm) <= 1e-4
for f in ("PBE", "SCH-SI-B:v3.1"):
    r = rec(f, "rip")
    seeds = (r.get("ripple") or {}).get("seeds", [])
    for s in seeds:
        print(f"phase0 ripple {f} {s['seed']['label']}: {s.get('status')} it {s.get('iterations')} "
              f"dE {s.get('delta_etot_ry')} Ry, dExc(seed) {s.get('delta_e_xc_seed_ry')} Ry {s.get('error', '')}")
    if f == "PBE":                     # the gate: the seeded restart runs (QEpy setters, re-convergence); physics is reported
        good = r.get("status") == "ok" and len(seeds) == 6 and all(s.get("status") == "ok" for s in seeds)
        print(f"phase0 ripple PBE functional check: {'ok' if good else 'FAILED'} (record {r.get('status')}, {r.get('error')})")
        ok = ok and good
print("phase0", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PYCHK
  if [ $? -eq 0 ]; then touch "$P0/PASS"; else touch "$P0/FAIL"; echo "STOP: phase 0 failed" >> "$SUM"; exit 0; fi
fi

python -m xcsb.campaign --study m1_ladder --run "$RUN" --deadline "$DEADLINE" --slots 4 --stages "$STAGE" \
    >> "$RUN/campaign_$JOB.out" 2>&1
echo "campaign stage $STAGE rc=$? $(date -Is)" >> "$SUM"
if [ "$STAGE" = P ]; then
  python -m xcsb.cli m1 report-p --run "$RUN" > "$RUN/stage_p_report.md" 2>> "$SUM"
fi
python -m xcsb.cli m1 table --run "$RUN" > "$RUN/ladder_table.md" 2>> "$SUM"
echo "job $JOB done $(date -Is)" >> "$SUM"
