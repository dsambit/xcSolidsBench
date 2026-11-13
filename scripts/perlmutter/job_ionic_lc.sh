#!/bin/bash
# Relaxed lattice constants of rock-salt LiF, LiCl, NaF, NaCl, MgO (xcsb.studies.ionic_lc; DESIGN.md Sec. 6.12): inside ONE
# 1-node allocation, from the clone this script lives in.
#   bash scripts/perlmutter/job_ionic_lc.sh <run dir> <stages> <xchole pin (40 hex)>
# stages: comma-separated from R (PBE, r2SCAN base scans, -3 ... +3 %), RX (their extension points), M (SCH-SI-A:v3.2,
# SCH-SI-B:v3.2 base scans), MX (their extension points); e.g. R,RX or M,MX or R,RX,M,MX.
# The xchole clone $W/src/xchole-v3.2 must be at the given pin, which must be the adapter's v3.2 pin (else STOP).
# Phase 0 (once per run dir, a few minutes): the four functionals load; PBE and SCH-SI-A:v3.2 converge the plumbing case
# LC-phase0 (MgO, 40 Ry, 6^3 k, dual 8, 8 pools) with a negative total energy; a failure STOPS the job (no campaign).
# Then the campaign runner in quarter-node slots (8 ranks, 8 k-point pools, 4 threads) until done or the deadline (finished
# points are skipped: relaunching continues), then the table (lc_table.md, lc_table.json).
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
source "$REPO/scripts/perlmutter/env_xcsb.sh"
RUN=${1:?run dir}; STAGES=${2:?stages, e.g. R,RX}; PIN=${3:?xchole pin}
export XCSB_XCHOLE_SRC=$W/src/xchole-v3.2/src XCSB_XCHOLE_LINE=v3.2
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"     # this clone's xcsb, not the editable install of the main clone
mkdir -p "$RUN"
cd "$REPO"
JOB=${SLURM_JOB_ID:-nojob}
DEADLINE=$(( $(date +%s) + ${XCSB_ALLOC_S:-14400} - 420 ))
SUM="$RUN/summary.txt"
echo "job $JOB account ${SLURM_JOB_ACCOUNT:-unknown} stages $STAGES start $(date -Is) deadline $(date -d @$DEADLINE -Is)" >> "$SUM"
echo "job $JOB account ${SLURM_JOB_ACCOUNT:-unknown} stages $STAGES $(date -Is)" >> "$RUN/launch_tries.txt"
XHEAD=$(git -C "$(dirname "$XCSB_XCHOLE_SRC")" rev-parse HEAD 2>/dev/null)
echo "$JOB $(git rev-parse HEAD) xchole $XHEAD line v3.2 stages $STAGES $(date -Is)" >> "$RUN/COMMITS"
[ "$XHEAD" = "$PIN" ] || { echo "STOP: xchole clone $(dirname "$XCSB_XCHOLE_SRC") is at '$XHEAD', not the pin $PIN" >> "$SUM"; exit 0; }
python -c "import os, xcsb, sys; f = os.path.abspath(xcsb.__file__); sys.exit(0 if f.startswith('$REPO/') else 1)" \
  || { echo "STOP: xcsb is not imported from $REPO" >> "$SUM"; exit 0; }
python -c "import sys; from xcsb.functional import xchole as X; sys.exit(0 if X.XCHOLE_PIN == '$PIN' else 1)" \
  || { echo "STOP: the pin $PIN is not the adapter's v3.2 pin of this xcsb commit" >> "$SUM"; exit 0; }
export OMP_NUM_THREADS=4 OMP_PLACES=cores OMP_PROC_BIND=close

P0="$RUN/phase0"
if [ ! -f "$P0/PASS" ]; then
  mkdir -p "$P0"
  timeout 900 srun -N 1 -n 1 -c 8 --exact --cpu-bind=cores python -c "
from xcsb.functional import get
from xcsb.studies import ionic_lc as L
for n in L.FUNCTIONALS:
    print(get(n).provenance)
" > "$P0/functionals.log" 2>&1
  rc=$?; echo "phase0 functionals rc=$rc" >> "$SUM"
  [ $rc -eq 0 ] || { echo "STOP: phase 0 (the functionals do not load at the pin)" >> "$SUM"; exit 0; }
  for F in PBE SCH-SI-A:v3.2; do
    timeout 1500 srun -N 1 -n 8 -c 8 --exact --cpu-bind=cores python -m xcsb.cli scf --case LC-phase0 --mode hooked --name "$F" \
        --root "$P0" --npool 8 > "$P0/$F.log" 2>&1 &
  done
  wait
  python - "$P0" >> "$SUM" 2>&1 <<'PYCHK'
import json, os, sys
from xcsb.tiers import run_dir
bad = 0
for f in ("PBE", "SCH-SI-A:v3.2"):
    p = os.path.join(run_dir(sys.argv[1], "LC-phase0", "hooked", f), "record.json")
    r = json.load(open(p)) if os.path.exists(p) else {"status": "missing"}
    ok = r.get("status") == "ok" and r["energies_ry"]["etot"] < 0
    bad += not ok
    print(f"phase0 {f}: {r.get('status')} {r.get('error') or ''}", *( [f"etot {r['energies_ry']['etot']:.8f} Ry",
          f"it {r['carrier']['iterations']} + {r['iterations']}", f"{r['seconds']:.0f} s"] if r.get("status") == "ok" else []))
sys.exit(1 if bad else 0)
PYCHK
  if [ $? -eq 0 ]; then touch "$P0/PASS"; else echo "STOP: phase 0 failed" >> "$SUM"; exit 0; fi
fi

python -m xcsb.campaign --study ionic_lc --run "$RUN" --deadline "$DEADLINE" --slots 4 --stages "$STAGES" >> "$RUN/campaign_$JOB.out" 2>&1
echo "campaign rc=$? $(date -Is)" >> "$SUM"
python -m xcsb.cli lc table --run "$RUN" > "$RUN/lc_table.md" 2>> "$SUM"
tail -1 "$RUN/lc_table.md" >> "$SUM"
echo "job $JOB done $(date -Is)" >> "$SUM"
