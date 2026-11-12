#!/bin/bash
# MgO band gap (xcsb.studies.mgo_gap; DESIGN.md Sec. 6.8): inside ONE 1-node allocation, from the clone this script lives in.
#   bash scripts/perlmutter/job_mgo_gap.sh <run dir>
# Phase 0 (~2 min): PBE MgO 40 Ry 6^3 k; its record must give the gaps (eigenvalues, k list, Gamma/X/L found), else STOP.
# Then the campaign runner: stage P (PBE scan 40-140 Ry, 8^3 k -> EC.json), stage G (4 functionals x 3 cutoffs x 3 k meshes),
# quarter-node slots (8 ranks, one QE pool), until done or the deadline; then the tables.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
source "$REPO/scripts/perlmutter/env_xcsb.sh"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"     # this clone's xcsb, not the editable install of the main clone
RUN=${1:?run dir}
mkdir -p "$RUN"
cd "$REPO"
JOB=${SLURM_JOB_ID:-nojob}
DEADLINE=$(( $(date +%s) + ${XCSB_ALLOC_S:-14400} - 420 ))
SUM="$RUN/summary.txt"
echo "$JOB $(git rev-parse HEAD) $(date -Is)" >> "$RUN/COMMITS"
git -C "$(dirname "$XCSB_XCHOLE_SRC")" rev-parse HEAD > "$RUN/XCHOLE_COMMIT"
python -c "import os, xcsb, sys; f = os.path.abspath(xcsb.__file__); sys.exit(0 if f.startswith('$REPO/') else 1)" \
  || { echo "STOP: xcsb is not imported from $REPO" >> "$SUM"; exit 0; }
export OMP_NUM_THREADS=4 OMP_PLACES=cores OMP_PROC_BIND=close
echo "job $JOB start $(date -Is) deadline $(date -d @$DEADLINE -Is)" >> "$SUM"

P0="$RUN/phase0"
if [ ! -f "$P0/PASS" ]; then
  mkdir -p "$P0"
  timeout 1200 srun -N 1 -n 8 -c 8 --exact --cpu-bind=cores python -m xcsb.cli scf --case MgO-40-k6 --mode hooked --name PBE \
      --root "$P0" --npool 1 > "$P0/PBE.log" 2>&1
  echo "phase0 PBE MgO-40-k6 rc=$?" >> "$SUM"
  python - "$P0" >> "$SUM" 2>&1 <<'PYCHK'
import json, os, sys
from xcsb.studies import mgo_gap as G
p = os.path.join(sys.argv[1], "MgO-40-k6__hooked__PBE", "record.json")
r = json.load(open(p)) if os.path.exists(p) else {"status": "missing"}
if r.get("status") != "ok":
    print(f"phase0 MgO: record {r.get('status')} {r.get('error')}"); sys.exit(1)
g = G.gaps(r, G._lattice())
print("phase0 MgO gaps (eV):", json.dumps({k: v for k, v in g.items() if k.endswith("_ev")}))
sys.exit(0 if g["min_gap_ev"] > 0 else 1)
PYCHK
  if [ $? -eq 0 ]; then touch "$P0/PASS"; else echo "STOP: phase 0 failed" >> "$SUM"; exit 0; fi
fi

python -m xcsb.campaign --study mgo_gap --run "$RUN" --deadline "$DEADLINE" --slots 4 --stages P,G >> "$RUN/campaign_$JOB.out" 2>&1
echo "campaign rc=$? $(date -Is)" >> "$SUM"
python -m xcsb.cli mgo table --run "$RUN" > "$RUN/gap_table.md" 2>> "$SUM"
echo "job $JOB done $(date -Is)" >> "$SUM"
