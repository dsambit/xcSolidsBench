#!/bin/bash
# Diagnostic of the ionic lattice-constant stall (xcsb.studies.ionic_lc, DIAG_RUNS; DESIGN.md Sec. 6.12): inside ONE 1-node
# allocation of 30 min, from the clone this script lives in. LiF at a_ref, protocol cutoffs (96 / 768 Ry) and bands, PBE:
#   phase A (<= 8 min, 4 runs side by side): 12^3 k (72 irreducible): old start (native TPSS carrier) | new start (XC-free step
#     + PBE scaffold) | new start with SCH-SI-A:v3.2 (hook cost) | new start in the earlier layout (8 ranks x 4 threads);
#   phase B (the rest, 3 runs side by side): the full 28^3 mesh (624 irreducible): old start | new start (32 ranks x 1 thread,
#     32 pools: the production slot) | new start on 64 ranks / 64 pools (pool scaling).
# Every run is killed at its phase's limit; finished runs are skipped on a relaunch. The summary (diag_summary.md / .json:
# per-iteration wall times, QE's cpu times and c_bands warnings from pw.out, seconds per k point and iteration, energies of the
# two starts and layouts) is written after each phase; COST.json (the measured constants of the scheduling estimates) when the
# full-mesh run with the new start has per-iteration times.
#   bash scripts/perlmutter/job_ionic_lc_diag.sh <run dir> <xchole pin (40 hex)>
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
source "$REPO/scripts/perlmutter/env_xcsb.sh"
RUN=${1:?run dir}; PIN=${2:?xchole pin}
export XCSB_XCHOLE_SRC=$W/src/xchole-v3.2/src XCSB_XCHOLE_LINE=v3.2
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN/logs"
cd "$REPO"
JOB=${SLURM_JOB_ID:-nojob}
END=$(( $(date +%s) + ${XCSB_ALLOC_S:-1800} - 150 ))
SUM="$RUN/summary.txt"
echo "job $JOB account ${SLURM_JOB_ACCOUNT:-unknown} diag start $(date -Is) end of runs $(date -d @$END -Is)" >> "$SUM"
echo "job $JOB account ${SLURM_JOB_ACCOUNT:-unknown} diag $(date -Is)" >> "$RUN/launch_tries.txt"
XHEAD=$(git -C "$(dirname "$XCSB_XCHOLE_SRC")" rev-parse HEAD 2>/dev/null)
echo "$JOB $(git rev-parse HEAD) xchole $XHEAD line v3.2 diag $(date -Is)" >> "$RUN/COMMITS"
[ "$XHEAD" = "$PIN" ] || { echo "STOP: xchole clone $(dirname "$XCSB_XCHOLE_SRC") is at '$XHEAD', not the pin $PIN" >> "$SUM"; exit 0; }
python -c "import os, xcsb, sys; f = os.path.abspath(xcsb.__file__); sys.exit(0 if f.startswith('$REPO/') else 1)" \
  || { echo "STOP: xcsb is not imported from $REPO" >> "$SUM"; exit 0; }
python -c "import sys; from xcsb.functional import xchole as X; sys.exit(0 if X.XCHOLE_PIN == '$PIN' else 1)" \
  || { echo "STOP: the pin $PIN is not the adapter's v3.2 pin of this xcsb commit" >> "$SUM"; exit 0; }
export OMP_PLACES=cores OMP_PROC_BIND=close

phase() {    # phase <A|B> <limit s>: the runs of the phase side by side (not yet converged ones), each killed at the limit
  local ph=$1 lim=$2
  [ "$lim" -ge 60 ] || { echo "phase $ph: only $lim s left, skipped" >> "$SUM"; return; }
  echo "phase $ph start $(date -Is) limit $lim s" >> "$SUM"
  while read -r name ranks cpus threads args; do
    OMP_NUM_THREADS=$threads timeout --signal=TERM --kill-after=30 "$lim" srun -N 1 -n "$ranks" -c "$cpus" --exact --cpu-bind=cores \
        python -m xcsb.cli $args < /dev/null > "$RUN/logs/$name.log" 2>&1 &
    echo "  $name: $ranks ranks x $threads threads" >> "$SUM"
  done < <(python - "$RUN" "$ph" <<'PYLIST'
import json, os, sys
from xcsb.studies import ionic_lc as L
run, ph = sys.argv[1:3]
for n, spec in L.DIAG_RUNS.items():
    p = os.path.join(L.diag_dir(run, n), "record.json")
    if spec[5] == ph and not (os.path.exists(p) and json.load(open(p)).get("status") == "ok"):
        ranks, cpus, threads, args = L.diag_command(run, n)
        print(n, ranks, cpus, threads, " ".join(args))
PYLIST
)
  wait
  echo "phase $ph end $(date -Is)" >> "$SUM"
  python -m xcsb.cli lc diag-summary --run "$RUN" > "$RUN/diag_summary.md" 2>> "$SUM"
}

phase A 480
phase B $(( END - $(date +%s) ))
grep -E "^Verdict|^Measured|^Full study" "$RUN/diag_summary.md" >> "$SUM"
echo "job $JOB done $(date -Is)" >> "$SUM"
