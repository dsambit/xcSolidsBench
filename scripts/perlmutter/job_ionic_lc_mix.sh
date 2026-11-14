#!/bin/bash
# Mixer test of the ionic lattice-constant study (xcsb.studies.ionic_lc, MIX_POINTS; DESIGN.md Sec. 6.12): inside ONE 1-node
# allocation of 45 min, from the clone this script lives in. Coarse mesh (MIX_K^3 k), protocol cutoffs: the seven A/B points of
# the scan 61248332 that ended with a negative mixed density, LiF at a_ref (A-v3.2, r2SCAN), each with the mixer reset between
# the scaffold and the functional (start 'pbe-scaffold') and without it ('pbe-scaffold-stale'). Four runs side by side
# (32 ranks x 1 thread each), every run killed after 600 s; finished runs are skipped on a relaunch. Summary: mix_summary.md.
#   bash scripts/perlmutter/job_ionic_lc_mix.sh <run dir> <xchole pin (40 hex)>
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
source "$REPO/scripts/perlmutter/env_xcsb.sh"
RUN=${1:?run dir}; PIN=${2:?xchole pin}
export XCSB_XCHOLE_SRC=$W/src/xchole-v3.2/src XCSB_XCHOLE_LINE=v3.2
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN/logs"
cd "$REPO"
JOB=${SLURM_JOB_ID:-nojob}
END=$(( $(date +%s) + ${XCSB_ALLOC_S:-2700} - 150 ))
SUM="$RUN/summary.txt"
echo "job $JOB account ${SLURM_JOB_ACCOUNT:-unknown} mix start $(date -Is) end of runs $(date -d @$END -Is)" >> "$SUM"
XHEAD=$(git -C "$(dirname "$XCSB_XCHOLE_SRC")" rev-parse HEAD 2>/dev/null)
echo "$JOB $(git rev-parse HEAD) xchole $XHEAD line v3.2 mix $(date -Is)" >> "$RUN/COMMITS"
[ "$XHEAD" = "$PIN" ] || { echo "STOP: xchole clone $(dirname "$XCSB_XCHOLE_SRC") is at '$XHEAD', not the pin $PIN" >> "$SUM"; exit 0; }
python -c "import os, xcsb, sys; f = os.path.abspath(xcsb.__file__); sys.exit(0 if f.startswith('$REPO/') else 1)" \
  || { echo "STOP: xcsb is not imported from $REPO" >> "$SUM"; exit 0; }
export OMP_PLACES=cores OMP_PROC_BIND=close OMP_NUM_THREADS=1

one() {      # one <log name> <xcsb CLI arguments...>
  local name=$1; shift
  local lim=$(( END - $(date +%s) )); [ "$lim" -gt 600 ] && lim=600
  [ "$lim" -ge 60 ] || { echo "  $name: skipped, $lim s left" >> "$SUM"; return; }
  timeout --signal=TERM --kill-after=30 "$lim" srun -N 1 -n 32 -c 2 --exact --cpu-bind=cores \
      python -m xcsb.cli "$@" < /dev/null > "$RUN/logs/$name.log" 2>&1
  echo "  $name rc=$? $(date -Is)" >> "$SUM"
}
n=0
while read -r name args; do
  one "$name" $args &
  n=$(( n + 1 ))
  if [ $(( n % 4 )) -eq 0 ]; then wait; python -m xcsb.cli lc mix-summary --run "$RUN" > "$RUN/mix_summary.md" 2>> "$SUM"; fi
done < <(python - "$RUN" <<'PYLIST'
import json, os, sys
from xcsb.studies import ionic_lc as L
from xcsb.tiers import run_dir
run = sys.argv[1]
for case, f in L.mix_runs():
    p = os.path.join(run_dir(L.scf_root(run), case, "hooked", f), "record.json")
    if not (os.path.exists(p) and json.load(open(p)).get("status") in ("ok", "failed")):
        print(case + "__" + f.replace(":", "_"), " ".join(L.mix_command(run, case, f)))
PYLIST
)
wait
python -m xcsb.cli lc mix-summary --run "$RUN" > "$RUN/mix_summary.md" 2>> "$SUM"
echo "job $JOB done $(date -Is)" >> "$SUM"
