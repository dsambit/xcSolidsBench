#!/bin/bash
# Launch ONE 1-node interactive allocation (<= 4 h, m2360) of the M1 cutoff ladder (job_m1_ladder.sh) for ONE stage (P or L);
# self-terminating. Login node:
#   nohup bash <any clone>/scripts/perlmutter/launch_m1_ladder.sh P > $W/runs/xcsb/2026-11-18_m1_ladder/launch_P2.log 2>&1 &
# The job runs from its OWN clone of the bare repo at the current main ($W/src/xcSolidsBench-m1-<commit>), so the main clone
# (editable install, used by other running jobs) is never touched. Stage L only after the user's batch-vs-interactive decision
# on the stage-P report. Refuses to start while the user has >= 2 jobs queued or running (all workstreams). salloc is retried
# (12 times, 5 min apart) only when an interactive request is revoked while pending ("Unable to allocate resources").
W=/pscratch/sd/d/dsambit/xc_functional_dev
STAGE=${1:?stage P or L}
case $STAGE in P|L) ;; *) echo "stage must be P or L" >&2; exit 2;; esac
RUN=${XCSB_M1_RUN:-$W/runs/xcsb/2026-11-18_m1_ladder}
mkdir -p "$RUN"
C=$(git -C $W/xcSolidsBench.git rev-parse main)
REPO=$W/src/xcSolidsBench-m1-${C:0:8}
if [ ! -d "$REPO" ]; then git clone -q $W/xcSolidsBench.git "$REPO" && git -C "$REPO" checkout -q "$C"; fi
echo "launch stage $STAGE $(date -Is) commit $C clone $REPO" >> "$RUN/launch_tries.txt"
for try in $(seq 1 12); do
  n=$(squeue -u dsambit -h | wc -l)
  if [ "$n" -ge 2 ]; then
    echo "try $try: $n jobs of the user queued or running (limit 2): not launched $(date -Is)" >> "$RUN/launch_tries.txt"
    exit 1
  fi
  out="$RUN/job_$(date +%Y%m%dT%H%M%S)_try$try.out"
  salloc -N 1 -C cpu -q interactive -t 04:00:00 -A m2360 --job-name xcsb_m1$STAGE \
    bash -c "cd $REPO && bash scripts/perlmutter/job_m1_ladder.sh $RUN $STAGE" > "$out" 2>&1
  echo "try $try exit $? $(date -Is) ($out)" >> "$RUN/launch_tries.txt"
  grep -q "Unable to allocate resources" "$out" || break
  sleep 300
done
