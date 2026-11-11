#!/bin/bash
# Launch ONE 1-node interactive allocation (<= 4 h, m2360) of the M1 cutoff ladder (job_m1_ladder.sh); self-terminating.
# Login node, from the node clone after updating it:
#   cd $W/src/xcSolidsBench && git fetch -q origin && git reset -q --hard origin/main && \
#     nohup bash scripts/perlmutter/launch_m1_ladder.sh > $W/runs/xcsb/2026-11-18_m1_ladder/launch_<n>.log 2>&1 &
# Refuses to start while 2 or more jobs of the user are queued or running (all workstreams). salloc is retried (12 times,
# 5 min apart) only when the interactive request is revoked while pending ("Unable to allocate resources", 2026-11-11).
W=/pscratch/sd/d/dsambit/xc_functional_dev
REPO=$(cd "$(dirname "$0")/../.." && pwd)
RUN=${XCSB_M1_RUN:-$W/runs/xcsb/2026-11-18_m1_ladder}
mkdir -p "$RUN"
echo "launch $(date -Is) commit $(git -C "$REPO" rev-parse HEAD)" >> "$RUN/launch_tries.txt"
for try in $(seq 1 12); do
  n=$(squeue -u dsambit -h | wc -l)
  if [ "$n" -ge 2 ]; then
    echo "try $try: $n jobs of the user queued or running (limit 2): not launched $(date -Is)" >> "$RUN/launch_tries.txt"
    exit 1
  fi
  salloc -N 1 -C cpu -q interactive -t 04:00:00 -A m2360 --job-name xcsb_m1lad \
    bash -c "cd $REPO && bash scripts/perlmutter/job_m1_ladder.sh $RUN" > "$RUN/job_$(date +%Y%m%dT%H%M%S)_try$try.out" 2>&1
  rc=$?
  last=$(ls -t "$RUN"/job_*_try$try.out | head -1)
  echo "try $try exit $rc $(date -Is) ($last)" >> "$RUN/launch_tries.txt"
  grep -q "Unable to allocate resources" "$last" || break
  sleep 300
done
