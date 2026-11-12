#!/bin/bash
# Launch ONE 1-node interactive allocation (<= 4 h, m2360) of the MgO band-gap test; self-terminating. Login node:
#   nohup bash <any clone>/scripts/perlmutter/launch_mgo_gap.sh > $W/runs/xcsb/2026-11-11_mgo_gap/launch_1.log 2>&1 &
# The job runs from its OWN clone of the bare repo at the current main ($W/src/xcSolidsBench-mgo-<commit>), so the main
# clone (editable install, used by running jobs) is never touched. Refuses to start while the user has >= 2 jobs queued or
# running (all workstreams). salloc is retried (12 times, 5 min apart) only when an interactive request is revoked while pending.
W=/pscratch/sd/d/dsambit/xc_functional_dev
RUN=${XCSB_MGO_RUN:-$W/runs/xcsb/2026-11-11_mgo_gap}
mkdir -p "$RUN"
C=$(git -C $W/xcSolidsBench.git rev-parse main)
CL=$W/src/xcSolidsBench-mgo-${C:0:8}
if [ ! -d "$CL" ]; then git clone -q $W/xcSolidsBench.git "$CL" && git -C "$CL" checkout -q "$C"; fi
echo "launch $(date -Is) commit $C clone $CL" >> "$RUN/launch_tries.txt"
for try in $(seq 1 12); do
  n=$(squeue -u dsambit -h | wc -l)
  if [ "$n" -ge 2 ]; then
    echo "try $try: $n jobs of the user queued or running (limit 2): not launched $(date -Is)" >> "$RUN/launch_tries.txt"
    exit 1
  fi
  out="$RUN/job_$(date +%Y%m%dT%H%M%S)_try$try.out"
  salloc -N 1 -C cpu -q interactive -t 04:00:00 -A m2360 --job-name xcsb_mgo \
    bash -c "cd $CL && bash scripts/perlmutter/job_mgo_gap.sh $RUN" > "$out" 2>&1
  echo "try $try exit $? $(date -Is) ($out)" >> "$RUN/launch_tries.txt"
  grep -q "Unable to allocate resources" "$out" || break
  sleep 300
done
