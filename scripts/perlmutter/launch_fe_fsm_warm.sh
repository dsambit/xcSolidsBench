#!/bin/bash
# Launch ONE 1-node interactive allocation (1 h, m2360) for the warm-start rerun of Fe80-fsm-2.8, SCH-SI-B:v3.1, from the
# converged m = 2.6 point; self-terminating. Login node:
#   nohup bash <any clone>/scripts/perlmutter/launch_fe_fsm_warm.sh > $W/runs/xcsb/2026-11-11_fe_fsm/launch_warm_2.8.log 2>&1 &
# The job runs from its own clone of the bare repo at the current main ($W/src/xcSolidsBench-fsmwarm-<commit>); the FSM run
# directory is shared with job 61150345 only through other points' directories. Refuses to start while the user has >= 2 jobs
# queued or running (all workstreams); salloc retried (12 x 5 min) only when an interactive request is revoked while pending.
W=/pscratch/sd/d/dsambit/xc_functional_dev
RUN=$W/runs/xcsb/2026-11-11_fe_fsm
C=$(git -C $W/xcSolidsBench.git rev-parse main)
REPO=$W/src/xcSolidsBench-fsmwarm-${C:0:8}
if [ ! -d "$REPO" ]; then git clone -q $W/xcSolidsBench.git "$REPO" && git -C "$REPO" checkout -q "$C"; fi
SRC=$RUN/Fe80-fsm-2.6__hooked__SCH-SI-B_v3.1
grep -qs '"status": "ok"' $SRC/record.json && [ -d $SRC/out/xcsb.save ] || { echo "source $SRC not converged or no outdir"; exit 1; }
echo "launch warm 2.8 $(date -Is) commit $C clone $REPO" >> "$RUN/launch_tries_warm.txt"
for try in $(seq 1 12); do
  n=$(squeue -u dsambit -h | wc -l)
  if [ "$n" -ge 2 ]; then
    echo "try $try: $n jobs of the user queued or running (limit 2): not launched $(date -Is)" >> "$RUN/launch_tries_warm.txt"
    exit 1
  fi
  out="$RUN/job_warm_$(date +%Y%m%dT%H%M%S)_try$try.out"
  salloc -N 1 -C cpu -q interactive -t 01:00:00 -A m2360 --job-name xcsb_fsmwarm \
    bash -c "cd $REPO && bash scripts/perlmutter/job_fe_fsm_warm.sh $RUN Fe80-fsm-2.8 SCH-SI-B:v3.1 Fe80-fsm-2.6 'warm start from m = 2.6'" > "$out" 2>&1
  echo "try $try exit $? $(date -Is) ($out)" >> "$RUN/launch_tries_warm.txt"
  grep -q "Unable to allocate resources" "$out" || break
  sleep 300
done
