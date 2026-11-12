#!/bin/bash
# Launch ONE 1-node interactive allocation (m2360) for a v4 Fe item, from its own clone of the bare repo at the current main;
# self-terminating. Login node:
#   nohup bash <any clone>/scripts/perlmutter/launch_fe_v4.sh dense > $W/runs/xcsb/v4_2026-11-12_fe_fsm_dense/launch_1.log 2>&1 &
#   nohup bash <any clone>/scripts/perlmutter/launch_fe_v4.sh sb    > $W/runs/xcsb/v4_2026-11-12_fe_sb/launch_1.log 2>&1 &
# dense: item 1 (2 h: the dense scan + hysteresis, then item 2 if >= 50 min remain); sb: item 2 alone (1 h).
# Refuses to start while the user has >= 2 jobs queued or running (all workstreams); salloc retried (12 x 5 min) only when an
# interactive request is revoked while pending.
W=/pscratch/sd/d/dsambit/xc_functional_dev
WHAT=${1:?dense or sb}
DENSE=$W/runs/xcsb/v4_2026-11-12_fe_fsm_dense
SB=$W/runs/xcsb/v4_2026-11-12_fe_sb
FSM=$W/runs/xcsb/2026-11-11_fe_fsm
case $WHAT in
  dense) RUN=$DENSE; T=02:00:00; CMD="bash scripts/perlmutter/job_fe_fsm_dense.sh $DENSE $SB";;
  sb)    RUN=$SB; T=01:00:00; CMD="bash scripts/perlmutter/job_fe_sb.sh $SB $FSM $DENSE";;
  *) echo "dense or sb" >&2; exit 2;;
esac
mkdir -p "$RUN"
[ -d $W/src/xchole-dd8b5a51/src ] || { echo "missing the v4 xchole clone: run scripts/perlmutter/fetch_sources.sh"; exit 1; }
C=$(git -C $W/xcSolidsBench.git rev-parse main)
REPO=$W/src/xcSolidsBench-v4-${C:0:8}
if [ ! -d "$REPO" ]; then git clone -q $W/xcSolidsBench.git "$REPO" && git -C "$REPO" checkout -q "$C"; fi
echo "launch $WHAT $(date -Is) commit $C clone $REPO" >> "$RUN/launch_tries.txt"
for try in $(seq 1 12); do
  n=$(squeue -u dsambit -h | wc -l)
  if [ "$n" -ge 2 ]; then
    echo "try $try: $n jobs of the user queued or running (limit 2): not launched $(date -Is)" >> "$RUN/launch_tries.txt"
    exit 1
  fi
  out="$RUN/job_$(date +%Y%m%dT%H%M%S)_try$try.out"
  XCSB_ALLOC_S=$([ $WHAT = dense ] && echo 7200 || echo 3600) salloc -N 1 -C cpu -q interactive -t $T -A m2360 \
    --job-name xcsb_v4$WHAT bash -c "cd $REPO && $CMD" > "$out" 2>&1
  echo "try $try exit $? $(date -Is) ($out)" >> "$RUN/launch_tries.txt"
  grep -q "Unable to allocate resources" "$out" || break
  sleep 300
done
