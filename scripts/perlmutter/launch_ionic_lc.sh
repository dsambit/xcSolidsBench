#!/bin/bash
# Launch ONE 1-node interactive allocation (account m2360) of the ionic-solid lattice constants (job_ionic_lc.sh, <= 4 h) or of
# their start / layout diagnostic (job_ionic_lc_diag.sh, 30 min); self-terminating. Login node:
#   mkdir -p $W/runs/xcsb/2026-11-13_ionic_lc_diag $W/runs/xcsb/2026-11-13_ionic_lc_scaffold
#   nohup bash <any clone>/scripts/perlmutter/launch_ionic_lc.sh diag  > $W/runs/xcsb/2026-11-13_ionic_lc_diag/launch_1.log 2>&1 &
#   nohup bash <any clone>/scripts/perlmutter/launch_ionic_lc.sh ref   > $W/runs/xcsb/2026-11-13_ionic_lc_scaffold/launch_ref_1.log 2>&1 &
#   nohup bash <any clone>/scripts/perlmutter/launch_ionic_lc.sh model > $W/runs/xcsb/2026-11-13_ionic_lc_scaffold/launch_model_1.log 2>&1 &
# diag: LiF, old start vs new start, 12^3 and 28^3 k, layouts (run dir XCSB_LC_DIAG; writes diag_summary.md and COST.json, which
# the later modes copy into their run dir). ref: stages R,RX (PBE, r2SCAN, with their extension points); model: M,MX
# (SCH-SI-A:v3.2, SCH-SI-B:v3.2); all: R,RX,M,MX. The run dir of the stalled first attempt (2026-11-13_ionic_lc, native TPSS
# carrier) is not reused.
# Relaunching the same mode continues the run (finished points are skipped; two allocations may share the run dir).
# The job runs from its OWN clone of the bare repo at the current main ($W/src/xcSolidsBench-lc-<commit>); the xchole clone
# $W/src/xchole-v3.2 (a clone of $W/xcHoleModelMeta.git) is checked out at the pin: the optional second argument, which must be
# the adapter's v3.2 pin of that commit (default: that pin), and is passed to the job explicitly.
# Refuses to submit while 2 or more of the user's interactive jobs are queued or running (all workstreams). salloc is retried
# (12 times, 5 min apart) only when an interactive request is revoked while pending.
W=/pscratch/sd/d/dsambit/xc_functional_dev
ACCOUNT=m2360
WHAT=${1:-ref}
DIAG=${XCSB_LC_DIAG:-$W/runs/xcsb/2026-11-13_ionic_lc_diag}
RUN=${XCSB_LC_RUN:-$W/runs/xcsb/2026-11-13_ionic_lc_scaffold}
T=04:00:00; AS=14400
case $WHAT in
  diag)  STAGES=diag; RUN=$DIAG; T=00:30:00; AS=1800;;
  mix)   STAGES=mix; RUN=${XCSB_LC_MIX:-$W/runs/xcsb/2026-11-13_ionic_lc_mix}; T=00:45:00; AS=2700;;
  resp)  STAGES=resp; RUN=${XCSB_LC_RESP:-$W/runs/xcsb/2026-11-14_lif_response}; T=02:00:00; AS=7200;;
  ref)   STAGES=R,RX;;
  model) STAGES=M,MX;;
  all)   STAGES=R,RX,M,MX;;
  *) echo "diag, mix, resp, ref, model or all" >&2; exit 2;;
esac
mkdir -p "$RUN"
LOG="$RUN/launch_tries.txt"
C=$(git -C $W/xcSolidsBench.git rev-parse main)
REPO=$W/src/xcSolidsBench-lc-${C:0:8}
if [ ! -d "$REPO" ]; then git clone -q $W/xcSolidsBench.git "$REPO" && git -C "$REPO" checkout -q "$C"; fi
[ -f "$REPO/scripts/perlmutter/job_lif_response.sh" ] || { echo "no job_ionic_lc_diag.sh in $REPO (commit $C): push main to the perlmutter remote" | tee -a "$LOG"; exit 1; }
APIN=$(cd "$REPO" && python3 -c "import re; s = open('src/xcsb/functional/xchole.py').read(); m = re.search(r'\"v3.2\": \"([0-9a-f]{40})\"', s); print(m.group(1) if m else '')")
PIN=${2:-$APIN}
[ -n "$PIN" ] && [ "$PIN" = "$APIN" ] || { echo "xchole pin '$PIN' is not the adapter's v3.2 pin '$APIN' of $C" | tee -a "$LOG"; exit 1; }
X=$W/src/xchole-v3.2
if [ ! -d "$X" ]; then git clone -q $W/xcHoleModelMeta.git "$X"; fi
git -C "$X" fetch -q origin 2>/dev/null
git -C "$X" checkout -q "$PIN" || { echo "cannot check out $PIN in $X (is strongxc-v3 pushed to $W/xcHoleModelMeta.git?)" | tee -a "$LOG"; exit 1; }
[ "$(git -C "$X" rev-parse HEAD)" = "$PIN" ] && [ -f "$X/src/xchole/sch_si/parameter_sets/SCH-SI-A-v3.2.json" ] \
  || { echo "$X is not at $PIN with the v3.2 records" | tee -a "$LOG"; exit 1; }
echo "launch $(date -Is) mode $WHAT stages $STAGES commit $C clone $REPO xchole $PIN account $ACCOUNT" >> "$LOG"
for try in $(seq 1 12); do
  n=$(squeue -u dsambit -h -o "%q" | grep -c interactive)
  if [ "$n" -ge 2 ]; then
    echo "try $try: $n interactive jobs of the user queued or running (limit 2): not launched $(date -Is)" | tee -a "$LOG"
    exit 1
  fi
  out="$RUN/job_$(date +%Y%m%dT%H%M%S)_${WHAT}_try$try.out"
  if [ $WHAT = diag ]; then CMD="bash scripts/perlmutter/job_ionic_lc_diag.sh $RUN $PIN"
  elif [ $WHAT = mix ]; then CMD="bash scripts/perlmutter/job_ionic_lc_mix.sh $RUN $PIN"
  elif [ $WHAT = resp ]; then CMD="bash scripts/perlmutter/job_lif_response.sh $RUN $PIN"
  else CMD="bash scripts/perlmutter/job_ionic_lc.sh $RUN $STAGES $PIN $DIAG"; fi
  XCSB_ALLOC_S=$AS salloc -N 1 -C cpu -q interactive -t $T -A $ACCOUNT --job-name xcsb_lc_$WHAT \
    bash -c "cd $REPO && $CMD" > "$out" 2>&1
  rc=$?
  jid=$(grep -o "job allocation [0-9]*" "$out" | head -1 | grep -o "[0-9]*")
  echo "try $try exit $rc job ${jid:-none} account $ACCOUNT $(date -Is) ($out)" >> "$LOG"
  grep -q "Unable to allocate resources" "$out" || break
  sleep 300
done
