#!/bin/bash
# Launch ONE 1-node interactive allocation (2 h, m2360) of the self-consistent spin-balanced r2SCAN test; self-terminating.
# Needs the xchole clone $W/src/xchole-v4sb at the spin-balance agent's commit (the pin XCHOLE_PINS['v4-sb'] of this commit).
# Login node:  nohup bash <any clone>/scripts/perlmutter/launch_r2scan_sb.sh > $W/runs/xcsb/v4_2026-11-12_r2scan_sb/launch_1.log 2>&1 &
# Refuses to start while the user has >= 2 jobs queued or running (all workstreams); salloc retried (12 x 5 min) only when an
# interactive request is revoked while pending.
W=/pscratch/sd/d/dsambit/xc_functional_dev
RUN=$W/runs/xcsb/v4_2026-11-12_r2scan_sb
mkdir -p "$RUN"
C=$(git -C $W/xcSolidsBench.git rev-parse main)
REPO=$W/src/xcSolidsBench-sb-${C:0:8}
if [ ! -d "$REPO" ]; then git clone -q $W/xcSolidsBench.git "$REPO" && git -C "$REPO" checkout -q "$C"; fi
PIN=$(cd "$REPO" && python3 -c "import re; s = open('src/xcsb/functional/xchole.py').read(); m = re.search(r'\"v4-sb\": \"([0-9a-f]{40})\"', s); print(m.group(1) if m else '')")
[ -n "$PIN" ] || { echo "the v4-sb pin is not set in $C (waiting for the spin-balance functional)" | tee -a "$RUN/launch_tries.txt"; exit 1; }
X=$W/src/xchole-v4sb
if [ ! -d "$X" ]; then git clone -q $W/xcHoleModelMeta.git "$X"; fi
git -C "$X" fetch -q origin 2>/dev/null; git -C "$X" checkout -q "$PIN" || { echo "cannot check out $PIN in $X"; exit 1; }
[ -f "$X/explore/spin_balance/sb_r2scan.py" ] || { echo "no sb_r2scan.py at $PIN"; exit 1; }
echo "launch $(date -Is) commit $C clone $REPO xchole $PIN" >> "$RUN/launch_tries.txt"
for try in $(seq 1 12); do
  n=$(squeue -u dsambit -h | wc -l)
  if [ "$n" -ge 2 ]; then
    echo "try $try: $n jobs of the user queued or running (limit 2): not launched $(date -Is)" >> "$RUN/launch_tries.txt"
    exit 1
  fi
  out="$RUN/job_$(date +%Y%m%dT%H%M%S)_try$try.out"
  XCSB_ALLOC_S=7200 salloc -N 1 -C cpu -q interactive -t 02:00:00 -A m2360 --job-name xcsb_r2sb \
    bash -c "cd $REPO && bash scripts/perlmutter/job_r2scan_sb.sh $RUN" > "$out" 2>&1
  echo "try $try exit $? $(date -Is) ($out)" >> "$RUN/launch_tries.txt"
  grep -q "Unable to allocate resources" "$out" || break
  sleep 300
done
