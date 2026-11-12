#!/bin/bash
# Launch ONE 1-node interactive allocation (m2360) of the self-consistent spin-balanced r2SCAN test ('sb', 2 h) or of the E1
# omega fit to our PBE Fe moment ('fit', 1.5 h; after 'sb' and the frozen job); self-terminating.
# The xchole clone $W/src/xchole-<line> is checked out at the pin XCHOLE_PINS[<line>] of this commit (sb: v4-sb b2501727c;
# fit: v4-e1 76cfa7c20).
# Login node:  nohup bash <any clone>/scripts/perlmutter/launch_r2scan_sb.sh sb  > $W/runs/xcsb/v4_2026-11-12_r2scan_sb/launch_1.log 2>&1 &
#              nohup bash <any clone>/scripts/perlmutter/launch_r2scan_sb.sh fit > $W/runs/xcsb/v4_2026-11-12_fe_omega_fit/launch_1.log 2>&1 &
# Refuses to start while the user has >= 2 jobs queued or running (all workstreams); salloc retried (12 x 5 min) only when an
# interactive request is revoked while pending.
W=/pscratch/sd/d/dsambit/xc_functional_dev
WHAT=${1:-sb}       # sb: the r2SCAN-SB SCF test; fit: the E1 omega fit (after sb and the frozen job); tmo-pg / tmo-f: NiO, MnO gaps
                    # (PBE scan + short grid, then the four functionals at the selected settings)
case $WHAT in
  sb)  RUN=$W/runs/xcsb/v4_2026-11-12_r2scan_sb; JOBS=job_r2scan_sb.sh; T=02:00:00; AS=7200; LINE=v4-sb;;
  sb2) RUN=$W/runs/xcsb/v4_2026-11-12_r2scan_sb; JOBS=job_r2scan_sb.sh; ARG2=R; T=01:30:00; AS=5400; LINE=v4-sb;;
  fit) RUN=$W/runs/xcsb/v4_2026-11-12_fe_omega_fit; JOBS=job_fe_omega_fit.sh; T=01:30:00; AS=5400; LINE=v4-e1;;
  tmo-pg) RUN=$W/runs/xcsb/v4_2026-11-12_tmo_gap; JOBS="job_tmo_gap.sh"; ARG2=P,G; T=04:00:00; AS=14400; LINE=v4-sb;;
  tmo-f)  RUN=$W/runs/xcsb/v4_2026-11-12_tmo_gap; JOBS="job_tmo_gap.sh"; ARG2=F; T=04:00:00; AS=14400; LINE=v4-sb;;
  *) echo "sb, sb2, fit, tmo-pg or tmo-f" >&2; exit 2;;
esac
mkdir -p "$RUN"
C=$(git -C $W/xcSolidsBench.git rev-parse main)
REPO=$W/src/xcSolidsBench-sb-${C:0:8}
if [ ! -d "$REPO" ]; then git clone -q $W/xcSolidsBench.git "$REPO" && git -C "$REPO" checkout -q "$C"; fi
PIN=$(cd "$REPO" && python3 -c "import re; s = open('src/xcsb/functional/xchole.py').read(); m = re.search(r'\"$LINE\": \"([0-9a-f]{40})\"', s); print(m.group(1) if m else '')")
[ -n "$PIN" ] || { echo "the $LINE pin is not set in $C" | tee -a "$RUN/launch_tries.txt"; exit 1; }
X=$W/src/xchole-$LINE
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
  XCSB_ALLOC_S=$AS salloc -N 1 -C cpu -q interactive -t $T -A m2360 --job-name xcsb_v4$WHAT \
    bash -c "cd $REPO && bash scripts/perlmutter/$JOBS $RUN ${ARG2:-}" > "$out" 2>&1
  echo "try $try exit $? $(date -Is) ($out)" >> "$RUN/launch_tries.txt"
  grep -q "Unable to allocate resources" "$out" || break
  sleep 300
done
