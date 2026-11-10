"""Collect the tier JSON summaries of a run directory (copied back with JSON only) into one compact table.

Usage: python scripts/collect_tiers.py <local copy of the run dir>
"""
import glob
import json
import os
import sys


def main(run):
    out = {}
    for f in sorted(glob.glob(os.path.join(run, "V*", "V*.json"))):
        out[os.path.basename(f)[:-5]] = json.load(open(f))
    summary = os.path.join(run, "summary.txt")
    if os.path.exists(summary):
        print(open(summary).read())
    for name, r in out.items():
        print(f"== {name}: pass = {r.get('pass')}")
        for k, v in r.get("cases", {}).items():
            keep = {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, dict)) or kk in ("magnetization",)}
            print(f"   {k}: {json.dumps(keep, default=float)}")


if __name__ == "__main__":
    main(sys.argv[1])
