"""Table of the 80 Ry Fe magnetization analysis from a local copy of the run directory (JSON only).
Usage: python scripts/fe_mag80_table.py <run dir>"""
import glob
import json
import os
import sys


def main(run):
    rows = []
    for p in sorted(glob.glob(os.path.join(run, "Fe80__hooked__*", "record.json"))):
        r = json.load(open(p))
        name = os.path.basename(os.path.dirname(p)).replace("Fe80__hooked__", "")
        h = r.get("history", [])
        last = h[-50:]
        row = {"run": name, "status": r.get("status"), "iterations": len(h),
               "carrier_it": (r.get("carrier") or {}).get("iterations")}
        if r.get("status") == "ok":
            row.update(m=abs(r["magnetization"]["total"]), m_abs=r["magnetization"]["absolute"], E=r["energies_ry"]["etot"])
        elif last:
            ms = [abs(x["total"]) for x in last]
            ab = [x["absolute"] for x in last]
            row.update(m_band=(min(ms), max(ms)), m_abs_band=(min(ab), max(ab)), E_last=last[-1]["etot_ry"],
                       dr2_band=(min(x["dr2"] for x in last), max(x["dr2"] for x in last)), error=(r.get("error") or "")[:80])
        rows.append(row)
    for row in rows:
        print(json.dumps(row, default=float))
    for s in ("SCH-SI-B_v3.1", "SCH-SI-A_v3.1"):
        p = os.path.join(run, f"Fe80__hooked__{s}", "record.json")
        if os.path.exists(p):
            post = json.load(open(p)).get("post", {})
            print(f"== {s} split:", json.dumps(post.get("mag_split", {}), default=float)[:20000])


if __name__ == "__main__":
    main(sys.argv[1])
