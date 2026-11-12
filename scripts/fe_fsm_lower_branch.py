"""Lower-energy branch of the Fe FSM E(m) curves of SCH-SI-A/B:v3.1 (user 2026-11-11): every converged start at each m (the FSM
run's cold grid points, incl. the warm m = 2.8 point of B, and the v4 dense run's cold and warm points at m = 0.6-0.9), the
lowest E per m, with its start, m-weighted w_x per spin and the branch spread. Predicted moment (spline minimum), E''(0) (even fit
over m <= 0.3) and the 0.5 -> 1.0 step on the lower branch. JSON only.
Usage: python scripts/fe_fsm_lower_branch.py <FSM run dir> <dense run dir> [out.json [more run dirs ...]]"""
import glob
import json
import os
import re
import sys

import numpy as np

from xcsb.studies.fsm_curves import curv0, minimum

RYMEV = 13605.693122994


def points(run, f):
    out = {}
    for p in glob.glob(os.path.join(run, f"Fe80-fsm-*__hooked__{f}*", "record.json")):
        d = os.path.basename(os.path.dirname(p))
        mm = re.match(rf"Fe80-fsm-([0-9.]+)(?:-s([0-9.]+))?__hooked__{re.escape(f)}(?:__(warm[0-9.]+))?$", d)
        if not mm:
            continue
        r = json.load(open(p))
        fp = (r.get("post") or {}).get("fsm_parts", {})
        if r.get("status") != "ok" or "E_x_ry" not in fp:
            continue
        m = float(mm.group(1))
        start = (mm.group(3) or ((r.get("warm_start") or {}).get("label") and "warm (FSM run)")
                 or (f"cold s={mm.group(2)}" if mm.group(2) else "cold"))
        mw = fp.get("m_weighted_mean") or {}
        out.setdefault(m, []).append({"start": start, "etot_ry": r["energies_ry"]["etot"], "E_x_ry": fp["E_x_ry"],
                                      "E_c_ry": fp["E_c_ry"], "w_x_up": mw.get("w_x_up"), "w_x_dn": mw.get("w_x_dn"),
                                      "w_c": mw.get("w_c"), "iterations": r.get("iterations"), "source": run})
    return out


def main(fsm, dense, out=None, *more):
    """more: further run directories (e.g. the low-m cold starts) joined into the same curves."""
    res = {}
    for f in ("SCH-SI-A_v3.1", "SCH-SI-B_v3.1"):
        pts = points(fsm, f)
        for extra in (dense,) + tuple(more):
            for m, v in points(extra, f).items():
                pts.setdefault(m, []).extend(v)
        ms = np.array(sorted(pts))
        low = [min(pts[m], key=lambda x: x["etot_ry"]) for m in ms]
        E = np.array([x["etot_ry"] for x in low])
        mi, ei = minimum(ms, E)
        c0 = curv0(ms, E)
        Exc = np.array([x["E_x_ry"] + x["E_c_ry"] for x in low])
        cx = curv0(ms, Exc)
        row = {"m": ms.tolist(), "lower": [dict(x, m=float(m)) for m, x in zip(ms, low)],
               "spread_ry": {f"{m:g}": (max(x["etot_ry"] for x in pts[m]) - min(x["etot_ry"] for x in pts[m])) for m in ms
                             if len(pts[m]) > 1},
               "all_starts": {f"{m:g}": pts[m] for m in ms},
               "predicted_moment": mi, "E_min_ry": ei, "E2_m0_meV_per_muB2": None if c0 is None else c0 * RYMEV,
               "Exc2_m0_meV_per_muB2": None if cx is None else cx * RYMEV,
               "E_minus_E0_meV": ((E - E[0]) * RYMEV).tolist()}
        if 0.5 in ms and 1.0 in ms:
            row["step_0.5_to_1.0_meV"] = float((E[list(ms).index(1.0)] - E[list(ms).index(0.5)]) * RYMEV)
        res[f] = row
    if out:
        json.dump(res, open(out, "w"), indent=1, default=float)
    for f, row in res.items():
        print(f"== {f}: lower-branch predicted moment {row['predicted_moment']:.3f}; E''(0) {row['E2_m0_meV_per_muB2']:.1f} meV/muB^2; "
              f"E_xc''(0) {row['Exc2_m0_meV_per_muB2']:.1f}; step 0.5 -> 1.0 {row.get('step_0.5_to_1.0_meV', float('nan')):.1f} meV")
        for x in row["lower"]:
            sp = row["spread_ry"].get(f"{x['m']:g}")
            print(f"  m {x['m']:.1f} E {x['etot_ry']:.6f} dE {((x['etot_ry'] - row['lower'][0]['etot_ry']) * RYMEV):8.1f} meV "
                  f"start {x['start']:16s} w_x {x['w_x_up']:.3f}/{x['w_x_dn']:.3f}" + (f"  branch spread {sp * RYMEV:.1f} meV" if sp else ""))


if __name__ == "__main__":
    main(*sys.argv[1:])
