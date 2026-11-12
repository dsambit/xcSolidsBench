"""Fixed-spin-moment E(m) analysis of bcc Fe from a local copy of the FSM run directory (JSON only).

Per functional: E_tot(m), E_x(m), E_c(m), the rest R = E_tot - E_x - E_c; dE/dm and its x, c and rest parts (central
differences on the m grid); the predicted free moment (minimum of a cubic spline through E(m)); for A and B the frozen pairing
variants E'(m) = E_tot(m) - E_xc(m) + E_xc'(m) on the same densities, with their implied moments.
Usage: python scripts/fe_fsm_analysis.py <run dir> [out.json]
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

RY_TO_MEV = 13605.693122994
FUNCS = ("SCH-SI-B_v3.1", "SCH-SI-A_v3.1", "SCH-SI-B_v3.1#ueg-end", "SCH-SI-A_v3.1#ueg-end", "PBE", "r2SCAN")


def load(run):
    data = {}
    for p in glob.glob(os.path.join(run, "Fe80-fsm-*__hooked__*", "record.json")):
        d = os.path.basename(os.path.dirname(p))
        m = float(re.match(r"Fe80-fsm-([0-9.]+)__", d).group(1))
        f = d.split("__hooked__")[1]
        r = json.load(open(p))
        data.setdefault(f, {})[m] = r
    free = {}
    for p in glob.glob(os.path.join(run, "Fe80__hooked__*", "record.json")):
        r = json.load(open(p))
        if r.get("status") == "ok":
            free[os.path.basename(os.path.dirname(p)).split("__hooked__")[1]] = r
    return data, free


def minimum(ms, es):
    cs = CubicSpline(ms, es)
    i = int(np.argmin(es))
    lo, hi = ms[max(i - 1, 0)], ms[min(i + 1, len(ms) - 1)]
    if lo == hi:
        return float(ms[i]), float(es[i])
    r = minimize_scalar(cs, bounds=(lo, hi), method="bounded")
    return float(r.x), float(r.fun)


def curv0(ms, y, mmax=0.3):
    """E''(0) from the even fit y = c0 + c2 m^2 + c4 m^4 over m <= mmax (Ry per mu_B^2): 2 c2."""
    sel = ms <= mmax + 1e-12
    if np.count_nonzero(sel) < 3:
        return None
    A = np.stack([np.ones(np.count_nonzero(sel)), ms[sel] ** 2, ms[sel] ** 4], axis=1)
    c = np.linalg.lstsq(A, y[sel], rcond=None)[0]
    return float(2.0 * c[1])


def curv_at(ms, y, m0):
    return float(CubicSpline(ms, y)(m0, 2))


def stoner(ms, E, Exc, pieces, mfree):
    """Curvatures at m = 0 (even fit over 0-0.3) and at the free moment (cubic spline), in meV per mu_B^2."""
    out = {"units": "meV / mu_B^2 (second derivatives of the energy per cell = per atom)"}
    for tag, at in (("m0", None), ("free", mfree)):
        f = (lambda y: curv0(ms, y)) if at is None else (lambda y, a=at: curv_at(ms, y, a))
        d = {"E_tot": f(E), "E_xc": f(Exc)}
        if d["E_tot"] is not None and d["E_xc"] is not None:
            d["remainder"] = d["E_tot"] - d["E_xc"]
        d.update({f"xc_{k}": f(np.asarray(v)) for k, v in (pieces or {}).items()})
        out[tag] = {k: (None if v is None else v * RY_TO_MEV) for k, v in d.items()}
    if out["m0"].get("E_xc") is not None:
        out["Stoner_I_meV_per_muB2"] = -out["m0"]["E_xc"]
    return out


def main(run, out=None):
    data, free = load(run)
    res = {}
    for f in FUNCS:
        # the m grid only (points with --fsm-parts); the phase-0 constraint checks (m = 2.7400, 2.7646) are not curve points
        pts = {m: r for m, r in data.get(f, {}).items() if r.get("status") == "ok" and m <= 3.0 + 1e-9
               and "E_x_ry" in (r.get("post") or {}).get("fsm_parts", {})}
        missing = sorted((m, r.get("status")) for m, r in data.get(f, {}).items() if r.get("status") != "ok" and m <= 3.0 + 1e-9)
        if len(pts) < 4:
            res[f] = {"status": f"only {len(pts)} converged points"}
            continue
        ms = np.array(sorted(pts))
        rec = [pts[m] for m in ms]
        E = np.array([r["energies_ry"]["etot"] for r in rec])
        post = [r.get("post", {}).get("fsm_parts", {}) for r in rec]
        Ex = np.array([p.get("E_x_ry", np.nan) for p in post])
        Ec = np.array([p.get("E_c_ry", np.nan) for p in post])
        R = E - Ex - Ec
        g = lambda y: np.gradient(y, ms)        # noqa: E731
        mfree, efree = minimum(ms, E)
        row = {"m": ms.tolist(), "E_tot_ry": E.tolist(), "E_x_ry": Ex.tolist(), "E_c_ry": Ec.tolist(), "E_rest_ry": R.tolist(),
               "E_minus_E0_meV": ((E - E[0]) * RY_TO_MEV).tolist(),
               "dE_dm_meV": (g(E) * RY_TO_MEV).tolist(), "dEx_dm_meV": (g(Ex) * RY_TO_MEV).tolist(),
               "dEc_dm_meV": (g(Ec) * RY_TO_MEV).tolist(), "dRest_dm_meV": (g(R) * RY_TO_MEV).tolist(),
               "predicted_moment": mfree, "E_min_ry": efree,
               "iterations": [r.get("iterations") for r in rec],
               "start": [(r.get("warm_start") or {}).get("label", "cold") for r in rec],
               "missing_m_unconverged": missing,
               "m_achieved": [r["magnetization"]["total"] for r in rec],
               "abs_m": [r["magnetization"]["absolute"] for r in rec]}
        row["stoner"] = stoner(ms, E, Ex + Ec, {k: [p.get("pieces_ry", {}).get(k, np.nan) for p in post]
                                               for k in post[0].get("pieces_ry", {})}, mfree)
        if post[0].get("m_weighted_mean"):
            row["m_weighted_w_x_up"] = [p["m_weighted_mean"]["w_x_up"] for p in post]
            row["m_weighted_w_x_dn"] = [p["m_weighted_mean"]["w_x_dn"] for p in post]
            row["m_weighted_w_c"] = [p["m_weighted_mean"]["w_c"] for p in post]
            row["pieces_ry"] = {k: [p["pieces_ry"][k] for p in post] for k in post[0]["pieces_ry"]}
            row["pairing"] = {}
            for v in [k for k in post[0]["pairing_ry"] if "E_xc" in post[0]["pairing_ry"][k]]:
                Exc_v = np.array([p["pairing_ry"][v]["E_xc"] for p in post])
                Exc_m = np.array([p["E_xc_model_ry"] for p in post])
                Ev = E - Exc_m + Exc_v
                mv, ev = minimum(ms, Ev)
                row["pairing"][v] = {"E_xc_ry": Exc_v.tolist(), "E_tot_frozen_ry": Ev.tolist(),
                                     "E_xc_curvature_meV_per_muB2": {
                                         "m0": None if curv0(ms, Exc_v) is None else curv0(ms, Exc_v) * RY_TO_MEV,
                                         "free": curv_at(ms, Exc_v, mv) * RY_TO_MEV},
                                     "E_minus_E0_meV": ((Ev - Ev[0]) * RY_TO_MEV).tolist(), "implied_moment": mv,
                                     "m_weighted_w_c": [p["pairing_ry"][v].get("m_weighted_w_c",
                                                                               p["pairing_ry"][v].get("m_weighted_wbar_x"))
                                                        for p in post]}
        fr = free.get(f)
        if fr:
            row["free_scf"] = {"moment": abs(fr["magnetization"]["total"]), "E_tot_ry": fr["energies_ry"]["etot"]}
        res[f] = row
    if out:
        json.dump(res, open(out, "w"), indent=1, default=float)
    for f, row in res.items():
        if "m" not in row:
            print(f, row)
            continue
        print(f"== {f}: predicted moment {row['predicted_moment']:.3f}", row.get("free_scf", ""),
              f"[curve WITHOUT m = {', '.join(f'{x:g} ({st})' for x, st in row['missing_m_unconverged'])}]"
              if row["missing_m_unconverged"] else "")
        for i, m in enumerate(row["m"]):
            extra = ""
            if "m_weighted_w_c" in row:
                extra = f" w_x {row['m_weighted_w_x_up'][i]:.3f}/{row['m_weighted_w_x_dn'][i]:.3f} w_c {row['m_weighted_w_c'][i]:.3f}"
            print(f"  m {m:.1f} E {row['E_tot_ry'][i]:.6f} dE {row['E_minus_E0_meV'][i]:9.1f} meV  Ex {row['E_x_ry'][i]:.6f} "
                  f"Ec {row['E_c_ry'][i]:.6f}  dE/dm {row['dE_dm_meV'][i]:8.1f} = x {row['dEx_dm_meV'][i]:8.1f} + c "
                  f"{row['dEc_dm_meV'][i]:7.1f} + rest {row['dRest_dm_meV'][i]:8.1f}" + extra
                  + ("" if row["start"][i] == "cold" else f"  [{row['start'][i]}]"))
        for v, pv in row.get("pairing", {}).items():
            print(f"  pairing {v}: implied moment {pv['implied_moment']:.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
