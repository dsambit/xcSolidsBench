"""LiF lattice constant: frozen-density response to the model parameters (user 2026-11-14: A and B expand the alkali halides
by 4-7 %; find what brings LiF under 1 % with PBE-D4). DIAGNOSTIC, never a result.

Self-consistent SCH-SI-A:v3.2 at a = a_ref (1 + s), s in STEPS, on the coarse mesh of the mixer test (ionic_lc.MIX_K^3 k; LiF at
a_ref: E_tot equal to the 28^3 value to 1e-6 Ry), protocol cutoffs. On each converged density the E_xc of every entry of
frozen_names() is evaluated (scf.driver.run_hooked, frozen): one-at-a-time edits of A's theta fields to the other end (or
inside) of their fit ranges (xchole registry), plus SCH-SI-B:v3.2, r2SCAN and PBE. To first order in the density change

    E_tot^g(a) = E_tot^A(a) + E_xc^g[rho_A(a)] - E_xc^A[rho_A(a)],

so each entry gets an E(V) curve and a lattice constant from the A runs alone; r2SCAN and PBE check the estimate against their
self-consistent lattice constants (3.9934, 4.0622 A). Norm-dependent edits are re-solved first (xcsb prep-variant); values
xchole refuses are reported. The PBE-D4 correction is geometry-only and added afterwards (dftd4 on the laptop).
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import ionic_lc as L

SOLID = "LiF"
BASE = "SCH-SI-A:v3.2"
STEPS = (-2, 0, 2, 4, 6, 8)
# theta field -> the values tried (A's value in the comment; fit ranges from xchole registry.PARAMS[...].range)
EDITS = {
    "lam": (0.5, 0.65),            # 0.35   [0.35, 0.8]   iso-orbital end, exchange
    "eps": (0.001, 0.01),          # 0.00745 [0.001, 0.01]
    "a_L": (0.5, 1.5),             # 0.15   [0.15, 3]
    "c_min": (0.11,),              # 0.10   [0.1, 0.11]
    "C_w": (0.0837577015246072,),  # 0.2094 (v3.1's value)
    "nc_kd": (4.0, 13.0),          # 8.83   [2, 15]       inner switch
    "nc_ke": (5.0,),               # 10     [2, 10]
    "beta_g": (0.2,),              # 0.11   [0.11, 0.3]
    "a_c": (1.0,),                 # 4      [0.3, 4]      outer switches
    "kappa_r": (0.0059,),          # 0.00445 [0.00445, 0.0059]
    "t_H": (0.01,),                # 0.04
    "y_w": (4.0,),                 # 30
    "b_rise": (0.0,),              # 0.05
    "c4": (6.0, 10.0),             # 2      [2, 10]       UEG-gauge end, correlation
    "c4_os": (0.0, 6.0),           # 2.60   [0, 10]
    "c_qs": (-0.185,),             # -0.16  [-0.185, -0.16] UEG-gauge end, exchange
    "d_s4": (0.0,),                # 0.007  [0, 0.007]
    "r_c": (5.0,),                 # 7      [5, 7]        Lambda'
    "kappa_G": (0.0,),             # 1
}
OTHERS = ("SCH-SI-B:v3.2", "r2SCAN", "PBE")


def variant_names():
    return [f"{BASE}@{k}={v!r}" for k, vs in EDITS.items() for v in vs]


def frozen_names():
    return variant_names() + list(OTHERS)


def case_name(step):
    return L.mix_case_name(SOLID, step, L.MIX_K, "reset")


def command(run, step):
    return ["scf", "--case", case_name(step), "--mode", "hooked", "--name", BASE, "--root", L.scf_root(run), "--npool", str(L.NPOOL),
            "--frozen-file", os.path.join(run, "frozen.txt")]


def record(run, step):
    from ..tiers import run_dir
    p = os.path.join(run_dir(L.scf_root(run), case_name(step), "hooked", BASE), "record.json")
    return json.load(open(p)) if os.path.exists(p) else None


def summary(run):
    """Per step the base energies and every frozen E_xc; per entry the first-order E(V) and its unguarded BM3."""
    from ..bench.eos import bm3_fit
    from ..bench.structures import BOHR_PER_ANGSTROM
    pts = []
    for s in STEPS:
        r = record(run, s)
        if r is None or r.get("status") != "ok":
            pts.append({"step": s, "status": "missing" if r is None else r.get("status"), "error": None if r is None else r.get("error")})
            continue
        pts.append({"step": s, "status": "ok", "a_angstrom": L.A_REF[SOLID] * (1 + s / 100.0), "V_bohr3": L.volume(SOLID, s),
                    "etot_ry": r["energies_ry"]["etot"], "e_xc_ry": r["e_xc_final_ry"], "iterations": r["iterations"],
                    "frozen": {n: (v.get("e_xc_ry"), v.get("error")) for n, v in r.get("frozen", {}).items()}})
    ok = [p for p in pts if p["status"] == "ok"]
    rows = []
    for n in [BASE] + frozen_names():
        row = {"name": n}
        e = [p["etot_ry"] if n == BASE else (None if p["frozen"].get(n, (None,))[0] is None else p["etot_ry"] + p["frozen"][n][0] - p["e_xc_ry"])
             for p in ok]
        errs = sorted({p["frozen"][n][1] for p in ok if n != BASE and n in p["frozen"] and p["frozen"][n][1]})
        row.update(etot_ry=e, error=errs[0][:200] if errs else None)
        if len(ok) >= 4 and all(x is not None for x in e):
            try:
                ft = bm3_fit(np.array([p["V_bohr3"] for p in ok]), np.array(e), guard=False)
                row.update(a0_angstrom=(4 * ft["V0_bohr3"]) ** (1 / 3) / BOHR_PER_ANGSTROM, B0_GPa=ft["B0_GPa"], rms_ry=ft["rms_ry"],
                           inside=ft["v0_inner_two_thirds"])
            except Exception as exc:
                row["error"] = f"fit: {type(exc).__name__}: {exc}"
        rows.append(row)
    return {"what": __doc__.split("\n")[0], "solid": SOLID, "a_ref_angstrom": L.A_REF[SOLID], "base": BASE, "kmesh": L.MIX_K,
            "points": pts, "rows": rows}


def render(t) -> str:
    out = [f"# {t['what']}", "", f"a_ref {t['a_ref_angstrom']} A; base {t['base']}; first-order (frozen-density) estimates, bare (no D4)", "",
           "| functional / variant | a0 (A) | a0 - a_ref (%) | B0 (GPa) | BM3 rms (Ry) | note |", "|---|---|---|---|---|---|"]
    for r in t["rows"]:
        if "a0_angstrom" in r:
            out.append(f"| {r['name']} | {r['a0_angstrom']:.4f} | {100 * (r['a0_angstrom'] / t['a_ref_angstrom'] - 1):+.2f} | {r['B0_GPa']:.1f} | "
                       f"{r['rms_ry']:.1e} | {'' if r['inside'] else 'minimum near the scan edge'} |")
        else:
            out.append(f"| {r['name']} |  |  |  |  | {r.get('error') or 'incomplete'} |")
    out += ["", "Points: " + ", ".join(f"{p['step']:+d} %: {p['status']}" for p in t["points"])]
    return "\n".join(out) + "\n"
