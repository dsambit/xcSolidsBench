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


# ---- the exchange outer-switch floor (user 2026-11-14) ------------------------------------------------------------------------------
# Lever: let the exchange outer switch keep a nonzero iso-orbital weight at the uniform-gas end, w' = w0 + (1 - w0) w (breaks
# GE2X and GE4X; the uniform-gas limit itself unless the iso-orbital exchange equals LDA there: 'ueg_point' below measures it).
# At a frozen density the energy is linear in w0: E(w0) = E + w0 I, I = sum_sigma int (1 - w) (e_iso - e_UEG). The post below
# returns I and the pieces on the converged density (serial on the root).
XFLOOR_TAG = "xfloor"
XFLOOR_W0 = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)


def xfloor_post(functional_name):
    def post(grid, rho, tau):
        import jax
        import jax.numpy as jnp
        from xchole.sch_si.exchange import spin_pieces
        from ..functional import get
        from ..host.evaluate import point_inputs
        f = get(functional_name)
        Pm = f._model.P
        x, _ = point_inputs(grid, rho, tau, True, True)

        def k(r, g, t, l):
            ex, pc = spin_pieces(r, g, t, l, Pm)
            e_iso = (1 - pc["gs"]) * pc["e_br"] + pc["gs"] * pc["e_sh"]
            return ex, pc["w"], pc["gs"], e_iso, pc["e_def"]
        kj = jax.jit(k)
        w8 = grid.weight
        ry = lambda a: 2.0 * w8 * float(np.sum(a))          # noqa: E731
        out = {"functional": functional_name, "I_ry": 0.0, "E_x_ry": 0.0, "E_x_iso_only_ry": 0.0, "E_x_ueg_only_ry": 0.0,
               "X_iso_ry": 0.0, "X_ueg_ry": 0.0}
        nw = dw = 0.0
        hist = np.zeros(10)
        for sp in (0, 1):
            ex, w, gs, e_iso, e_def = (np.asarray(a) for a in kj(x.rho[sp], x.grad[sp], x.tau[sp], x.lapl[sp]))
            out["I_ry"] += ry((1 - w) * (e_iso - e_def))
            out["E_x_ry"] += ry(ex)
            out["E_x_iso_only_ry"] += ry(e_iso)
            out["E_x_ueg_only_ry"] += ry(e_def)
            out["X_iso_ry"] += ry(w * e_iso)
            out["X_ueg_ry"] += ry((1 - w) * e_def)
            nw += float(np.sum(x.rho[sp] * w)); dw += float(np.sum(x.rho[sp]))
            hist += np.histogram(w, bins=10, range=(0.0, 1.0), weights=x.rho[sp])[0]
        out["rho_weighted_mean_w_x"] = nw / dw
        out["electron_fraction_by_w_x_decile"] = (hist / hist.sum()).tolist()
        # the uniform gas (per spin density n / 2, grad = lapl = 0, tau = tau_unif): iso-orbital exchange relative to the UEG end
        ueg = {}
        for rs in (0.5, 1.0, 2.0, 5.0):
            r = jnp.array([0.5 * 3.0 / (4.0 * np.pi * rs ** 3)])
            t = 0.3 * (6.0 * np.pi ** 2) ** (2.0 / 3.0) * r ** (5.0 / 3.0)
            ex, w, gs, e_iso, e_def = (float(np.asarray(a)[0]) for a in kj(r, jnp.zeros((3, 1)), t, jnp.zeros(1)))
            ueg[f"rs{rs:g}"] = {"e_iso_over_e_ueg": e_iso / e_def, "w_x": w, "shell_weight": gs,
                                "e_ueg_over_lda": e_def / (-0.75 * (3.0 / np.pi) ** (1.0 / 3.0) * 2.0 ** (1.0 / 3.0) * float(r[0]) ** (4.0 / 3.0))}
        out["ueg_point"] = ueg
        return out
    return post


def xfloor_command(run, step):
    return ["scf", "--case", case_name(step), "--mode", "hooked", "--name", BASE, "--root", L.scf_root(run), "--npool", str(L.NPOOL),
            "--tag", XFLOOR_TAG, "--xfloor-diag"]


def xfloor_summary(run):
    from ..bench.eos import bm3_fit
    from ..bench.structures import BOHR_PER_ANGSTROM
    from ..tiers import run_dir
    pts = []
    for s in STEPS:
        p = os.path.join(run_dir(L.scf_root(run), case_name(s), "hooked", BASE) + "__" + XFLOOR_TAG, "record.json")
        r = json.load(open(p)) if os.path.exists(p) else None
        if r is None or r.get("status") != "ok" or "I_ry" not in r.get("post", {}).get("xfloor", {}):
            pts.append({"step": s, "status": "missing" if r is None else r.get("status"),
                        "error": None if r is None else (r.get("error") or r.get("post", {}).get("xfloor", {}).get("error"))})
            continue
        pts.append(dict(r["post"]["xfloor"], step=s, status="ok", a_angstrom=L.A_REF[SOLID] * (1 + s / 100.0), V_bohr3=L.volume(SOLID, s),
                        etot_ry=r["energies_ry"]["etot"]))
    ok = [p for p in pts if p["status"] == "ok"]
    rows = []
    for w0 in XFLOOR_W0:
        row = {"w0": w0, "etot_ry": [p["etot_ry"] + w0 * p["I_ry"] for p in ok]}
        if len(ok) >= 4:
            ft = bm3_fit(np.array([p["V_bohr3"] for p in ok]), np.array(row["etot_ry"]), guard=False)
            row.update(a0_angstrom=(4 * ft["V0_bohr3"]) ** (1 / 3) / BOHR_PER_ANGSTROM, B0_GPa=ft["B0_GPa"], inside=ft["v0_inner_two_thirds"])
        rows.append(row)
    return {"what": "LiF: exchange outer-switch floor w' = w0 + (1 - w0) w, frozen-density estimate (bare, no D4)", "base": BASE,
            "a_ref_angstrom": L.A_REF[SOLID], "points": pts, "rows": rows}


def render_xfloor(t) -> str:
    out = [f"# {t['what']}", "", "| step (%) | E_tot (Ry) | I (Ry) | E_x (Ry) | E_x iso only | E_x UEG only | mean w_x |", "|---|---|---|---|---|---|---|"]
    for p in t["points"]:
        if p["status"] == "ok":
            out.append(f"| {p['step']:+d} | {p['etot_ry']:.6f} | {p['I_ry']:+.6f} | {p['E_x_ry']:.5f} | {p['E_x_iso_only_ry']:.5f} | "
                       f"{p['E_x_ueg_only_ry']:.5f} | {p['rho_weighted_mean_w_x']:.3f} |")
        else:
            out.append(f"| {p['step']:+d} | {p['status']}: {p.get('error')} | | | | | |")
    out += ["", "| w0 | a0 (A) | a0 - a_ref (%) | B0 (GPa) |", "|---|---|---|---|"]
    for r in t["rows"]:
        if "a0_angstrom" in r:
            out.append(f"| {r['w0']:g} | {r['a0_angstrom']:.4f} | {100 * (r['a0_angstrom'] / t['a_ref_angstrom'] - 1):+.2f} | {r['B0_GPa']:.1f} |"
                       + ("" if r["inside"] else " (minimum near the scan edge)"))
    ok = [p for p in t["points"] if p["status"] == "ok"]
    if ok:
        out += ["", "Uniform gas, iso-orbital exchange / UEG-end exchange: " + ", ".join(
            f"{k}: {v['e_iso_over_e_ueg']:.4f} (w_x {v['w_x']:.3f}, UEG end / LDA {v['e_ueg_over_lda']:.4f})" for k, v in ok[0]["ueg_point"].items()),
            "", "Electron fraction by w_x decile at step " + f"{ok[0]['step']:+d} %: " + ", ".join(f"{v:.3f}" for v in ok[0]["electron_fraction_by_w_x_decile"])]
    return "\n".join(out) + "\n"
