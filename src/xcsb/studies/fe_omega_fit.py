"""Fit of the E1 gate width omega to the bcc Fe moment (user decision 2026-11-11; strongxc-v4 rules/implementation.md exception
92b1cfb56), per set A and B, self-consistently. Co and Ni stay hold-outs (never evaluated here).

xchole line v4-e1 (strongxc-v4 76cfa7c20). Settings (identical for the target and the fit; target confirmed by the user 2026-11-11): SB-Fe of xcsb.studies.r2scan_sb: bcc Fe, static a 2.858 A, nspin 2, FM start,
80/320 Ry, 8^3 k, Gaussian 0.02 Ry, conv_thr 1e-8, beta 0.3, cold carrier start. TARGET: our own PBE moment at these settings,
read from the PBE record of the r2SCAN-SB run (TARGET_RUN; that job computes it once); the pseudopotential offset cancels.
(The FSM free PBE run, 2.381 mu_B, used a = 2.866 A: not the same settings.)
Evaluator: SCH-SI-X:v3.1 + E1 (UEG-gauge reference, base placement, R0 half: the model form since 2026-11-12) via strongxc-v4 sb_model.install (xcsb.functional.
xchole.XcholeE1; exact JAX derivatives; FD-checked in phase 0).

1. Bracket (frozen, item 2's omega scan; sb_frozen_summary.json): the implied frozen moments at omega in {10, 30, 90, 300, 1000}
   and lam = 1 (omega = 0); the bracket is the adjacent pair straddling the target. No bracket, or a bracket below omega = 10
   (between lam = 1 and 10: the gate nearly inactive), STOPS.
2. Fit: SC free moments at the bracket's two ends and their geometric mean, per set (stage F).
3. omega*: interpolation of m(log omega) through the three points; m must be monotone in omega and the target inside the
   computed range, else STOP. Then the one-electron check at omega*: the term on the exact H-atom density must vanish
   (|Delta E| <= H_TOL); a failure STOPS.
4. Confirm: one SC run at omega* per set (stage C) with the E1 parts (post 'e1') and the base pieces (--fsm-parts)."""
from __future__ import annotations

import json
import math
import os

import numpy as np

SETS = ("SCH-SI-A:v3.1", "SCH-SI-B:v3.1")
SCAN_W = (10.0, 30.0, 90.0, 300.0, 1000.0)
W_MIN = 10.0
H_TOL = 1e-6                     # Ha: |Delta E| of the term on the H atom
CASE = "SB-Fe"
TARGET_RUN = os.environ.get("XCSB_SB_RUN", "/pscratch/sd/d/dsambit/xc_functional_dev/runs/xcsb/v4_2026-11-12_r2scan_sb")
FROZEN = os.environ.get("XCSB_SB_FROZEN", "/pscratch/sd/d/dsambit/xc_functional_dev/runs/xcsb/v4_2026-11-12_fe_sb")


def fname(s, w):
    return f"{s}+E1(omega={w:.6g})"


def target():
    from ..tiers import run_dir
    p = os.path.join(run_dir(TARGET_RUN, CASE, "hooked", "PBE"), "record.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}: the PBE target at the fit settings is missing (run the r2SCAN-SB job first)")
    r = json.load(open(p))
    if r.get("status") != "ok":
        raise RuntimeError(f"{p}: PBE target run {r.get('status')}")
    return abs(r["magnetization"]["total"]), p


def bracket(summary, set_key, m_target):
    """(w_lo, w_hi) from the frozen implied moments; raises (STOP) without a bracket or below W_MIN."""
    cur = summary["curves"][set_key]
    rows = [(0.0, cur["lam1_no_gate"]["implied_moment"])] + [(w, cur["default" if w == 90.0 else f"omega{w:g}"]["implied_moment"])
                                                             for w in SCAN_W]
    for (w0, m0), (w1, m1) in zip(rows[:-1], rows[1:]):
        if (m0 - m_target) * (m1 - m_target) <= 0:
            if w0 < W_MIN:
                raise RuntimeError(f"{set_key}: the frozen moments bracket the target between omega {w0:g} and {w1:g}: "
                                   f"the fit would need omega below ~{W_MIN:g} (gate nearly inactive): STOP")
            return w0, w1, rows
    raise RuntimeError(f"{set_key}: the frozen implied moments {rows} never reach the target {m_target:.4f}: STOP")


def plan(run):
    """FIT.json (written once): the target, the frozen rows and the three omega values per set."""
    p = os.path.join(run, "FIT.json")
    if os.path.exists(p):
        return json.load(open(p))
    mt, src = target()
    summary = json.load(open(os.path.join(FROZEN, "sb_frozen_summary.json")))
    out = {"target_moment": mt, "target_record": src, "sets": {}}
    for s in SETS:
        w0, w1, rows = bracket(summary, "A" if "-A:" in s else "B", mt)
        out["sets"][s] = {"frozen_rows": rows, "bracket": [w0, w1], "omega": [w0, math.sqrt(w0 * w1), w1]}
    json.dump(out, open(p, "w"), indent=1)
    return out


def _rec(run, f):
    from ..tiers import run_dir
    p = os.path.join(run_dir(run, CASE, "hooked", f), "record.json")
    return json.load(open(p)) if os.path.exists(p) else None


def omega_star(run):
    """{set: omega*} from the three SC fit points (monotone m(omega), target inside), else raises (STOP)."""
    fit = plan(run)
    mt, out = fit["target_moment"], {}
    for s, v in fit["sets"].items():
        pts = []
        for w in v["omega"]:
            r = _rec(run, fname(s, w))
            if r is None or r.get("status") != "ok":
                raise RuntimeError(f"{s} omega {w:g}: SC run {None if r is None else r.get('status')}")
            pts.append((w, abs(r["magnetization"]["total"])))
        ms = [m for _, m in pts]
        if not (all(a < b for a, b in zip(ms[:-1], ms[1:])) or all(a > b for a, b in zip(ms[:-1], ms[1:]))):
            raise RuntimeError(f"{s}: m(omega) {pts} is not monotone: STOP")
        if not min(ms) <= mt <= max(ms):
            raise RuntimeError(f"{s}: m(omega) {pts} does not reach the target {mt:.4f}: STOP")
        lw = np.log([w for w, _ in pts])
        order = np.argsort(ms)
        out[s] = {"points": pts, "omega_star": float(np.exp(np.interp(mt, np.array(ms)[order], lw[order])))}
    return out


def tasks(run):
    t = []
    try:
        fit = plan(run)
    except (FileNotFoundError, RuntimeError):
        return t
    for s, v in fit["sets"].items():
        for w in v["omega"]:
            t.append({"id": f"F__{fname(s, w)}", "functional": fname(s, w), "stage": "F", "width": 1, "est_s": 1500.0})
    p = os.path.join(run, "OMEGA_STAR.json")
    if os.path.exists(p):
        for s, v in json.load(open(p))["sets"].items():
            t.append({"id": f"C__{fname(s, v['omega_star'])}", "functional": fname(s, v["omega_star"]), "stage": "C", "width": 1,
                      "est_s": 1500.0, "confirm": True})
    return t


def done(run, task):
    r = _rec(run, task["functional"])
    return r is not None and r.get("status") == "ok"


def failed(run, task):
    r = _rec(run, task["functional"])
    return r is not None and r.get("status") == "failed"


def command(run, task):
    return ["scf", "--case", CASE, "--mode", "hooked", "--name", task["functional"], "--root", run] + \
        (["--fsm-parts", "--e1-diag"] if task.get("confirm") else ["--e1-diag"])


# ---- the one-electron check (H atom, exact density) -------------------------------------------------------------------------------
def h_atom_inputs(nr=4000, rmax=30.0):
    """Exact 1s density (spin up only) on a radial grid as 10 point arrays (sb_model convention) and the radial weights 4 pi r^2 dr."""
    r = np.linspace(rmax / nr, rmax, nr)
    w = 4 * np.pi * r ** 2 * np.full(nr, r[1] - r[0])
    n = np.exp(-2 * r) / np.pi
    dn, d2n = -2 * n, 4 * n
    rhat = np.stack([np.ones(nr), np.zeros(nr), np.zeros(nr)])      # along x (spherical: any direction)
    g = dn * rhat
    lap = d2n + 2 * dn / r
    H = np.zeros((6, nr))
    H[0] = d2n                                                       # xx (radial)
    H[3] = dn / r                                                    # yy
    H[5] = dn / r                                                    # zz
    tau = np.sum(g ** 2, axis=0) / (8 * n)                           # tau_W: one orbital
    z = np.zeros(nr)
    return (n, z, g, np.zeros((3, nr)), tau, z, lap, z, H, np.zeros((6, nr))), w


def h_check(omega, sets=("A", "B")):
    import sys
    from ..functional.xchole import import_pinned_xchole
    import_pinned_xchole()
    root = os.path.dirname(os.path.abspath(os.environ["XCSB_XCHOLE_SRC"]))
    sys.path.insert(0, os.path.join(root, "explore", "spin_balance"))
    import sb_model as SB
    d, w = h_atom_inputs()
    out = {}
    for b in sets:
        o = SB.e_xc_points(*d, base=b, ref="ueg_end", omega=float(omega[b]), placement="base", indicators=False, r0_form="half")
        out[b] = {"omega": float(omega[b]), "delta_E_Ha": float(w @ o["de"]), "E_c_si_Ha": float(w @ o["e_c_si"]),
                  "E_x_Ha": float(w @ o["e_x"]), "pass": bool(abs(float(w @ o["de"])) <= H_TOL)}
    return out


def main(argv=None):
    """python -m xcsb.studies.fe_omega_fit RUN: omega* per set and the H check -> OMEGA_STAR.json, or STOP.txt."""
    import sys
    run = (argv or sys.argv[1:])[0]
    try:
        st = omega_star(run)
        hc = h_check({("A" if "-A:" in s else "B"): v["omega_star"] for s, v in st.items()})
        if not all(v["pass"] for v in hc.values()):
            raise RuntimeError(f"the one-electron (H) check fails at omega*: {hc}: STOP")
        json.dump({"sets": st, "h_check": hc}, open(os.path.join(run, "OMEGA_STAR.json"), "w"), indent=1)
        print(json.dumps({"sets": st, "h_check": hc}, default=float))
        return 0
    except Exception as exc:
        open(os.path.join(run, "STOP.txt"), "w").write(f"{type(exc).__name__}: {exc}\n")
        print(f"STOP: {exc}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
