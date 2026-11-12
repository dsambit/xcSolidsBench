"""Self-consistent E1 on free bcc Fe (user go 2026-11-12), at the SB-Fe settings (static a 2.858 A, 80/320 Ry, 8^3 k, Gaussian
0.02 Ry, beta 0.3, FM start, cold carrier start): the installed SCH-SI-A/B:v3.1 (no E1, now at 2.858 A) and each + E1 at omega 10
and 30 with R0 half (the MODEL form since 2026-11-12, '+E1(omega=<w>)') and R0 kin (labelled comparison, '+E1kin(omega=<w>)').
Per run: moment, |m|, E, iterations, the term's Delta E and the gate retention in the 3d region (post 'e1'), and the per-spin w_x
(post 'fsm_parts'); xchole line v4-e1."""
from __future__ import annotations

import json
import os

SETS = ("SCH-SI-A:v3.1", "SCH-SI-B:v3.1")
OMEGAS = (10.0, 30.0)
CASE = "SB-Fe"


def names():
    out = list(SETS)
    for s in SETS:
        for w in OMEGAS:
            out += [f"{s}+E1(omega={w:g})", f"{s}+E1kin(omega={w:g})"]
    return out


SCAN_OMEGAS = (60.0, 90.0, 150.0, 300.0)      # stage S (user go 2026-11-12): the SC omega scan of E1 (half, the model form)


def scan_names():
    return [f"{s}+E1(omega={w:g})" for s in SETS for w in SCAN_OMEGAS]


def tasks(run):
    return ([{"id": f"E__{f}", "case": CASE, "functional": f, "width": 1, "stage": "E", "est_s": 900.0} for f in names()]
            + [{"id": f"S__{f}", "case": CASE, "functional": f, "width": 1, "stage": "S", "est_s": 900.0} for f in scan_names()])


def record(run, task):
    from ..tiers import run_dir
    p = os.path.join(run_dir(run, task["case"], "hooked", task["functional"]), "record.json")
    return json.load(open(p)) if os.path.exists(p) else None


def done(run, task):
    r = record(run, task)
    return r is not None and r.get("status") == "ok"


def failed(run, task):
    r = record(run, task)
    return r is not None and r.get("status") == "failed"


def command(run, task):
    a = ["scf", "--case", task["case"], "--mode", "hooked", "--name", task["functional"], "--root", run, "--fsm-parts"]
    return a + (["--e1-diag"] if "+E1" in task["functional"] else [])


def table(run):
    out = {}
    seen = set()
    for t in tasks(run):
        if t["functional"] in seen:
            continue
        seen.add(t["functional"])
        r = record(run, t)
        if r is None or r.get("status") != "ok":
            out[t["functional"]] = {"status": None if r is None else r.get("status"), "error": None if r is None else r.get("error")}
            continue
        post = r.get("post") or {}
        e1 = post.get("e1") or {}
        mw = (post.get("fsm_parts") or {}).get("m_weighted_mean") or {}
        out[t["functional"]] = {"status": "ok", "m": r["magnetization"]["total"], "abs_m": r["magnetization"]["absolute"],
                                "etot_ry": r["energies_ry"]["etot"], "iterations": r["iterations"],
                                "carrier_iterations": r["carrier"]["iterations"],
                                "delta_E_meV": None if "delta_E_ry" not in e1 else e1["delta_E_ry"] * 13605.693122994,
                                "retained_3d": (e1.get("gate_3d") or {}).get("retained"),
                                "retained_all": (e1.get("gate_all") or {}).get("retained"),
                                "w_x_up": mw.get("w_x_up"), "w_x_dn": mw.get("w_x_dn"), "w_c": mw.get("w_c")}
    return out


PBE_REF = 2.359          # our PBE Fe moment at these settings (r2SCAN-SB run, SB-Fe PBE): a reference column only
# the omega-fit TARGET (user decision 2026-11-12): 2.22 mu_B, the experimental TOTAL moment of bcc Fe (spin + orbital);
# "user-provided experimental total moment; no archive source read by us". Our moments are SPIN moments (collinear, no SOC):
# the fit compares a spin moment to a total moment, by the user's choice; it also absorbs our SG15 (no NLCC) offset.
TARGET_MOMENT = 2.22
TARGET_NOTE = ("user-provided experimental total moment (spin + orbital) of bcc Fe; no archive source read by us; compared with "
               "our spin moment (collinear, no SOC) by the user's choice")


def m_of_omega(run, target=TARGET_MOMENT, pbe=PBE_REF):
    """Per set: |m|(omega) of the SC half-form E1 runs (stages E and S), and log-omega interpolation to the target (the
    EXPERIMENTAL spin moment, a parameter: None until the user provides a sourced value) and to our PBE moment (reference).
    A crossing is reported only where m(omega) is monotone between the bracketing points."""
    import math
    import re
    t = table(run)
    out = {"target": target, "target_note": TARGET_NOTE if target == TARGET_MOMENT else "explicit", "pbe_reference": pbe,
           "sets": {}}
    for s in SETS:
        pts = []
        for f, v in t.items():
            m = re.fullmatch(re.escape(s) + r"\+E1\(omega=([0-9.eE+-]+)\)", f)
            if m and v.get("status") == "ok":
                pts.append((float(m.group(1)), abs(v["m"])))
        pts.sort()
        r = {"points": pts}
        for key, val in (("target", target), ("pbe", pbe)):
            if val is None:
                r[f"omega_at_{key}"] = None
                continue
            hit = None
            for (w0, m0), (w1, m1) in zip(pts[:-1], pts[1:]):
                if (m0 - val) * (m1 - val) <= 0 and m0 != m1:
                    x = (val - m0) / (m1 - m0)
                    hit = math.exp(math.log(w0) + x * (math.log(w1) - math.log(w0)))
                    break
            r[f"omega_at_{key}"] = hit
        ms = [m for _, m in pts]
        r["monotone"] = all(a <= b for a, b in zip(ms[:-1], ms[1:])) or all(a >= b for a, b in zip(ms[:-1], ms[1:]))
        out["sets"][s] = r
    return out
