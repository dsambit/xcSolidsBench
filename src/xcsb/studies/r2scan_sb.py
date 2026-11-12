"""Self-consistent test of the Perdew-group spin-balanced r2SCAN (user 2026-11-11; strongxc-v4 line, xchole line 'v4-sb').

Claim under test: 'r2SCAN-SB moments ~ PBE' (their presentation, private communication). Every functional at the SAME settings:
nspin 2, FM start (starting_magnetization 0.5), 80/320 Ry, Gaussian 0.02 Ry, 8^3 Gamma-centred k, conv_thr 1e-8, beta 0.3,
static-lattice experimental constants (Q10): bcc Fe 2.858 A (refdata solids/lattice_constants_static_2026-11-11), fcc Ni 3.507 A
(solids/lattice_constants_static_ni_2026-11-11). hcp Co: not in the source (Zhang et al. NJP 2018 SM), not staged. The cutoff
80 Ry is the Fe setting; the M1 ladder has not covered Ni. Absolute moments carry the SG15 (no NLCC) caveat; the comparison is
between functionals at identical settings.

Cases: SB-<El> (free moment) and SB-Fe-fsm-<m> (tot_magnetization m, m in FSM_M). Functionals: PBE, r2SCAN (reference
functionals: computed once at this protocol and banked; the earlier Fe80 runs used a = 2.866 A, not these settings),
r2SCAN-SB (R0 half split, primary), r2SCAN-SB-kin."""
from __future__ import annotations

import json
import os

from ..bench.structures import BOHR_PER_ANGSTROM, bcc, fcc
from ..host.qe import Case

A_STATIC = {"Fe": ("bcc", 2.858, "refdata solids/lattice_constants_static_2026-11-11"),
            "Ni": ("fcc", 3.507, "refdata solids/lattice_constants_static_ni_2026-11-11")}
FUNCTIONALS = ("PBE", "r2SCAN", "r2SCAN-SB", "r2SCAN-SB-kin")
FSM_M = (2.2, 2.4, 2.6)


def get_case(name) -> Case:
    parts = name.split("-")
    if parts[0] != "SB" or parts[1] not in A_STATIC:
        raise KeyError(f"not an SB case: {name!r}")
    el = parts[1]
    kind, a, _ = A_STATIC[el]
    s = (bcc if kind == "bcc" else fcc)(el, a * BOHR_PER_ANGSTROM)
    kw = dict(ecutwfc=80.0, kmesh=(8, 8, 8), occupations="smearing", degauss=0.02, nspin=2, starting_magnetization={el: 0.5},
              conv_class="bench")
    if len(parts) == 4 and parts[2] == "fsm":
        kw["tot_magnetization"] = float(parts[3])
    elif len(parts) != 2:
        raise KeyError(f"not an SB case: {name!r}")
    return Case(s, **kw)


def tasks(run):
    t = [{"id": f"SB-{el}__{f}", "case": f"SB-{el}", "functional": f, "width": 1, "stage": "S", "est_s": 1200.0}
         for el in A_STATIC for f in FUNCTIONALS]
    t += [{"id": f"SB-Fe-fsm-{m:.1f}__{f}", "case": f"SB-Fe-fsm-{m:.1f}", "functional": f, "width": 1, "stage": "S",
           "est_s": 1200.0} for m in FSM_M for f in FUNCTIONALS]
    return t


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
    return ["scf", "--case", task["case"], "--mode", "hooked", "--name", task["functional"], "--root", run]


def table(run):
    RYMEV = 13605.693122994
    out = {"settings": "80/320 Ry, 8^3 k, Gaussian 0.02 Ry, conv_thr 1e-8, beta 0.3, FM start 0.5, static lattice",
           "caveat": "SG15 ONCV without NLCC: absolute moments carry a pseudopotential error; compare functionals at equal settings",
           "free": {}, "fsm_Fe": {}}
    for el in A_STATIC:
        for f in FUNCTIONALS:
            r = record(run, {"case": f"SB-{el}", "functional": f})
            out["free"][f"{el}|{f}"] = (None if r is None else
                                        {"status": r.get("status"), "m": (r.get("magnetization") or {}).get("total"),
                                         "abs_m": (r.get("magnetization") or {}).get("absolute"),
                                         "etot_ry": (r.get("energies_ry") or {}).get("etot"), "iterations": r.get("iterations"),
                                         "error": r.get("error")})
    for f in FUNCTIONALS:
        rows = {}
        for m in FSM_M:
            r = record(run, {"case": f"SB-Fe-fsm-{m:.1f}", "functional": f})
            rows[f"{m:.1f}"] = None if r is None or r.get("status") != "ok" else r["energies_ry"]["etot"]
        e0 = rows.get(f"{FSM_M[0]:.1f}")
        out["fsm_Fe"][f] = {"etot_ry": rows, "E_minus_E2.2_meV": {k: None if v is None or e0 is None else (v - e0) * RYMEV
                                                                  for k, v in rows.items()}}
    pbe = {el: (out["free"].get(f"{el}|PBE") or {}).get("m") for el in A_STATIC}
    out["m_minus_PBE"] = {k: (None if v is None or v.get("m") is None or pbe[k.split("|")[0]] is None
                              else abs(v["m"]) - abs(pbe[k.split("|")[0]])) for k, v in out["free"].items()}
    return out
