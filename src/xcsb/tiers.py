"""Node verification tiers V3-V6 (DESIGN.md Sec. 7).

Every SCF runs in its OWN process (`xcsb scf ...`, one srun each): QE keeps functional state (iexch, ...) between QEpy
Drivers in one process, so a native PBE run followed by a TPSS-carrier run in the same process is refused by QE
(set_dft_from_name: conflicting values). The comparisons are then made from the records (`xcsb compare v3|v4|v6`).

    V3  hooks reproduce QE native: libxc PBE / TPSS through the hooks vs QE's input_dft 'PBE' / 'TPSS', on bulk Si
        (nspin 1, fixed occupations) and FM bcc Fe (nspin 2, cold smearing); plus the host's E_xc on each native
        ground-state density vs QE's etxc (also on a strained, displaced, low-symmetry Si cell that pins the grid order).
    V4  nspin = 2 at zero magnetization (Si, tot_magnetization = 0) equals nspin = 1.
    V5  the host step is bitwise independent of the rank count (serial evaluation on the root vs all ranks).
    V6  smoke SCFs through the hooks, cold starts: Si and FM bcc Fe with PBE, r2SCAN, SCH-SI-A:v3.1, SCH-SI-B:v3.1.
"""
from __future__ import annotations

import glob
import json
import os
import time

import numpy as np

from .bench.structures import BOHR_PER_ANGSTROM, Structure, bcc, diamond
from .host.qe import Case

A_SI = 5.431 * BOHR_PER_ANGSTROM          # verification geometry only (not a benchmark reference)
A_FE = 2.866 * BOHR_PER_ANGSTROM

TOL_V3_E = 1e-5        # Ry/atom
TOL_V3_EIG = 1e-5      # Ry
TOL_V3_MAG = 1e-4      # mu_B / cell
TOL_V3_EXC = 1e-6      # relative, host E_xc on the native density vs QE etxc (QE's meta threshold points removed)
TOL_V4_E = 1e-8        # Ry
VALENCE = {"Si": 4, "Fe": 16}      # SG15 1.2 z_valence of the verification cells (pseudo/MANIFEST.json)
FE_MIXING = 0.3        # verification Fe cells: Broyden beta (the global default; 0.1 did not help from the old start)


def case_si(**kw):
    return Case(diamond("Si", A_SI), ecutwfc=kw.pop("ecutwfc", 30.0), kmesh=(4, 4, 4), occupations="fixed", conv_class="verification", **kw)


def case_si_lowsym():
    s = diamond("Si", A_SI)
    strain = np.array([[1.02, 0.01, 0.0], [0.0, 0.99, 0.015], [0.0, 0.0, 1.005]])
    frac = s.frac.copy()
    frac[1] += [0.013, -0.007, 0.004]
    low = Structure("Si-lowsym", s.lattice @ strain, s.labels, frac, s.elements)
    return Case(low, ecutwfc=30.0, kmesh=(3, 3, 3), occupations="fixed", nosym=True, conv_class="verification")


def case_fe(**kw):
    return Case(bcc("Fe", A_FE), ecutwfc=kw.pop("ecutwfc", 60.0), kmesh=(8, 8, 8), occupations="smearing", degauss=0.02,
                nspin=2, starting_magnetization={"Fe": 0.5}, mixing_beta=FE_MIXING, conv_class="verification", **kw)


def case_fe_offset():
    """bcc Fe with the atom at a generic fractional position and no symmetry: no grid point sits at a centre of symmetry,
    so QE's native meta-GGA thresholds (e_xc := 0 where |grad rho|^2 <= 1e-24) never trigger."""
    s = bcc("Fe", A_FE)
    off = Structure("bcc-Fe-offset", s.lattice, s.labels, np.array([[0.137, 0.271, 0.419]]), s.elements)
    return Case(off, ecutwfc=60.0, kmesh=(4, 4, 4), occupations="smearing", degauss=0.02, nspin=2,
                starting_magnetization={"Fe": 0.5}, nosym=True, mixing_beta=FE_MIXING, conv_class="verification")


CASES = {
    "Si": case_si,
    "Si-lowsym": case_si_lowsym,
    "Si-nspin2-m0": lambda: case_si(nspin=2, tot_magnetization=0.0),
    "Fe": case_fe,
    "Fe-offset": case_fe_offset,
}


# ---- one SCF per process ----------------------------------------------------------------------------------------------
def all_cases():
    from .studies import fe_magnetism, fe_payback
    return {**CASES, **fe_payback.CASES, **fe_magnetism.CASES}


def get_case(name):
    """A named case; 'Fe80-fsm-<m>': bcc Fe at 80 Ry with the total magnetization fixed to m (mu_B per cell); 'M1-...': the
    M1 ladder cases (xcsb.studies.m1_ladder.parse_case)."""
    if name.startswith("SB-"):
        from .studies.r2scan_sb import get_case as sb_case
        return sb_case(name)
    if name.startswith("MgO-"):
        from .studies.mgo_gap import get_case as mgo_case
        return mgo_case(name)
    if name.startswith("M1-"):
        from .studies.m1_ladder import get_case as m1_case
        return m1_case(name)
    if name.startswith("Fe80-fsm-"):
        from .studies.fe_magnetism import case_fe80_fsm
        return case_fe80_fsm(float(name.split("-")[-1]))
    return all_cases()[name]()


def scf(case_name: str, mode: str, name: str, out: str, host_check: str | None = None, payback_diag: bool = False,
        maxiter: int | None = None, mixing_beta: float | None = None, mag_split: bool = False, fsm_parts: bool = False,
        npool: int = 1, ripple: str | None = None, m1_diag: bool = False, warm_from: str | None = None,
        warm_label: str | None = None, warm_mode: str | None = None):
    """mode 'native': QE's input_dft = name (host_check: a libxc name for the host E_xc of the native density);
    mode 'hooked': the functional `name` through the hooks. Writes <out>/record.json."""
    from mpi4py import MPI

    from .scf.driver import _write, run_hooked, run_native
    comm = MPI.COMM_WORLD
    case = get_case(case_name)
    deviations = {}
    if mixing_beta is not None:                       # a labelled diagnostic deviation from the global/case beta
        deviations["mixing_beta"] = {"case": case.mixing_beta, "used": mixing_beta}
        case.mixing_beta = mixing_beta
    if maxiter is not None:
        deviations["maxiter"] = {"global": None, "used": maxiter}
    rec_path = os.path.join(out, "record.json")
    if mode == "native":
        try:
            rec = run_native(case, name, out, comm, host_functional=host_check)
            rec["status"] = "ok"
        except Exception as exc:
            rec = {"mode": "native", "input_dft": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        if comm.Get_rank() == 0:
            _write(rec_path, rec)
    else:
        try:
            posts = []
            if payback_diag:
                from .studies.fe_payback import attribution
                posts.append(("payback", lambda g, r, t: attribution(name, g, r, t)))
            if mag_split:
                from .studies.fe_magnetism import split
                posts.append(("mag_split", lambda g, r, t: split(name, g, r, t)))
            if fsm_parts:
                from .studies.fe_magnetism import fsm_parts as _fp
                posts.append(("fsm_parts", lambda g, r, t: _fp(name, g, r, t)))
            if m1_diag:
                from .studies.m1_ladder import post_diag
                posts.append(("m1", post_diag))
            seeds = None
            if ripple:
                from .studies.m1_ladder import SeedPlan
                from .studies.m1_ladder import ripple_bands
                seeds = SeedPlan(case.ecutwfc, bands=ripple_bands(ripple))

            def run_posts(g, r, t):
                out = {}
                for key, fn in posts:                      # each diagnostic separately: one failure does not hide the other
                    try:
                        out[key] = fn(g, r, t)
                    except Exception as exc:
                        out[key] = {"error": f"{type(exc).__name__}: {exc}"}
                return out
            post = run_posts if posts else None
            kw = {"maxiter": maxiter} if maxiter is not None else {}
            warm, previous = None, None
            if warm_from:
                if not warm_label:
                    raise ValueError("a warm start needs its label (e.g. 'warm start from m = 2.6')")
                warm = {"from": warm_from, "label": warm_label, "mode": warm_mode}
                previous = keep_previous_attempts(out, comm)
            rec = run_hooked(case, name, out, comm, record_path=rec_path, post=post, deviations=deviations, npool=npool,
                             seeds=seeds, warm_start=warm, previous_attempts=previous, **kw)
        except Exception as exc:
            rec = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    rec["case_name"] = case_name
    return rec


def keep_previous_attempts(out, comm):
    """A failed record of this point is kept: moved to record_attempt<n>.json (n = 1, 2, ...) and summarized (status,
    error, start, iterations, the full dr2 history) for the new record. Raises if the point already has an ok record."""
    import shutil
    summary, err = None, None
    if comm.Get_rank() == 0:
        p = os.path.join(out, "record.json")
        summary = []
        n = 1
        while os.path.exists(os.path.join(out, f"record_attempt{n}.json")):
            summary.append(_attempt(json.load(open(os.path.join(out, f"record_attempt{n}.json"))), n))
            n += 1
        if os.path.exists(p):
            r = json.load(open(p))
            if r.get("status") == "ok":
                err = f"{p}: the point is already converged; nothing to rerun"
            else:
                shutil.move(p, os.path.join(out, f"record_attempt{n}.json"))
                summary.append(_attempt(r, n))
    summary, err = comm.bcast((summary, err), root=0)
    if err:
        raise RuntimeError(err)
    return summary


def _attempt(r, n):
    return {"attempt": n, "file": f"record_attempt{n}.json", "status": r.get("status"), "error": r.get("error"),
            "start": r.get("start"), "carrier_iterations": (r.get("carrier") or {}).get("iterations"),
            "hook_iterations": len(r.get("history", [])), "dr2_history": [h.get("dr2") for h in r.get("history", [])],
            "xcsb_commit": (r.get("provenance") or {}).get("xcsb_commit")}


def run_dir(root, case_name, mode, name):
    safe = name.replace(":", "_").replace("@", "__").replace(",", "_").replace("=", "")
    return os.path.join(root, f"{case_name}__{mode}__{safe}")


def _load(root, case_name, mode, name):
    p = os.path.join(run_dir(root, case_name, mode, name), "record.json")
    if not os.path.exists(p):
        return {"status": "missing", "path": p}
    return json.load(open(p))


def _cmp(nat, a, b):
    """Native record a vs hooked record b. A global spin flip (the mirror ferromagnet, m -> -m) is the same state: then
    the hooked spin blocks of the eigenvalues (QE lists the k points of spin up, then spin down) are swapped."""
    ea, eb = a["energies_ry"]["etot"], b["energies_ry"]["etot"]
    eiga, eigb = np.asarray(a["eigenvalues_ry"]), np.asarray(b["eigenvalues_ry"])
    ma, mb = a["magnetization"], b["magnetization"]
    flipped = bool(ma["total"] * mb["total"] < 0 and abs(ma["total"]) > 1e-3)
    if flipped:
        half = eigb.shape[1] // 2
        eigb = np.concatenate([eigb[:, half:], eigb[:, :half]], axis=1)
    # occupied and partly occupied bands only: QE converges empty bands loosely (diago_full_acc = .false.)
    case = a["case"]
    if case["occupations"] == "smearing":
        occ = eiga <= a["fermi_ry"] + 3.0 * case["degauss"]
    else:
        st = case["structure"]
        nocc = sum(VALENCE[st["elements"][lab]] for lab in st["labels"]) // 2        # fixed occupations, nspin 1
        occ = np.zeros_like(eiga, dtype=bool)
        occ[:nocc] = True
    out = {"etot_native": ea, "etot_hooked": eb, "d_etot_per_atom": abs(ea - eb) / nat,
           "d_eig_max": float(np.max(np.abs(eiga - eigb)[occ])) if eiga.shape == eigb.shape else None,
           "d_eig_max_all_bands": float(np.max(np.abs(eiga - eigb))) if eiga.shape == eigb.shape else None,
           "mag_native": ma, "mag_hooked": mb, "spin_flipped": flipped,
           "d_mag_total": abs(abs(ma["total"]) - abs(mb["total"])), "d_mag_abs": abs(ma["absolute"] - mb["absolute"]),
           "iterations_hooked": b.get("iterations")}
    out["pass"] = bool(out["d_etot_per_atom"] <= TOL_V3_E and out["d_eig_max"] is not None
                       and out["d_eig_max"] <= TOL_V3_EIG and out["d_mag_total"] <= TOL_V3_MAG
                       and out["d_mag_abs"] <= TOL_V3_MAG)
    return out


def _host_check(a):
    """host E_xc on the native density vs QE's etxc, after removing QE's meta-GGA threshold points (qe_meta_threshold)."""
    etxc = a["energies_ry"]["etxc"]
    thr = a.get("qe_meta_threshold", {"npoints": 0, "host_exc_ry_at_points": 0.0})
    d = a["host_exc_ry_on_native_density"] - etxc
    resid = d - thr["host_exc_ry_at_points"]
    return {"qe_etxc": etxc, "host_minus_qe": d, "qe_threshold_points": thr["npoints"],
            "host_exc_at_threshold_points": thr["host_exc_ry_at_points"], "residual": resid,
            "residual_rel": abs(resid / etxc), "pass": bool(abs(resid / etxc) <= TOL_V3_EXC)}


V3_SCF = (("Si", "PBE"), ("Fe", "PBE"), ("Si-lowsym", "TPSS"), ("Fe-offset", "TPSS"))     # pass criteria
V3_DIAG = (("Si", "TPSS"), ("Fe", "TPSS"))      # diagnostics: QE's thresholds change its TPSS at symmetric grid points


def compare_v3(root):
    res = {"tolerances": {"etot_ry_per_atom": TOL_V3_E, "eig_ry": TOL_V3_EIG, "mag_muB": TOL_V3_MAG,
                          "exc_rel": TOL_V3_EXC}, "cases": {}, "diagnostics": {}}
    for group, items in (("cases", V3_SCF), ("diagnostics", V3_DIAG)):
        for cname, xc in items:
            nat = CASES[cname]().structure.nat
            a, b = _load(root, cname, "native", xc), _load(root, cname, "hooked", xc)
            key = f"{cname}/{xc}"
            if a.get("status") != "ok" or b.get("status") != "ok":
                res[group][key] = {"pass": False, "native": a.get("status"), "hooked": b.get("status"),
                                   "error": a.get("error") or b.get("error")}
                continue
            r = _cmp(nat, a, b)
            r["host_check"] = _host_check(a)
            r["pass"] = bool(r["pass"] and r["host_check"]["pass"])
            if group == "diagnostics":
                r["note"] = ("not a pass criterion: QE's native meta-GGA sets e_xc = 0 at grid points with |grad rho|^2 <= "
                             "1e-24 (centres of symmetry); the host check removes exactly those points")
            res[group][key] = r
    for xc in ("PBE",):
        a = _load(root, "Si-lowsym", "native", xc)
        key = f"Si-lowsym/{xc}/host_exc"
        res["cases"][key] = _host_check(a) if a.get("status") == "ok" else {"pass": False, "native": a.get("status")}
    res["pass"] = all(v["pass"] for v in res["cases"].values())
    res["diagnostics_host_check_pass"] = all(v.get("host_check", {}).get("pass", False) for v in res["diagnostics"].values())
    return res


def compare_v4(root):
    res = {"tolerance_ry": TOL_V4_E, "cases": {}}
    for xc in ("PBE", "r2SCAN"):
        a, b = _load(root, "Si", "hooked", xc), _load(root, "Si-nspin2-m0", "hooked", xc)
        if a.get("status") != "ok" or b.get("status") != "ok":
            res["cases"][xc] = {"pass": False, "nspin1": a.get("status"), "nspin2": b.get("status")}
            continue
        d = abs(a["energies_ry"]["etot"] - b["energies_ry"]["etot"])
        res["cases"][xc] = {"etot_nspin1": a["energies_ry"]["etot"], "etot_nspin2": b["energies_ry"]["etot"],
                            "d_etot": d, "mag_abs_nspin2": b["magnetization"]["absolute"], "pass": bool(d <= TOL_V4_E)}
    res["pass"] = all(v["pass"] for v in res["cases"].values())
    return res


def compare_v6(root):
    res = {"cases": {}}
    for p in sorted(glob.glob(os.path.join(root, "*__hooked__*", "record.json"))):
        r = json.load(open(p))
        key = os.path.basename(os.path.dirname(p))
        if r.get("status") != "ok":
            res["cases"][key] = {"status": r.get("status"), "error": r.get("error")}
            continue
        h = r["history"]
        res["cases"][key] = {"status": "ok", "iterations": r["iterations"], "etot_ry": r["energies_ry"]["etot"],
                             "magnetization": r["magnetization"], "seconds": r["seconds"],
                             "t_hook_mean_s": float(np.mean([x["t_hook_s"] for x in h])),
                             "t_iter_mean_s": float(np.mean([x["t_iter_s"] for x in h])),
                             "min_rho": r.get("density_stats", {}).get("min_rho"),
                             "frac_tau_below_tauW": r.get("density_stats", {}).get("frac_points_tau_below_tauW")}
    res["pass"] = bool(res["cases"]) and all(v["status"] == "ok" for v in res["cases"].values())
    return res


# ---- V5: one MPI process group, no QE -----------------------------------------------------------------------------------
def v5(out, names=("model:test", "r2SCAN", "SCH-SI-A:v3.1")):
    """Bitwise equality of the host step: all ranks vs the root alone (SerialComm), on a synthetic polarized density
    with several chunks."""
    from mpi4py import MPI

    from .functional import get
    from .host.evaluate import evaluate_on_grid
    from .parallel.mpi import SerialComm
    from .testing import BandLimitedDensity, cells, grid_for, tau_above_tw
    comm = MPI.COMM_WORLD
    g = grid_for(cells()["triclinic"], 140.0)
    rho = tau = None
    if comm.Get_rank() == 0:
        rho = BandLimitedDensity(g, nterms=12, c=(0.25, 0.12), amp=0.008, seed=9).rho()
        tau = tau_above_tw(g, rho)
    res = {"ranks": comm.Get_size(), "npts": g.npts, "chunk": 4096, "cases": {}}
    for name in names:
        f = get(name)
        t0 = time.time()
        par = evaluate_on_grid(f, g if comm.Get_rank() == 0 else None, rho, tau, comm, chunk=4096)
        tp = time.time() - t0
        if comm.Get_rank() == 0:
            ser = evaluate_on_grid(f, g, rho, tau, SerialComm(), chunk=4096)
            same = (par.e_xc == ser.e_xc and np.array_equal(par.v, ser.v) and np.array_equal(par.v_tau, ser.v_tau))
            res["cases"][name] = {"bitwise_equal": bool(same), "e_xc": par.e_xc, "d_e_xc": par.e_xc - ser.e_xc,
                                  "max_dv": float(np.max(np.abs(par.v - ser.v))), "t_parallel_s": tp}
        comm.Barrier()
    if comm.Get_rank() == 0:
        res["pass"] = all(v["bitwise_equal"] for v in res["cases"].values())
        os.makedirs(out, exist_ok=True)
        json.dump(res, open(os.path.join(out, f"V5_np{comm.Get_size()}.json"), "w"), indent=1, default=float)
    return res
