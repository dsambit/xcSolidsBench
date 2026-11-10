"""Node verification tiers V3-V6 (DESIGN.md Sec. 7). Each writes a JSON summary with its pass/fail numbers.

    V3  hooks reproduce QE native: libxc PBE / TPSS through the hooks vs QE's input_dft 'PBE' / 'TPSS', on bulk Si
        (nspin 1, fixed occupations) and FM bcc Fe (nspin 2, cold smearing); plus the host's E_xc on the native
        ground-state density vs QE's etxc (also on a strained, displaced, low-symmetry Si cell that pins the grid order).
    V4  nspin = 2 at zero magnetization (Si, tot_magnetization = 0) equals nspin = 1.
    V5  the host step is bitwise independent of the rank count (serial evaluation on the root vs all ranks).
    V6  smoke SCFs through the hooks, cold starts: Si and FM bcc Fe with PBE, r2SCAN, SCH-SI-A:v3.1, SCH-SI-B:v3.1.
"""
from __future__ import annotations

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
TOL_V4_E = 1e-8        # Ry


def case_si(**kw):
    return Case(diamond("Si", A_SI), ecutwfc=kw.pop("ecutwfc", 30.0), kmesh=(4, 4, 4), occupations="fixed", **kw)


def case_si_lowsym():
    s = diamond("Si", A_SI)
    strain = np.array([[1.02, 0.01, 0.0], [0.0, 0.99, 0.015], [0.0, 0.0, 1.005]])
    frac = s.frac.copy()
    frac[1] += [0.013, -0.007, 0.004]
    low = Structure("Si-lowsym", s.lattice @ strain, s.labels, frac, s.elements)
    return Case(low, ecutwfc=30.0, kmesh=(3, 3, 3), occupations="fixed", nosym=True)


def case_fe(**kw):
    return Case(bcc("Fe", A_FE), ecutwfc=kw.pop("ecutwfc", 60.0), kmesh=(8, 8, 8), occupations="smearing", degauss=0.02,
                nspin=2, starting_magnetization={"Fe": 0.5}, **kw)


def _comm():
    from mpi4py import MPI
    return MPI.COMM_WORLD


def _dump(out, name, obj, comm):
    if comm.Get_rank() == 0:
        os.makedirs(out, exist_ok=True)
        json.dump(obj, open(os.path.join(out, name), "w"), indent=1, default=float)


def _guard(res, key, fn):
    """Run fn() for one case; a failure is recorded (status, error) and the tier continues with the next case."""
    try:
        return fn()
    except Exception as exc:
        res["cases"][key] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "pass": False}
        return None


def _cmp(nat, a, b):
    ea, eb = a["energies_ry"]["etot"], b["energies_ry"]["etot"]
    eiga, eigb = np.asarray(a["eigenvalues_ry"]), np.asarray(b["eigenvalues_ry"])
    out = {"etot_a": ea, "etot_b": eb, "d_etot_per_atom": abs(ea - eb) / nat,
           "d_eig_max": float(np.max(np.abs(eiga - eigb))) if eiga.shape == eigb.shape else None,
           "d_mag_total": abs(a["magnetization"]["total"] - b["magnetization"]["total"]),
           "d_mag_abs": abs(a["magnetization"]["absolute"] - b["magnetization"]["absolute"])}
    out["pass"] = (out["d_etot_per_atom"] <= TOL_V3_E and out["d_eig_max"] is not None and out["d_eig_max"] <= TOL_V3_EIG
                   and out["d_mag_total"] <= TOL_V3_MAG and out["d_mag_abs"] <= TOL_V3_MAG)
    return out


def v3(out):
    from .scf.driver import run_hooked, run_native
    comm = _comm()
    res = {"tolerances": {"etot_ry_per_atom": TOL_V3_E, "eig_ry": TOL_V3_EIG, "mag_muB": TOL_V3_MAG}, "cases": {}}
    for cname, mk in (("Si", case_si), ("Fe", case_fe)):
        c = mk()
        for qe_name, xc in (("PBE", "PBE"), ("TPSS", "TPSS")):
            nat = c.structure.nat
            key = f"{cname}/{xc}"
            nat_rec = _guard(res, key, lambda: run_native(c, qe_name, os.path.join(out, f"{cname}_{qe_name}_native"),
                                                           comm, host_functional=xc))
            hk_rec = nat_rec and _guard(res, key, lambda: run_hooked(c, xc, os.path.join(out, f"{cname}_{xc}_hooked"), comm))
            if hk_rec is None:
                _dump(out, "V3.json", res, comm)
                continue
            r = _cmp(nat, nat_rec, hk_rec)
            if comm.Get_rank() == 0:
                etxc = nat_rec["energies_ry"]["etxc"]
                r["host_exc_minus_qe_etxc_ry"] = nat_rec["host_exc_ry_on_native_density"] - etxc
                r["host_exc_rel"] = abs(r["host_exc_minus_qe_etxc_ry"]) / abs(etxc)
                r["iterations_hooked"] = hk_rec["iterations"]
            res["cases"][key] = r
            _dump(out, "V3.json", res, comm)
    c = case_si_lowsym()
    for qe_name in ("PBE", "TPSS"):
        key = f"Si-lowsym/{qe_name}/host_exc"
        rec = _guard(res, key, lambda: run_native(c, qe_name, os.path.join(out, f"Si-lowsym_{qe_name}_native"), comm,
                                                  host_functional=qe_name))
        if rec is not None and comm.Get_rank() == 0:
            etxc = rec["energies_ry"]["etxc"]
            d = rec["host_exc_ry_on_native_density"] - etxc
            res["cases"][key] = {"qe_etxc": etxc, "host_minus_qe": d, "rel": abs(d / etxc), "pass": abs(d / etxc) < 1e-6}
    if comm.Get_rank() == 0:
        res["pass"] = all(v.get("pass", True) for v in res["cases"].values())
    _dump(out, "V3.json", res, comm)
    return res


def v4(out):
    from .scf.driver import run_hooked
    comm = _comm()
    res = {"tolerance_ry": TOL_V4_E, "cases": {}}
    for xc in ("PBE", "r2SCAN"):
        r1 = _guard(res, xc, lambda: run_hooked(case_si(), xc, os.path.join(out, f"Si_{xc}_nspin1"), comm))
        r2 = r1 and _guard(res, xc, lambda: run_hooked(case_si(nspin=2, tot_magnetization=0.0), xc,
                                                         os.path.join(out, f"Si_{xc}_nspin2_m0"), comm))
        if r2 is None:
            continue
        d = abs(r1["energies_ry"]["etot"] - r2["energies_ry"]["etot"])
        res["cases"][xc] = {"etot_nspin1": r1["energies_ry"]["etot"], "etot_nspin2": r2["energies_ry"]["etot"],
                            "d_etot": d, "mag_abs_nspin2": r2["magnetization"]["absolute"], "pass": d <= TOL_V4_E}
    res["pass"] = all(v["pass"] for v in res["cases"].values())
    _dump(out, "V4.json", res, comm)
    return res


def v5(out, names=("model:test", "r2SCAN", "SCH-SI-A:v3.1")):
    """Bitwise equality of the host step: all ranks vs the root alone (SerialComm), on a synthetic polarized density
    with several chunks."""
    from .functional import get
    from .host.evaluate import evaluate_on_grid
    from .parallel.mpi import SerialComm
    from .testing import BandLimitedDensity, cells, grid_for, tau_above_tw
    comm = _comm()
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
    _dump(out, f"V5_np{comm.Get_size()}.json", res, comm)
    return res


def v6(out, names=("PBE", "r2SCAN", "SCH-SI-A:v3.1", "SCH-SI-B:v3.1"), systems=("Si", "Fe")):
    from .scf.driver import run_hooked
    comm = _comm()
    res = {"ranks": comm.Get_size(), "cases": {}}
    mk = {"Si": case_si, "Fe": case_fe}
    for sysname in systems:
        for xc in names:
            key = f"{sysname}/{xc}"
            try:
                r = run_hooked(mk[sysname](), xc, os.path.join(out, f"{sysname}_{xc.replace(':', '_')}"), comm)
                res["cases"][key] = {"status": r["status"], "iterations": r["iterations"],
                                     "etot_ry": r["energies_ry"]["etot"], "magnetization": r["magnetization"],
                                     "seconds": r["seconds"],
                                     "t_hook_mean_s": float(np.mean([h["t_hook_s"] for h in r["history"]])),
                                     "t_iter_mean_s": float(np.mean([h["t_iter_s"] for h in r["history"]])),
                                     "min_rho": r.get("density_stats", {}).get("min_rho")}
            except Exception as exc:
                res["cases"][key] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            _dump(out, f"V6_np{comm.Get_size()}.json", res, comm)
    res["pass"] = all(v["status"] == "ok" for v in res["cases"].values())
    _dump(out, f"V6_np{comm.Get_size()}.json", res, comm)
    return res
