"""Band gap of rock-salt MgO (user 2026-11-11): PBE, r2SCAN, SCH-SI-A:v3.1, SCH-SI-B:v3.1 through the hooks.

Cell: primitive fcc, 2 atoms, static-lattice a (refdata solids/lattice_constants_static_mgo_2026-11-11: 4.207 measured - 0.018
PBE zero-point expansion = 4.189 A), nspin 1, fixed occupations (16 valence electrons with SG15 Mg 10, O 6: 8 occupied bands),
NBND = 12 with diago_full_acc (empty bands converged like the occupied ones), conv_thr bench, cold starts (carrier TPSS).

Cases 'MgO-<ecutwfc>-k<n>': n^3 Gamma-centred unshifted k mesh (Gamma, X and L belong to every even mesh), dual 4.

Stage P: PBE on ecutwfc in SCAN_ECUT at k 8^3; E_c^PBE = the smallest scan cutoff from which every successive pair has
|Delta E_tot| <= 1 meV/atom and |Delta gap_Gamma| <= 0.02 eV (the F2/F3 energy criterion carried over to an insulator,
plus the family's gap criterion). Stage G: all four functionals on ecutwfc = r E_c^PBE, r in RUNGS, times k in KMESHES
(the full 3 x 3 grid). Selected settings: the smallest r with |gap(r) - gap(r_max)| <= GAP_TOL at k_max for all four, and
the smallest k with |gap(k) - gap(k_max)| <= GAP_TOL at r_max for all four (r_max or k_max itself: not shown converged).

The gap: GKS eigenvalue gap eps_CBM - eps_VBM of the converged SCF (the tau potential is non-multiplicative): the direct gaps
at Gamma, X and L, and the minimum gap over the SCF mesh (min over k of the lowest empty band - max over k of the highest
occupied band).
"""
from __future__ import annotations

import itertools
import json
import os

import numpy as np

from ..bench.structures import BOHR_PER_ANGSTROM, rocksalt
from ..host.qe import Case

A_STATIC_ANGSTROM = 4.189
A_SOURCE = "refdata solids/lattice_constants_static_mgo_2026-11-11 (4.207 measured - 0.018 PBE zero-point expansion)"
N_OCC = 8                           # (10 + 6) / 2: SG15 1.2 z_valence Mg 10, O 6 (pseudo/MANIFEST.json)
NBND = 12
FUNCTIONALS = ("PBE", "r2SCAN", "SCH-SI-A:v3.1", "SCH-SI-B:v3.1")
REFERENCE = ("PBE", "r2SCAN")       # computed once at this protocol, then banked in the archive
SCAN_ECUT = (40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 140.0)
SCAN_K = 8
RUNGS = (1.0, 1.5, 2.0)
KMESHES = (6, 8, 10)
CRIT_DE_MEV = 1.0                   # per atom
GAP_TOL = 0.02                      # eV
RY_EV = 13.605693122994
RY_MEV = 1000 * RY_EV
NPOOL = 1                           # one pool: QE's k list (eigenvalues) is then complete on every rank


def case_name(ecut, k, a=None):
    """'MgO-<ecut>-k<n>' at the static lattice (the PBE cutoff scan), 'MgO-<ecut>-k<n>-a<a>' at lattice constant a (Angstrom)."""
    return f"MgO-{ecut:g}-k{int(k)}" + ("" if a is None else f"-a{a:.6f}")


def parse_case(name):
    parts = name.split("-")
    if parts[0] != "MgO" or len(parts) not in (3, 4) or not parts[2].startswith("k"):
        raise KeyError(f"not an MgO case name: {name!r}")
    a = float(parts[3][1:]) if len(parts) == 4 else A_STATIC_ANGSTROM
    return float(parts[1]), int(parts[2][1:]), a


def get_case(name) -> Case:
    ecut, k, a = parse_case(name)
    return Case(rocksalt("Mg", "O", a * BOHR_PER_ANGSTROM), ecutwfc=ecut, kmesh=(k, k, k), kshift=(0, 0, 0),
                occupations="fixed", nspin=1, nbnd=NBND, diago_full_acc=True, conv_class="bench")


# ---- gaps from a record ---------------------------------------------------------------------------------------------------------
def cubic_ops():
    ops = []
    for p in itertools.permutations(range(3)):
        for sg in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            for i in range(3):
                m[i, p[i]] = sg[i]
            ops.append(m)
    return ops


def _recip(lattice):
    return 2.0 * np.pi * np.linalg.inv(np.asarray(lattice, dtype=float)).T          # rows b_i


def special_points(lattice):
    """Gamma, X, L of the fcc lattice in crystal coordinates of the given primitive cell (a = conventional constant)."""
    lat = np.asarray(lattice, dtype=float)
    a = np.sqrt(2.0) * np.linalg.norm(lat[0])
    cart = {"Gamma": np.zeros(3), "X": np.array([1.0, 0.0, 0.0]), "L": np.array([0.5, 0.5, 0.5])}
    return {k: (2.0 * np.pi / a) * v @ lat.T / (2.0 * np.pi) for k, v in cart.items()}          # frac_i = k . a_i / 2 pi


def star_frac(lattice, target):
    """The images of a k point (crystal coordinates) under the cubic point group, (48, 3) crystal coordinates."""
    b = _recip(lattice)
    tc = np.asarray(target, dtype=float) @ b
    return np.array([np.linalg.solve(b.T, op @ tc) for op in cubic_ops()])


def equivalent(lattice, kf, target):
    """kf (..., 3), target in crystal coordinates: equal up to a cubic point operation and a reciprocal-lattice vector."""
    d = np.asarray(kf, dtype=float)[..., None, :] - star_frac(lattice, target)
    return np.any(np.all(np.abs(d - np.round(d)) <= 1e-6, axis=-1), axis=-1)


def gaps(rec, lattice):
    """Direct gaps at Gamma, X, L and the minimum gap over the mesh (eV) from a record's eigenvalues (nbnd, nks) and k list."""
    et = np.asarray(rec["eigenvalues_ry"]) * RY_EV
    kf = np.asarray(rec["kpoints_frac"])
    if et.shape[0] <= N_OCC or et.shape[1] != kf.shape[0]:
        raise ValueError(f"eigenvalues {et.shape} vs {N_OCC} occupied bands and {kf.shape[0]} k points")
    vb, cb = et[N_OCC - 1], et[N_OCC]
    out = {"vbm_ev": float(vb.max()), "cbm_ev": float(cb.min()), "min_gap_ev": float(cb.min() - vb.max()),
           "vbm_k": kf[int(vb.argmax())].tolist(), "cbm_k": kf[int(cb.argmin())].tolist()}
    for name, t in special_points(lattice).items():
        idx = np.flatnonzero(equivalent(lattice, kf, t))
        if idx.size == 0:
            raise ValueError(f"{name} is not in the SCF k list")
        out[f"gap_{name}_ev"] = float(cb[idx[0]] - vb[idx[0]])
    return out


# ---- records ----------------------------------------------------------------------------------------------------------------------
def scf_root(run):
    return os.path.join(run, "scf")


def record(run, case, functional, tag=""):
    from ..tiers import run_dir
    p = os.path.join(run_dir(scf_root(run), case, "hooked", functional) + (f"__{tag}" if tag else ""), "record.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _ok(r):
    return r is not None and r.get("status") == "ok"


def _row(r, lattice):
    return dict(gaps(r, lattice), etot_ry=r["energies_ry"]["etot"], iterations=r["iterations"],
                carrier_iterations=r["carrier"]["iterations"], seconds=r.get("seconds"))


def _lattice():
    return get_case(case_name(80, 8)).structure.lattice


# ---- stage P: E_c^PBE ---------------------------------------------------------------------------------------------------------------
def stage_p_tasks():
    return [{"id": f"P__{case_name(e, SCAN_K)}", "case": case_name(e, SCAN_K), "functional": "PBE", "width": 1, "stage": "P",
             "tag": ""} for e in SCAN_ECUT]


def scan_table(run):
    lat = _lattice()
    rows = []
    for e in SCAN_ECUT:
        r = record(run, case_name(e, SCAN_K), "PBE")
        rows.append(dict(ecutwfc=e, **_row(r, lat)) if _ok(r) else {"ecutwfc": e, "status": None if r is None else r.get("status")})
    return rows


def e_c_pbe(run):
    """E_c^PBE, or None while the scan is incomplete; raises if a complete scan never meets the criteria."""
    rows = scan_table(run)
    if any("etot_ry" not in r for r in rows):
        return None
    ok = [abs(b["etot_ry"] - a["etot_ry"]) * RY_MEV / 2 <= CRIT_DE_MEV and abs(b["gap_Gamma_ev"] - a["gap_Gamma_ev"]) <= GAP_TOL
          for a, b in zip(rows[:-1], rows[1:])]
    n = next((i for i in range(len(ok)) if all(ok[i:])), None)
    if n is None:
        raise RuntimeError(f"MgO: the PBE scan {SCAN_ECUT} never meets the criteria between successive cutoffs")
    return rows[n]["ecutwfc"]


# ---- stage E: the r2SCAN lattice constant (user 2026-11-11: r2SCAN lattices for the gap tests) -----------------------------------
EOS_DELTA = tuple(np.linspace(-0.03, 0.03, 9))   # a = a_static (1 + delta): 9 volumes over about +-3 % in a
EOS_RUNG, EOS_K = 2.0, 8                          # the EOS at the top cutoff rung (RUNGS[-1] E_c^PBE) and 8^3 k


def stage_e_tasks(ec):
    e = round(EOS_RUNG * ec, 6)
    return [{"id": f"E__{case_name(e, EOS_K, A_STATIC_ANGSTROM * (1 + d))}__r2SCAN", "case": case_name(e, EOS_K, A_STATIC_ANGSTROM * (1 + d)),
             "functional": "r2SCAN", "width": 1, "stage": "E", "tag": ""} for d in EOS_DELTA]


def eos(run, ec):
    """The r2SCAN BM3 fit over stage E, or None while incomplete; raises (EOSError) on a failed fit."""
    from ..bench.eos import bm3_fit
    V, E, rows = [], [], []
    for t in stage_e_tasks(ec):
        r = record(run, t["case"], "r2SCAN")
        if not _ok(r):
            return None
        _, _, a = parse_case(t["case"])
        v = abs(float(np.linalg.det(get_case(t["case"]).structure.lattice)))
        V.append(v), E.append(r["energies_ry"]["etot"]), rows.append({"a_angstrom": a, "V_bohr3": v, "etot_ry": E[-1]})
    fit = bm3_fit(V, E)
    a0 = (4.0 * fit["V0_bohr3"]) ** (1.0 / 3.0) / BOHR_PER_ANGSTROM            # fcc primitive: V = a^3 / 4
    return dict(fit, a0_angstrom=float(a0), points=rows, ecutwfc=round(EOS_RUNG * ec, 6), k=EOS_K)


# ---- stage G: the (cutoff, k) grid at the r2SCAN a0 -------------------------------------------------------------------------------
def stage_g_tasks(ec, a0):
    return [{"id": f"G__{case_name(round(r * ec, 6), k, a0)}__{f}", "case": case_name(round(r * ec, 6), k, a0), "functional": f,
             "width": 1, "stage": "G", "tag": "", "rung": r, "k": k} for r in RUNGS for k in KMESHES for f in FUNCTIONALS]


# cost model (scheduling): QE s per SCF iteration on a quarter node, from the Fe 80 Ry 8^3 k timing (3.7 s, nspin 2, 29
# irreducible k, cell 79 bohr^3, 12 bands) scaled by the cell volume (124 bohr^3), nspin 1 (x 0.5), the irreducible k count
# and (ecutwfc/80)^1.5; hook SCH-SI 5.3 s per round of 16384-point chunks (8 ranks), libxc 0.3 s; iterations: carrier 15,
# hook 15 (libxc) or 20 (SCH-SI).
NK_IBZ = {6: 16, 8: 29, 10: 47}


def fft_points(ecut, dual=4):
    a = A_STATIC_ANGSTROM * BOHR_PER_ANGSTROM
    n = 2 * int(np.sqrt(dual * ecut) * (a / np.sqrt(2.0)) / (2.0 * np.pi)) + 1
    while True:
        m = n
        for p in (2, 3, 5, 7):
            while m % p == 0:
                m //= p
        if m == 1:
            return n ** 3
        n += 1


def estimate_s(task):
    ecut, k, _ = parse_case(task["case"])
    qe = 3.7 * (124.0 / 79.4) * 0.5 * (NK_IBZ[k] / 29.0) * (ecut / 80.0) ** 1.5
    sch = task["functional"].startswith("SCH-SI")
    chunks = -(-fft_points(ecut) // 16384)
    rounds = -(-chunks // 8)
    hook = rounds * (5.3 if sch else 0.3)
    return float(15 * qe + (20 if sch else 15) * (qe + hook))


def tasks(run):
    """Stage P; then (scan complete; EC.json written once) stage E, the r2SCAN EOS; then (fit done; EOS.json written once, or
    EOS_STOP.txt on a failed fit and no stage G) stage G at a0."""
    out = [dict(t, est_s=estimate_s(t)) for t in stage_p_tasks()]
    p = os.path.join(run, "EC.json")
    if os.path.exists(p):
        ec = json.load(open(p))["ec_pbe"]
    else:
        ec = e_c_pbe(run)
        if ec is None:
            return out
        json.dump({"ec_pbe": ec, "scan": scan_table(run), "criteria": {"de_mev_per_atom": CRIT_DE_MEV, "gap_ev": GAP_TOL}},
                  open(p, "w"), indent=1)
    out += [dict(t, est_s=estimate_s(t)) for t in stage_e_tasks(ec)]
    pe = os.path.join(run, "EOS.json")
    if not os.path.exists(pe):
        if os.path.exists(os.path.join(run, "EOS_STOP.txt")):
            return out
        from ..bench.eos import EOSError
        try:
            fit = eos(run, ec)
        except EOSError as exc:
            open(os.path.join(run, "EOS_STOP.txt"), "w").write(f"{exc}\n")
            return out
        if fit is None:
            return out
        json.dump(fit, open(pe, "w"), indent=1)
    a0 = round(json.load(open(pe))["a0_angstrom"], 6)
    return out + [dict(t, est_s=estimate_s(t)) for t in stage_g_tasks(ec, a0)]


def done(run, task):
    return _ok(record(run, task["case"], task["functional"], task.get("tag", "")))


def failed(run, task):
    r = record(run, task["case"], task["functional"], task.get("tag", ""))
    return r is not None and r.get("status") == "failed"


def command(run, task):
    return ["scf", "--case", task["case"], "--mode", "hooked", "--name", task["functional"], "--root", scf_root(run),
            "--npool", str(NPOOL)]


# ---- tables ----------------------------------------------------------------------------------------------------------------------
def _select(grid, axis_vals, fixed_key, key_of):
    """The smallest axis value whose gap is within GAP_TOL of the largest one's for every functional (None if not all
    present); 'converged' is False when only the largest value itself qualifies."""
    for v in axis_vals:
        if all(grid.get(key_of(f, v)) is not None and grid.get(key_of(f, axis_vals[-1])) is not None and
               abs(grid[key_of(f, v)]["gap_Gamma_ev"] - grid[key_of(f, axis_vals[-1])]["gap_Gamma_ev"]) <= GAP_TOL
               for f in FUNCTIONALS):
            return v, v != axis_vals[-1]
    return None, False


def table(run):
    lat = _lattice()
    pe = os.path.join(run, "EC.json")
    ec = json.load(open(pe))["ec_pbe"] if os.path.exists(pe) else None
    pf = os.path.join(run, "EOS.json")
    fit = json.load(open(pf)) if os.path.exists(pf) else None
    a0 = None if fit is None else round(fit["a0_angstrom"], 6)
    out = {"a_scan_angstrom": A_STATIC_ANGSTROM, "a_scan_source": A_SOURCE, "a_angstrom": a0,
           "a_source": "r2SCAN BM3 a0 of this run (EOS.json; user 2026-11-11)", "eos": fit, "ec_pbe": ec, "scan": scan_table(run),
           "grid": {}, "selected": None}
    if ec is None or a0 is None:
        return out
    grid = {}
    for r in RUNGS:
        for k in KMESHES:
            for f in FUNCTIONALS:
                rec = record(run, case_name(round(r * ec, 6), k, a0), f)
                grid[(f, r, k)] = _row(rec, lat) if _ok(rec) else None
    out["grid"] = {f"{f}|r{r:g}|k{k}": v for (f, r, k), v in grid.items()}
    r_sel, r_conv = _select(grid, RUNGS, None, lambda f, v: (f, v, KMESHES[-1]))
    k_sel, k_conv = _select(grid, KMESHES, None, lambda f, v: (f, RUNGS[-1], v))
    if r_sel is not None and k_sel is not None and all(grid.get((f, r_sel, k_sel)) for f in FUNCTIONALS):
        sel = {f: grid[(f, r_sel, k_sel)] for f in FUNCTIONALS}
        pbe = sel["PBE"]
        out["selected"] = {
            "r": r_sel, "ecutwfc": round(r_sel * ec, 6), "k": k_sel, "cutoff_converged": r_conv, "k_converged": k_conv,
            "per_functional": {f: {**v, "minus_pbe_gap_Gamma_ev": v["gap_Gamma_ev"] - pbe["gap_Gamma_ev"],
                                   "minus_pbe_min_gap_ev": v["min_gap_ev"] - pbe["min_gap_ev"]} for f, v in sel.items()}}
    return out


def bank_reference(run):
    """The archive-ready result of the reference functionals (PBE, r2SCAN) at this protocol (written to the archive by hand,
    once; write-once)."""
    t = table(run)
    if t["selected"] is None:
        raise RuntimeError("no selected settings yet")
    return {"what": "MgO GKS band gaps of the reference functionals through the xcSolidsBench hooks",
            "protocol": {"cell": "rock salt, primitive, 2 atoms", "a_angstrom": t["a_angstrom"], "a_source": t["a_source"],
                         "pseudopotentials": "SG15 ONCV 1.2 (Mg, O), no NLCC", "nspin": 1, "occupations": "fixed",
                         "nbnd": NBND, "diago_full_acc": True, "dual": 4, "ec_pbe": t["ec_pbe"],
                         "ecutwfc": t["selected"]["ecutwfc"], "kmesh": t["selected"]["k"], "kshift": [0, 0, 0],
                         "gap": "GKS eigenvalue gap; direct at Gamma, X, L; minimum over the SCF mesh"},
            "results": {f: t["selected"]["per_functional"][f] for f in REFERENCE},
            "grid": {k: v for k, v in t["grid"].items() if k.split("|")[0] in REFERENCE}}


def bank_eos(run):
    """The archive-ready r2SCAN EOS result (a reference-functional result: banked once, write-once)."""
    p = os.path.join(run, "EOS.json")
    if not os.path.exists(p):
        raise RuntimeError("no EOS.json")
    fit = json.load(open(p))
    return {"what": "r2SCAN lattice constant of rock-salt MgO through the xcSolidsBench hooks (BM3 fit; user 2026-11-11)",
            "protocol": {"pseudopotentials": "SG15 ONCV 1.2 (Mg, O), no NLCC", "nspin": 1, "occupations": "fixed", "dual": 4,
                         "ecutwfc": fit["ecutwfc"], "kmesh": fit["k"], "volumes": "a = a_static (1 + delta), delta in [-0.03, 0.03], 9 points",
                         "a_static_angstrom": A_STATIC_ANGSTROM, "fit": "third-order Birch-Murnaghan (xcsb.bench.eos)"},
            "a0_angstrom": fit["a0_angstrom"], "B0_GPa": fit["B0_GPa"], "B0_prime": fit["B0_prime"], "rms_ry": fit["rms_ry"],
            "points": fit["points"]}


def render(t) -> str:
    out = ["# MgO band gap (GKS eigenvalue gaps, eV)", f"a = {t['a_angstrom']} A ({t['a_source']}); the PBE cutoff scan at "
           f"{t['a_scan_angstrom']} A", ""]
    if t.get("eos"):
        e = t["eos"]
        out.append(f"r2SCAN EOS: a0 {e['a0_angstrom']:.5f} A, B0 {e['B0_GPa']:.1f} GPa, B0' {e['B0_prime']:.2f}, rms {e['rms_ry']:.1e} Ry")
    out.append("## PBE cutoff scan (k 8^3): criteria |dE| <= 1 meV/atom, |d gap_Gamma| <= 0.02 eV")
    out.append("| ecutwfc | E (Ry) | gap Gamma | min gap | it |")
    out.append("|---|---|---|---|---|")
    for r in t["scan"]:
        if "etot_ry" in r:
            out.append(f"| {r['ecutwfc']:g} | {r['etot_ry']:.8f} | {r['gap_Gamma_ev']:.4f} | {r['min_gap_ev']:.4f} | "
                       f"{r['carrier_iterations']} + {r['iterations']} |")
        else:
            out.append(f"| {r['ecutwfc']:g} | {r.get('status') or 'missing'} | | | |")
    out.append(f"\nE_c^PBE = {t['ec_pbe']}")
    if t["grid"]:
        out.append("\n## Grid: gap at Gamma / min gap / gap X / gap L (eV), iterations (carrier + hook)")
        out.append("| functional | r | k | gap Gamma | min gap | gap X | gap L | it |")
        out.append("|---|---|---|---|---|---|---|---|")
        for key, v in t["grid"].items():
            f, r, k = key.split("|")
            if v is None:
                out.append(f"| {f} | {r[1:]} | {k[1:]} | missing | | | | |")
            else:
                out.append(f"| {f} | {r[1:]} | {k[1:]} | {v['gap_Gamma_ev']:.4f} | {v['min_gap_ev']:.4f} | {v['gap_X_ev']:.4f} | "
                           f"{v['gap_L_ev']:.4f} | {v['carrier_iterations']} + {v['iterations']} |")
    s = t["selected"]
    if s:
        out.append(f"\n## Selected: ecutwfc {s['ecutwfc']:g} Ry (r = {s['r']:g}, converged: {s['cutoff_converged']}), "
                   f"k {s['k']}^3 (converged: {s['k_converged']})")
        out.append("| functional | gap Gamma | min gap | - PBE (Gamma) | - PBE (min) | it |")
        out.append("|---|---|---|---|---|---|")
        for f, v in s["per_functional"].items():
            out.append(f"| {f} | {v['gap_Gamma_ev']:.4f} | {v['min_gap_ev']:.4f} | {v['minus_pbe_gap_Gamma_ev']:+.4f} | "
                       f"{v['minus_pbe_min_gap_ev']:+.4f} | {v['carrier_iterations']} + {v['iterations']} |")
    return "\n".join(out)
