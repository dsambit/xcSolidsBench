"""Relaxed lattice constants of five rock-salt ionic solids (user 2026-11-13: "test the relaxed lattice constants of ionic
solids using the approach my collaborator has used"): LiF, LiCl, NaF, NaCl, MgO with PBE, r2SCAN, SCH-SI-A:v3.2 and
SCH-SI-B:v3.2 through the hooks (xchole line v3.2), in the collaborator's LC20 protocol. One protocol for every solid and
functional.

Cell: rock salt, primitive fcc, 2 atoms, nspin 1, fixed occupations, nbnd = n_occ + 4 (n_occ from the SG15 1.2 valence charges
of pseudo/MANIFEST.json). ecutwfc = max(80 Ry, the per-element floors), ecutrho = 8 ecutwfc (DUAL), both fixed across the scan.
k mesh: N_i = max(1, ceil(|b_i| / 0.1 A^-1)) with |b_i| the reciprocal-lattice vector lengths (2 pi included; the VASP
k-spacing convention) of the REFERENCE cell, fixed across the scan; Gamma-centred, unshifted (this host's convention, DESIGN 6.0,
and VASP's for an automatic spacing mesh). conv_thr 1e-8 Ry (bench), Broyden beta 0.3, cold starts: every point starts from QE's
cold start and the converged carrier (QE's own TPSS) in the same session, then the hook (the collaborator starts from PBE).

Cases 'LC-<solid>-<step>': a = a_ref (1 + step / 100), step an integer written m3 ... p3 (percent). Reference a_ref: the
zero-point-corrected experimental values of the LC20 set (ionic_lc_reference.json next to this module, with the sources; not
yet an archive entry).

Stages. R / M: the base scan step = -3 ... +3 (7 points) of the reference functionals (PBE, r2SCAN: computed once in this
protocol, then banked) / of the two model sets. RX / MX, the EXTENSION (scan design, not a relaxed check): once every point of a
(solid, functional) scan is done, the unguarded BM3 fit of all its points gives V0; if V0 is not inside the inner two thirds of
the scanned volumes, points are added at the same 1 % spacing on the side of the minimum (as many as put that V0 inside, at
least one, |step| <= MAX_STEP = 8), and the test repeats with all points. The final fit (xcsb.bench.eos.bm3_fit, guarded: V0
inside the inner two thirds, rms <= 2e-5 Ry) uses all points; a scan that reaches |step| = 8 without bracketing, or a failed
fit, is reported as such (no value). The table also gives the plain fit of the base +-3 % points (unguarded, labelled): the
collaborator's protocol like for like, next to the collaborator's own values (a cross-check column).

'LC-phase0': the job's plumbing check only (MgO at a_ref, 40 Ry, 6^3 k; never a result).
"""
from __future__ import annotations

import itertools
import json
import math
import os

import numpy as np

from ..bench.structures import BOHR_PER_ANGSTROM, rocksalt
from ..host.qe import Case

REF = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ionic_lc_reference.json")))
SOLIDS = ("LiF", "LiCl", "NaF", "NaCl", "MgO")
SPECIES = {"LiF": ("Li", "F"), "LiCl": ("Li", "Cl"), "NaF": ("Na", "F"), "NaCl": ("Na", "Cl"), "MgO": ("Mg", "O")}
A_REF = REF["a_ref_angstrom"]                      # conventional lattice constants, Angstrom
REFERENCE = ("PBE", "r2SCAN")                      # computed once in this protocol, then banked in the archive
MODELS = ("SCH-SI-A:v3.2", "SCH-SI-B:v3.2")
FUNCTIONALS = REFERENCE + MODELS
XCHOLE_LINE = "v3.2"                               # the xchole line (functional.xchole.XCHOLE_PINS) the model sets need
DUAL = 8                                           # ecutrho = 8 ecutwfc
KSPACING = 0.1                                     # 1/Angstrom, 2 pi included
NBND_EXTRA = 4
BASE_STEPS = (-3, -2, -1, 0, 1, 2, 3)              # percent of a_ref
MAX_STEP = 8
NPOOL = 8                                          # one rank per k-point pool on a quarter node (as M1; no eigenvalues needed)
PHASE0 = "LC-phase0"
STAGE = {"PBE": "R", "r2SCAN": "R", "SCH-SI-A:v3.2": "M", "SCH-SI-B:v3.2": "M"}
RY_BOHR3_TO_GPA = 14710.507848260711


# ---- protocol -----------------------------------------------------------------------------------------------------------------------
def ecutwfc(solid):
    return float(max([REF["ecutwfc_min_ry"]] + [REF["ecutwfc_floor_ry"][el] for el in SPECIES[solid]]))


def n_occ(solid):
    from ..pseudo import manifest
    z = sum(manifest()["files"][el]["z_valence"] for el in SPECIES[solid])
    if abs(z - round(z)) > 1e-12 or int(round(z)) % 2:
        raise ValueError(f"{solid}: {z} valence electrons: not a closed-shell insulator")
    return int(round(z)) // 2


def structure(solid, step=0):
    return rocksalt(*SPECIES[solid], a_of(solid, step) * BOHR_PER_ANGSTROM)


def a_of(solid, step):
    return A_REF[solid] * (1.0 + step / 100.0)


def volume(solid, step):
    """Primitive-cell volume, bohr^3 (a^3 / 4)."""
    return (a_of(solid, step) * BOHR_PER_ANGSTROM) ** 3 / 4.0


def kmesh(solid):
    """N_i = max(1, ceil(|b_i| / KSPACING)) of the reference cell (|b_i| in 1/Angstrom, 2 pi included)."""
    lat = structure(solid).lattice / BOHR_PER_ANGSTROM
    b = 2.0 * np.pi * np.linalg.inv(lat).T
    return tuple(max(1, math.ceil(float(np.linalg.norm(bi)) / KSPACING)) for bi in b)


def case_name(solid, step):
    return f"LC-{solid}-{'m' if step < 0 else 'p'}{abs(int(step))}"


def parse_case(name):
    parts = name.split("-")
    if len(parts) != 3 or parts[0] != "LC" or parts[1] not in SOLIDS or parts[2][:1] not in ("m", "p") or not parts[2][1:].isdigit():
        raise KeyError(f"not an ionic lattice-constant case name: {name!r}")
    step = int(parts[2][1:]) * (-1 if parts[2][0] == "m" else 1)
    if abs(step) > MAX_STEP or name != case_name(parts[1], step):
        raise KeyError(f"{name!r}: step outside +-{MAX_STEP} % or not in canonical form")
    return parts[1], step


def get_case(name) -> Case:
    if name == PHASE0:
        return Case(structure("MgO"), ecutwfc=40.0, kmesh=(6, 6, 6), kshift=(0, 0, 0), occupations="fixed", nspin=1,
                    nbnd=n_occ("MgO") + NBND_EXTRA, dual=DUAL, conv_class="bench")
    solid, step = parse_case(name)
    return Case(structure(solid, step), ecutwfc=ecutwfc(solid), kmesh=kmesh(solid), kshift=(0, 0, 0), occupations="fixed",
                nspin=1, nbnd=n_occ(solid) + NBND_EXTRA, dual=DUAL, conv_class="bench")


def protocol():
    """Per solid: the settings that follow from the rules above (for the table header, the tests and the report)."""
    return {s: {"species": list(SPECIES[s]), "a_ref_angstrom": A_REF[s], "ecutwfc_ry": ecutwfc(s), "ecutrho_ry": DUAL * ecutwfc(s),
                "kmesh": list(kmesh(s)), "kshift": [0, 0, 0], "nk_irreducible": nk_irreducible(kmesh(s)[0]),
                "n_occ": n_occ(s), "nbnd": n_occ(s) + NBND_EXTRA, "V_ref_bohr3": volume(s, 0)} for s in SOLIDS}


# ---- records ------------------------------------------------------------------------------------------------------------------------
def scf_root(run):
    return os.path.join(run, "scf")


_CACHE = {}


def record(run, case, functional):
    """The fields of a point's record this study uses (None: no record); cached by the file's modification time."""
    from ..tiers import run_dir
    p = os.path.join(run_dir(scf_root(run), case, "hooked", functional), "record.json")
    try:
        mt = os.stat(p).st_mtime_ns
    except FileNotFoundError:
        return None
    if _CACHE.get(p, (None,))[0] != mt:
        r = json.load(open(p))
        slim = {"status": r.get("status"), "error": r.get("error")}
        if r.get("status") == "ok":
            slim.update(etot_ry=r["energies_ry"]["etot"], iterations=r["iterations"],
                        carrier_iterations=r["carrier"]["iterations"], seconds=r.get("seconds"),
                        hook_seconds=float(sum(h.get("t_hook_s", 0.0) for h in r.get("history", []))))
        _CACHE[p] = (mt, slim)
    return _CACHE[p][1]


def _ok(r):
    return r is not None and r.get("status") == "ok"


# ---- the scan and its extension -------------------------------------------------------------------------------------------------------
def _inner(vol):
    lo, hi = min(vol), max(vol)
    return lo + (hi - lo) / 6.0, hi - (hi - lo) / 6.0


def extension(solid, steps, energies):
    """The scan-design decision for a complete scan (steps: contiguous integers, energies in Ry):
    {'action': 'done'} (V0 of the unguarded BM3 fit inside the inner two thirds of the scanned volumes),
    {'action': 'extend', 'add': [steps]} (points on the side of the minimum, as many as put this V0 inside, at least one,
    |step| <= MAX_STEP; without a BM3 minimum: one point beyond the end that has the lowest energy), or
    {'action': 'stop', 'reason': ...} (not bracketed within +-MAX_STEP %, or no minimum with an interior lowest energy)."""
    from ..bench.eos import EOSError, bm3_fit
    steps = [int(n) for n in steps]
    if sorted(steps) != list(range(min(steps), max(steps) + 1)) or len(energies) != len(steps):
        raise ValueError(f"extension: steps {steps} are not a contiguous scan with one energy each")
    order = np.argsort(steps)
    steps, E = [steps[i] for i in order], np.asarray(energies, float)[order]
    V = [volume(solid, n) for n in steps]
    try:
        v0 = bm3_fit(V, E, guard=False)["V0_bohr3"]
    except EOSError as exc:
        i = int(np.argmin(E))
        if 0 < i < len(E) - 1:
            return {"action": "stop", "reason": f"no BM3 minimum although the lowest energy is at an interior point: {exc}"}
        v0, side = None, (1 if i else -1)
    else:
        lo, hi = _inner(V)
        if lo <= v0 <= hi:
            return {"action": "done", "V0_bohr3": v0}
        side = 1 if v0 > hi else -1
    edge = steps[-1] if side > 0 else steps[0]
    if abs(edge) >= MAX_STEP:
        return {"action": "stop", "V0_bohr3": v0,
                "reason": f"the minimum is not inside the inner two thirds of the scan at |step| = {MAX_STEP} %"}
    add, n = [], edge
    while abs(n) < MAX_STEP:
        n += side
        add.append(n)
        lo, hi = _inner(V + [volume(solid, m) for m in add])
        if v0 is None or lo <= v0 <= hi:
            break
    return {"action": "extend", "add": add, "side": side, "V0_bohr3": v0}


def scan(run, solid, functional):
    """The scan of (solid, functional) as its records determine it: {'steps', 'status', ...} with status 'running' (points
    missing), 'failed' (a point's SCF failed), 'done' (bracketed) or 'stop' (extension stopped; 'reason'). A pure function
    of the records: nothing else is stored, so stages resume."""
    steps, rounds = list(BASE_STEPS), []
    while True:
        recs = [record(run, case_name(solid, n), functional) for n in steps]
        if any(r is not None and r.get("status") == "failed" for r in recs):
            return {"steps": steps, "status": "failed", "rounds": rounds,
                    "failed_steps": [n for n, r in zip(steps, recs) if r is not None and r.get("status") == "failed"]}
        if not all(_ok(r) for r in recs):
            return {"steps": steps, "status": "running", "rounds": rounds}
        d = extension(solid, steps, [r["etot_ry"] for r in recs])
        if d["action"] != "extend":
            return {"steps": steps, "status": d["action"], "rounds": rounds, "reason": d.get("reason")}
        rounds.append(d)
        steps = sorted(steps + d["add"])


# ---- cost model (scheduling only) -----------------------------------------------------------------------------------------------------
# QE s per SCF iteration on a quarter node, from the Fe 80 Ry 8^3 k timing as in mgo_gap (3.7 s; nspin 2, 29 irreducible k, cell
# 79.4 bohr^3, 12 bands, npool 1), scaled by the cell volume, nspin 1 (x 0.5), the irreducible k count, (ecutwfc/80)^1.5 and
# nbnd / 12; the wavefunction FFTs stay on the 4 ecutwfc grid at dual 8 (QE's smooth grid), so dual does not enter the QE part.
# Hook: SCH-SI 5.3 s per round of 16384-point chunks (8 ranks) of the dense (8 ecutwfc) grid, libxc 0.3 s. Iterations:
# carrier 15, hook 15 (libxc) or 20 (SCH-SI). UNCALIBRATED for k-point pools and these cells: once >= CAL_MIN points of the run
# are done, every estimate is multiplied by the median of measured / estimated seconds (cost_scale).
CAL_MIN = 4


def _cubic_ops():
    ops = []
    for p in itertools.permutations(range(3)):
        for sg in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3), dtype=int)
            for i in range(3):
                m[i, p[i]] = sg[i]
            ops.append(m)
    return ops


_NK = {}


def nk_irreducible(n):
    """Irreducible points of the Gamma-centred n^3 mesh of the fcc primitive cell under the cubic point group (48 operations)."""
    if n not in _NK:
        b = np.linalg.inv(0.5 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)).T
        idx = np.array(list(itertools.product(range(n), repeat=3)))
        key = lambda a: (a[:, 0] % n) * n * n + (a[:, 1] % n) * n + (a[:, 2] % n)       # noqa: E731
        canon = key(idx)
        for op in _cubic_ops():
            m = np.rint(b @ op.T @ np.linalg.inv(b)).astype(int)          # the operation on crystal coordinates (row vectors)
            canon = np.minimum(canon, key(idx @ m))
        _NK[n] = int(len(np.unique(canon)))
    return _NK[n]


def fft_points(solid, ecut, dual=DUAL):
    """QE's dense FFT grid of the reference cell (2 n_max + 1 rounded up to 2^a 3^b 5^c 7^d, cubed)."""
    length = A_REF[solid] * BOHR_PER_ANGSTROM / np.sqrt(2.0)
    n = 2 * int(np.sqrt(dual * ecut) * length / (2.0 * np.pi)) + 1
    while True:
        m = n
        for p in (2, 3, 5, 7):
            while m % p == 0:
                m //= p
        if m == 1:
            return n ** 3
        n += 1


def estimate_s(task):
    solid, f = task["solid"], task["functional"]
    ecut = ecutwfc(solid)
    qe = 3.7 * (volume(solid, 0) / 79.4) * 0.5 * (nk_irreducible(kmesh(solid)[0]) / 29.0) * (ecut / 80.0) ** 1.5 \
        * ((n_occ(solid) + NBND_EXTRA) / 12.0)
    sch = f.startswith("SCH-SI")
    rounds = -(-(-(-fft_points(solid, ecut) // 16384)) // 8)
    hook = rounds * (5.3 if sch else 0.3)
    return float(15 * qe + (20 if sch else 15) * (qe + hook))


def cost_scale(run, tasks_):
    """Median of measured / estimated seconds over the finished points (1.0 below CAL_MIN points)."""
    ratios = []
    for t in tasks_:
        r = record(run, t["case"], t["functional"])
        if _ok(r) and r.get("seconds"):
            ratios.append(r["seconds"] / estimate_s(t))
    return float(np.median(ratios)) if len(ratios) >= CAL_MIN else 1.0


# ---- tasks ----------------------------------------------------------------------------------------------------------------------------
def _task(solid, f, step, stage):
    c = case_name(solid, step)
    return {"id": f"{stage}__{c}__{f}", "case": c, "functional": f, "solid": solid, "step": step, "width": 1, "stage": stage,
            "tag": ""}


def tasks(run):
    """Base scans (stage R: PBE, r2SCAN; stage M: the model sets), solid by solid, then per stage the extension points the
    finished scans ask for (stages RX, MX). Order: R, RX, M, MX."""
    out = []
    for group, stage in ((REFERENCE, "R"), (MODELS, "M")):
        out += [_task(s, f, n, stage) for s in SOLIDS for f in group for n in BASE_STEPS]
        for s in SOLIDS:
            for f in group:
                out += [_task(s, f, n, stage + "X") for n in scan(run, s, f)["steps"] if n not in BASE_STEPS]
    k = cost_scale(run, out)
    return [dict(t, est_s=k * estimate_s(t)) for t in out]


def done(run, task):
    return _ok(record(run, task["case"], task["functional"]))


def failed(run, task):
    r = record(run, task["case"], task["functional"])
    return r is not None and r.get("status") == "failed"


def command(run, task):
    return ["scf", "--case", task["case"], "--mode", "hooked", "--name", task["functional"], "--root", scf_root(run),
            "--npool", str(NPOOL)]


# ---- tables ---------------------------------------------------------------------------------------------------------------------------
def _a0(v0):
    return float((4.0 * v0) ** (1.0 / 3.0) / BOHR_PER_ANGSTROM)            # fcc primitive: V = a^3 / 4


def _fit(solid, steps, energies, guard):
    """BM3 fit of the given points -> (dict with a0_angstrom, B0_GPa, B0_prime, rms_ry, ..., None) or (None, error text)."""
    from ..bench.eos import EOSError, bm3_fit
    try:
        fit = bm3_fit([volume(solid, n) for n in steps], energies, guard=guard)
    except EOSError as exc:
        return None, str(exc)
    return dict(fit, a0_angstrom=_a0(fit["V0_bohr3"])), None


def entry(run, solid, f):
    sc = scan(run, solid, f)
    steps = sc["steps"]
    recs = {n: record(run, case_name(solid, n), f) for n in steps}
    col = REF["collaborator"]
    out = {"solid": solid, "functional": f, "a_ref_angstrom": A_REF[solid], "status": sc["status"], "reason": sc.get("reason"),
           "steps": steps, "extended": len(steps) > len(BASE_STEPS), "failed_steps": sc.get("failed_steps", []),
           "points": [dict(step=n, a_angstrom=a_of(solid, n), V_bohr3=volume(solid, n),
                           **({k: recs[n][k] for k in ("etot_ry", "iterations", "carrier_iterations", "seconds")} if _ok(recs[n])
                              else {"status": None if recs[n] is None else recs[n].get("status")})) for n in steps],
           "fit": None, "fit_error": None, "fit_pm3": None, "fit_pm3_error": None,
           "collaborator_a0_angstrom": col["a0_angstrom"].get(f, {}).get(solid),
           "collaborator_B0_GPa": col["B0_GPa"].get(f, {}).get(solid)}
    if all(_ok(recs[n]) for n in BASE_STEPS):
        # the collaborator's protocol like for like: the base +-3 % points, no guard (flags v0_inner_two_thirds, rms_ok kept)
        out["fit_pm3"], out["fit_pm3_error"] = _fit(solid, BASE_STEPS, [recs[n]["etot_ry"] for n in BASE_STEPS], guard=False)
    if sc["status"] == "done":
        out["fit"], out["fit_error"] = _fit(solid, steps, [recs[n]["etot_ry"] for n in steps], guard=True)
    elif sc["status"] == "stop":
        out["fit_error"] = sc["reason"]
    for key in ("fit", "fit_pm3"):
        if out[key]:
            out[key]["a0_minus_ref_angstrom"] = out[key]["a0_angstrom"] - A_REF[solid]
            if out["collaborator_a0_angstrom"] is not None:
                out[key]["a0_minus_collaborator_angstrom"] = out[key]["a0_angstrom"] - out["collaborator_a0_angstrom"]
    return out


def _stats(rows, key):
    d = [(r[key]["a0_minus_ref_angstrom"], r["a_ref_angstrom"]) for r in rows if r[key]]
    if not d:
        return {"n": 0, "complete": False}
    e = np.array([x[0] for x in d])
    return {"n": len(d), "complete": len(d) == len(SOLIDS), "ME_angstrom": float(e.mean()), "MAE_angstrom": float(np.abs(e).mean()),
            "MARE_percent": float(np.mean([100.0 * abs(x) / a for x, a in d]))}


def table(run):
    rows = [entry(run, s, f) for f in FUNCTIONALS for s in SOLIDS]
    ts = tasks(run)
    secs = [r["seconds"] for t in ts if _ok(r := record(run, t["case"], t["functional"])) and r.get("seconds")]
    return {"what": "relaxed lattice constants of rock-salt ionic solids (BM3 of E(V) scans; conventional a0 in Angstrom)",
            "protocol": {"per_solid": protocol(), "functionals": list(FUNCTIONALS), "reference_functionals": list(REFERENCE),
                         "xchole_line": XCHOLE_LINE, "dual": DUAL, "kspacing_per_angstrom": KSPACING,
                         "kmesh_convention": "Gamma-centred, unshifted; from the reference cell, fixed across the scan",
                         "base_steps_percent": list(BASE_STEPS), "max_step_percent": MAX_STEP, "npool": NPOOL,
                         "scf": "conv_thr 1e-8 Ry, Broyden beta 0.3, cold start + converged carrier (QE TPSS), then the hook",
                         "fit": "third-order Birch-Murnaghan, linear least squares in V^(-2/3) (xcsb.bench.eos.bm3_fit)",
                         "a_ref_source": REF["a_ref_source"], "collaborator_source": REF["collaborator"]["source"],
                         "protocol_source": REF["protocol_source"]},
            "rows": rows,
            "summary": {f: {"fit": _stats([r for r in rows if r["functional"] == f], "fit"),
                            "fit_pm3": _stats([r for r in rows if r["functional"] == f], "fit_pm3")} for f in FUNCTIONALS},
            "timing": {"points_done": len(secs), "points_listed": len(ts), "quarter_node_hours_done": float(sum(secs)) / 3600.0,
                       "cost_scale": cost_scale(run, ts),
                       "quarter_node_hours_pending_estimate": float(sum(t["est_s"] for t in ts if not done(run, t))) / 3600.0}}


def bank_reference(run):
    """The archive-ready result of the reference functionals (PBE, r2SCAN) in this protocol (written to the archive by hand,
    once; write-once). Raises until every solid has its final fit for both."""
    t = table(run)
    rows = [r for r in t["rows"] if r["functional"] in REFERENCE]
    missing = [(r["solid"], r["functional"]) for r in rows if not r["fit"]]
    if missing:
        raise RuntimeError(f"no final fit yet for {missing}")
    keep = ("a0_angstrom", "B0_GPa", "B0_prime", "rms_ry", "E0_ry", "V0_bohr3", "n_points")
    return {"what": "relaxed lattice constants of rock-salt LiF, LiCl, NaF, NaCl, MgO with the reference functionals through "
                    "the xcSolidsBench hooks (LC20 protocol of the collaborator; user 2026-11-13)",
            "protocol": dict(t["protocol"], pseudopotentials="SG15 ONCV 1.2, no NLCC", cell="rock salt, primitive, 2 atoms",
                             nspin=1, occupations="fixed"),
            "results": {f: {r["solid"]: {"fit": {k: r["fit"][k] for k in keep}, "steps_percent": r["steps"],
                                         "extended": r["extended"],
                                         "fit_pm3_unguarded": None if not r["fit_pm3"] else
                                         {k: r["fit_pm3"][k] for k in keep + ("v0_inner_two_thirds", "rms_ok")},
                                         "points": r["points"]} for r in rows if r["functional"] == f} for f in REFERENCE},
            "summary": {f: t["summary"][f] for f in REFERENCE}}


def _f(x, fmt):
    return "" if x is None else format(x, fmt)


def render(t) -> str:
    p = t["protocol"]
    out = ["# Relaxed lattice constants of rock-salt ionic solids (conventional a0, Angstrom; BM3 of E(V))", "",
           f"Reference a_ref: {p['a_ref_source']}.", f"Cross-check column: {p['collaborator_source']} (their own host).",
           f"Protocol: ecutrho = {p['dual']} ecutwfc; k spacing {p['kspacing_per_angstrom']} 1/A ({p['kmesh_convention']}); "
           f"{p['scf']}; a = a_ref (1 + s), s = {p['base_steps_percent'][0]} ... {p['base_steps_percent'][-1]} %, extended in "
           f"1 % steps (|s| <= {p['max_step_percent']} %) when V0 is outside the inner two thirds of the scan.", "",
           "| solid | a_ref | ecutwfc (Ry) | k mesh | irreducible k | n_occ | nbnd |", "|---|---|---|---|---|---|---|"]
    for s, v in p["per_solid"].items():
        out.append(f"| {s} | {v['a_ref_angstrom']:.3f} | {v['ecutwfc_ry']:g} | {'x'.join(str(k) for k in v['kmesh'])} | "
                   f"{v['nk_irreducible']} | {v['n_occ']} | {v['nbnd']} |")
    out += ["", "## Results (final fit: all scanned points, guarded; '+-3 % plain': the 7 base points, unguarded)",
            "| functional | solid | a_ref | a0 | a0 - a_ref | B0 (GPa) | B0' | rms (Ry) | scan (%) | a0 (+-3 % plain) | "
            "V0 inside (plain) | collaborator a0 | a0 (plain) - collaborator | collaborator B0 | B0 (plain) | status |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in t["rows"]:
        fit, pm = r["fit"] or {}, r["fit_pm3"] or {}
        scan_txt = f"{r['steps'][0]:+d} ... {r['steps'][-1]:+d}" + (" (extended)" if r["extended"] else "")
        status = r["status"] + (f": {r['fit_error']}" if r["fit_error"] else "") \
            + (f" (failed steps {r['failed_steps']})" if r["failed_steps"] else "")
        out.append(f"| {r['functional']} | {r['solid']} | {r['a_ref_angstrom']:.3f} | {_f(fit.get('a0_angstrom'), '.4f')} | "
                   f"{_f(fit.get('a0_minus_ref_angstrom'), '+.4f')} | {_f(fit.get('B0_GPa'), '.2f')} | "
                   f"{_f(fit.get('B0_prime'), '.2f')} | {_f(fit.get('rms_ry'), '.1e')} | {scan_txt} | "
                   f"{_f(pm.get('a0_angstrom'), '.4f')} | {'' if not pm else ('yes' if pm['v0_inner_two_thirds'] else 'NO')} | "
                   f"{_f(r['collaborator_a0_angstrom'], '.4f')} | {_f(pm.get('a0_minus_collaborator_angstrom'), '+.4f')} | "
                   f"{_f(r['collaborator_B0_GPa'], '.2f')} | {_f(pm.get('B0_GPa'), '.2f')} | {status} |")
    out += ["", "## Errors against a_ref over the five solids (Angstrom; MARE in %)",
            "| functional | n (final) | ME | MAE | MARE | n (+-3 % plain) | ME (plain) | MAE (plain) | MARE (plain) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for f, s in t["summary"].items():
        a, b = s["fit"], s["fit_pm3"]
        out.append(f"| {f} | {a['n']}{'' if a['complete'] else ' (incomplete)'} | {_f(a.get('ME_angstrom'), '+.4f')} | "
                   f"{_f(a.get('MAE_angstrom'), '.4f')} | {_f(a.get('MARE_percent'), '.3f')} | "
                   f"{b['n']}{'' if b['complete'] else ' (incomplete)'} | {_f(b.get('ME_angstrom'), '+.4f')} | "
                   f"{_f(b.get('MAE_angstrom'), '.4f')} | {_f(b.get('MARE_percent'), '.3f')} |")
    tm = t["timing"]
    out += ["", f"Timing: {tm['points_done']} of {tm['points_listed']} listed points done, {tm['quarter_node_hours_done']:.2f} "
            f"quarter-node h used; measured / estimated cost {tm['cost_scale']:.2f}; pending estimate "
            f"{tm['quarter_node_hours_pending_estimate']:.2f} quarter-node h."]
    return "\n".join(out)
