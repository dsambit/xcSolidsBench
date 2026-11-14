"""Relaxed lattice constants of five rock-salt ionic solids (user 2026-11-13: "test the relaxed lattice constants of ionic
solids using the approach my collaborator has used"): LiF, LiCl, NaF, NaCl, MgO with PBE, r2SCAN, SCH-SI-A:v3.2 and
SCH-SI-B:v3.2 through the hooks (xchole line v3.2), in the collaborator's LC20 protocol. One protocol for every solid and
functional.

Cell: rock salt, primitive fcc, 2 atoms, nspin 1, fixed occupations, nbnd = n_occ + 4 (n_occ from the SG15 1.2 valence charges
of pseudo/MANIFEST.json). ecutwfc = max(80 Ry, the per-element floors), ecutrho = 8 ecutwfc (DUAL), both fixed across the scan.
k mesh: N_i = max(1, ceil(|b_i| / 0.1 A^-1)) with |b_i| the reciprocal-lattice vector lengths (2 pi included; the VASP
k-spacing convention) of the REFERENCE cell, fixed across the scan; Gamma-centred, unshifted (this host's convention, DESIGN 6.0,
and VASP's for an automatic spacing mesh). conv_thr 1e-8 Ry (bench), Broyden beta 0.3, cold starts with the start
'pbe-scaffold' (scf.driver.run_hooked): one XC-free diagonalization of QE's starting state, then PBE through the hooks to
convergence, then the functional; QE's own TPSS is never applied (the collaborator: iteration 0 = libxc PBE through the hooks).
The native TPSS carrier of the earlier studies stalled here (jobs 61245307, 61245548, 2026-11-13: LiF, 624 k, 96 Ry, every SCF
still in the first carrier iteration after 48 min with c_bands non-convergence warnings).

Parallel layout: all parallelism in k-point pools: a quarter-node slot = 32 ranks x 1 thread (one physical core each), npool 32
(2-atom cells with 256-624 irreducible k: plane-wave and thread parallelism inside a pool buys little on 36^3-48^3 FFT grids).

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

'LC-phase0': the job's plumbing check only (MgO at a_ref, 40 Ry, 6^3 k; never a result). 'LC-diag-k<n>-<tpss|pbe>': the
diagnostic job's cases (LiF at a_ref, protocol cutoffs and bands, n^3 k, start 'tpss-carrier' or 'pbe-scaffold'; never a result).
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
START = "pbe-scaffold"                             # the cold start of every point (scf.settings.STARTS)
RANKS_PER_SLOT = 32                                # a quarter node: 32 ranks x 2 logical CPUs (one core), OMP_NUM_THREADS = 1
NPOOL = 32                                         # one rank per k-point pool (no eigenvalues needed)
DIAG_STARTS = {"tpss": "tpss-carrier", "pbe": "pbe-scaffold"}
MIX_STARTS = {"reset": "pbe-scaffold", "stale": "pbe-scaffold-stale"}
MIX_K = 10                                         # the mixer test's coarse mesh (47 irreducible k: one per pool of a slot)
PHASE0 = "LC-phase0"
DIAG_SOLID = "LiF"                                 # the solid of the stall (the largest mesh)
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


def diag_case_name(k, start):
    return f"LC-diag-k{int(k)}-{start}"


def parse_diag(name):
    parts = name.split("-")
    if len(parts) != 4 or parts[:2] != ["LC", "diag"] or not parts[2][1:].isdigit() or parts[2][0] != "k" or parts[3] not in DIAG_STARTS:
        raise KeyError(f"not a diagnostic case name: {name!r}")
    return int(parts[2][1:]), parts[3]


def mix_case_name(solid, step, k, start):
    return f"LC-mix-{solid}-{'m' if step < 0 else 'p'}{abs(int(step))}-k{int(k)}-{start}"


def parse_mix(name):
    parts = name.split("-")
    if len(parts) != 6 or parts[:2] != ["LC", "mix"] or parts[4][:1] != "k" or not parts[4][1:].isdigit() or parts[5] not in MIX_STARTS:
        raise KeyError(f"not a mixer-test case name: {name!r}")
    solid, step = parse_case("-".join(["LC"] + parts[2:4]))
    return solid, step, int(parts[4][1:]), parts[5]


def get_case(name) -> Case:
    if name.startswith("LC-mix-"):
        solid, step, k, st = parse_mix(name)
        return Case(structure(solid, step), ecutwfc=ecutwfc(solid), kmesh=(k, k, k), kshift=(0, 0, 0), occupations="fixed",
                    nspin=1, nbnd=n_occ(solid) + NBND_EXTRA, dual=DUAL, conv_class="bench", start=MIX_STARTS[st])
    if name == PHASE0:
        return Case(structure("MgO"), ecutwfc=40.0, kmesh=(6, 6, 6), kshift=(0, 0, 0), occupations="fixed", nspin=1,
                    nbnd=n_occ("MgO") + NBND_EXTRA, dual=DUAL, conv_class="bench", start=START)
    if name.startswith("LC-diag-"):
        k, st = parse_diag(name)
        return Case(structure(DIAG_SOLID), ecutwfc=ecutwfc(DIAG_SOLID), kmesh=(k, k, k), kshift=(0, 0, 0), occupations="fixed",
                    nspin=1, nbnd=n_occ(DIAG_SOLID) + NBND_EXTRA, dual=DUAL, conv_class="bench", start=DIAG_STARTS[st])
    solid, step = parse_case(name)
    return Case(structure(solid, step), ecutwfc=ecutwfc(solid), kmesh=kmesh(solid), kshift=(0, 0, 0), occupations="fixed",
                nspin=1, nbnd=n_occ(solid) + NBND_EXTRA, dual=DUAL, conv_class="bench", start=START)


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
# seconds = overhead_s + (1 + it_scaffold + it_hook) * ceil(nk / NPOOL) * c_k + it_hook * hook, with
#   c_k = c_k_lif_s * (V / V_LiF) * (ecutwfc / 96)^1.5 * (nbnd / 9): QE seconds per k point and SCF iteration on one single-thread
#         rank (one pool), LiF at 96 Ry with 9 bands as the unit;
#   hook = hook_sch_s (SCH-SI) or hook_libxc_s per hook iteration (the dense grids fit one round of chunks on 32 ranks).
# COST below is a PRIOR, not a measurement: c_k_lif_s 6 s from phase 0 of job 61245307 (MgO 40 Ry, 12 bands, 2 k per rank with 4
# threads: 104 s / 30 iterations / 2 k = 1.7 s, x 2.4 for LiF at 96 Ry with 9 bands, x 1.5 for one thread, rounded up; an upper
# bound, since that 1.7 s contains the per-iteration host overhead); the collaborator's 49 s per LC20 r2SCAN point on 128
# ranks would mean a few tenths of a second. Iterations from the same phase 0 (carrier 18; hook PBE 12, SCH-SI 17).
# The diagnostic job measures c_k_lif_s, hook_sch_s and the iteration counts and writes COST.json (diag_summary); a run reads
# <run>/COST.json over the prior. Within a run, once >= CAL_MIN points are done, every estimate is also multiplied by the
# median of measured / estimated seconds (cost_scale).
COST = {"c_k_lif_s": 6.0, "hook_sch_s": 25.0, "hook_libxc_s": 1.0, "overhead_s": 120.0, "it_scaffold": 15,
        "it_hook": {"PBE": 2, "r2SCAN": 12, "SCH-SI": 17}}
CAL_MIN = 4
_COST_CACHE = {}


def cost(run=None):
    """COST, with the keys of <run>/COST.json (the diagnostic's measurements) over the prior; 'source' says which."""
    out = dict(COST, it_hook=dict(COST["it_hook"]), source="prior (xcsb.studies.ionic_lc.COST)")
    p = None if run is None else os.path.join(run, "COST.json")
    if p is None or not os.path.exists(p):
        return out
    mt = os.stat(p).st_mtime_ns
    if _COST_CACHE.get(p, (None,))[0] != mt:
        _COST_CACHE[p] = (mt, json.load(open(p)))
    m = _COST_CACHE[p][1]
    unknown = set(m) - set(COST) - {"source", "measured"}
    if unknown:
        raise KeyError(f"{p}: unknown cost keys {sorted(unknown)}")
    out.update({k: v for k, v in m.items() if k not in ("it_hook", "source", "measured")})
    out["it_hook"].update(m.get("it_hook", {}))
    out["source"] = f"{p}: {m.get('source', 'measured')}"
    return out


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


def k_per_pool(nk, npool=NPOOL):
    return -(-int(nk) // int(npool))


def c_k_scale(solid):
    """c_k of a solid in units of LiF's (plane waves x bands)."""
    return (volume(solid, 0) / volume(DIAG_SOLID, 0)) * (ecutwfc(solid) / ecutwfc(DIAG_SOLID)) ** 1.5 \
        * ((n_occ(solid) + NBND_EXTRA) / (n_occ(DIAG_SOLID) + NBND_EXTRA))


def estimate_s(task, c=None):
    c = c or cost()
    solid, f = task["solid"], task["functional"]
    sch = f.startswith("SCH-SI")
    it_hook = c["it_hook"]["SCH-SI" if sch else f]
    qe = k_per_pool(nk_irreducible(kmesh(solid)[0])) * c["c_k_lif_s"] * c_k_scale(solid)
    return float(c["overhead_s"] + (1 + c["it_scaffold"] + it_hook) * qe + it_hook * (c["hook_sch_s"] if sch else c["hook_libxc_s"]))


def cost_scale(run, tasks_):
    """Median of measured / estimated seconds over the finished points (1.0 below CAL_MIN points)."""
    c = cost(run)
    ratios = []
    for t in tasks_:
        r = record(run, t["case"], t["functional"])
        if _ok(r) and r.get("seconds"):
            ratios.append(r["seconds"] / estimate_s(t, c))
    return float(np.median(ratios)) if len(ratios) >= CAL_MIN else 1.0


def srun_layout(task):
    """(ranks, logical CPUs per rank) of a task for xcsb.campaign: one single-thread rank per physical core and pool."""
    return RANKS_PER_SLOT * task["width"], 2


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
    c, k = cost(run), cost_scale(run, out)
    return [dict(t, est_s=k * estimate_s(t, c)) for t in out]


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
                         "ranks_per_slot": RANKS_PER_SLOT, "threads": 1, "start": START,
                         "scf": "conv_thr 1e-8 Ry, Broyden beta 0.3, cold start: one XC-free diagonalization, converged PBE "
                                "through the hooks, then the hook of the functional",
                         "fit": "third-order Birch-Murnaghan, linear least squares in V^(-2/3) (xcsb.bench.eos.bm3_fit)",
                         "a_ref_source": REF["a_ref_source"], "collaborator_source": REF["collaborator"]["source"],
                         "protocol_source": REF["protocol_source"]},
            "rows": rows,
            "summary": {f: {"fit": _stats([r for r in rows if r["functional"] == f], "fit"),
                            "fit_pm3": _stats([r for r in rows if r["functional"] == f], "fit_pm3")} for f in FUNCTIONALS},
            "timing": {"points_done": len(secs), "points_listed": len(ts), "quarter_node_hours_done": float(sum(secs)) / 3600.0,
                       "cost_scale": cost_scale(run, ts), "cost": cost(run),
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
            f"quarter-node h used; measured / estimated cost {tm['cost_scale']:.2f} (cost model: {tm['cost']['source']}); "
            f"pending estimate {tm['quarter_node_hours_pending_estimate']:.2f} quarter-node h."]
    return "\n".join(out)


# ---- the diagnostic job (job_ionic_lc_diag.sh) ------------------------------------------------------------------------------------
# name -> (case, functional, tag, ranks, threads, phase): LiF at a_ref, PBE unless named. Phase A: 12^3 k (72 irreducible), the old
# and the new start side by side, the SCH-SI hook cost, and the earlier layout (8 ranks x 4 threads); phase B: the full 28^3 mesh,
# old start, new start, and the new start on 64 pools (pool scaling).
DIAG_K = (12, 28)
DIAG_RUNS = {
    "k12_tpss": (diag_case_name(12, "tpss"), "PBE", "", 32, 1, "A"),
    "k12_pbe": (diag_case_name(12, "pbe"), "PBE", "", 32, 1, "A"),
    "k12_pbe_schA": (diag_case_name(12, "pbe"), "SCH-SI-A:v3.2", "", 32, 1, "A"),
    "k12_pbe_r8t4": (diag_case_name(12, "pbe"), "PBE", "r8t4", 8, 4, "A"),
    "k28_tpss": (diag_case_name(28, "tpss"), "PBE", "", 32, 1, "B"),
    "k28_pbe": (diag_case_name(28, "pbe"), "PBE", "", 32, 1, "B"),
    "k28_pbe_r64": (diag_case_name(28, "pbe"), "PBE", "r64", 64, 1, "B"),
}


def diag_dir(run, name):
    from ..tiers import run_dir
    case, f, tag, *_ = DIAG_RUNS[name]
    return run_dir(scf_root(run), case, "hooked", f) + (f"__{tag}" if tag else "")


def diag_command(run, name):
    """(ranks, logical CPUs per rank, threads, xcsb CLI arguments) of a diagnostic run (one rank per pool)."""
    case, f, tag, ranks, threads, _ = DIAG_RUNS[name]
    args = ["scf", "--case", case, "--mode", "hooked", "--name", f, "--root", scf_root(run), "--npool", str(ranks)]
    return ranks, 2 * threads, threads, args + (["--tag", tag] if tag else [])


def parse_pw_out(text):
    """From a pw.out (complete or cut off): the k-point count, per started SCF iteration QE's cumulative cpu time printed after
    its header and the number of c_bands non-convergence warnings (lines and eigenvalues) that follow it."""
    import re
    out = {"nk": None, "iterations": [], "last_cpu_s": None, "warnings_before_first_iteration": 0}
    cur = None
    for line in text.splitlines():
        if (m := re.search(r"number of k points=\s*(\d+)", line)):
            out["nk"] = int(m.group(1))
        elif (m := re.search(r"iteration #\s*(\d+)", line)):
            cur = {"iteration": int(m.group(1)), "cpu_s_at_start": None, "c_bands_warnings": 0, "eigenvalues_not_converged": 0}
            out["iterations"].append(cur)
        elif (m := re.search(r"total cpu time spent up to now is\s+([0-9.]+)\s+secs", line)):
            out["last_cpu_s"] = float(m.group(1))
            if cur is not None and cur["cpu_s_at_start"] is None:
                cur["cpu_s_at_start"] = float(m.group(1))
        elif (m := re.search(r"c_bands:\s*(\d+)\s+eigenvalues not converged", line)):
            if cur is None:
                out["warnings_before_first_iteration"] += 1
            else:
                cur["c_bands_warnings"] += 1
                cur["eigenvalues_not_converged"] += int(m.group(1))
    return out


def _median(x):
    return None if not len(x) else float(np.median(x))


def diag_entry(run, name):
    case, f, tag, ranks, threads, phase = DIAG_RUNS[name]
    k, st = parse_diag(case)
    d = diag_dir(run, name)
    nk = nk_irreducible(k)
    out = {"name": name, "case": case, "functional": f, "start": DIAG_STARTS[st], "kmesh": k, "nk_irreducible": nk,
           "ranks": ranks, "npool": ranks, "threads": threads, "k_per_pool": k_per_pool(nk, ranks), "phase": phase,
           "status": "missing", "pw_out": None}
    p = os.path.join(d, "pw.out")
    if os.path.exists(p):
        pw = parse_pw_out(open(p, errors="replace").read())
        out["pw_out"] = {"nk": pw["nk"], "iterations_started": len(pw["iterations"]), "last_cpu_s": pw["last_cpu_s"],
                         "c_bands_warnings": sum(i["c_bands_warnings"] for i in pw["iterations"]) + pw["warnings_before_first_iteration"],
                         "c_bands_warnings_per_iteration": [i["c_bands_warnings"] for i in pw["iterations"]],
                         "cpu_s_at_iteration_start": [i["cpu_s_at_start"] for i in pw["iterations"]]}
        out["status"] = "no record (killed or stalled)"
    p = os.path.join(d, "record.json")
    if os.path.exists(p):
        r = json.load(open(p))
        car, hist = r.get("carrier") or {}, r.get("history") or []
        t_start = [h["t_iter_s"] - h.get("t_hook_s", 0.0) for h in car.get("history", []) if "t_iter_s" in h]
        t_hook_qe = [h["t_iter_s"] - h["t_hook_s"] for h in hist]
        qe = t_start + t_hook_qe
        out.update(status=r.get("status"), error=r.get("error"), seconds=r.get("seconds"),
                   etot_ry=(r.get("energies_ry") or {}).get("etot"),
                   bootstrap_s=(car.get("bootstrap") or {}).get("t_iter_s"), start_iterations=car.get("iterations"),
                   start_iterations_recorded=len(car.get("history", [])), hook_iterations=r.get("iterations"),
                   hook_iterations_recorded=len(hist), start_t_iter_s=[h.get("t_iter_s") for h in car.get("history", [])],
                   hook_t_iter_s=[h["t_iter_s"] for h in hist],
                   t_hook_s_median=_median([h["t_hook_s"] for h in hist] or [h["t_hook_s"] for h in car.get("history", []) if "t_hook_s" in h]),
                   qe_s_per_iteration_median=_median(qe),
                   c_k_s=None if not qe else _median(qe) / out["k_per_pool"])
    return out


def diag_summary(run):
    """All diagnostic runs, the stall verdict and the measured cost constants (written to <run>/COST.json by the CLI when
    the full-mesh run with the new start has per-iteration times)."""
    rows = {n: diag_entry(run, n) for n in DIAG_RUNS}
    new, old = rows["k28_pbe"], rows["k28_tpss"]
    measured = {}
    if new.get("c_k_s"):
        measured["c_k_lif_s"] = new["c_k_s"]
    if new.get("status") == "ok":
        measured["it_scaffold"] = new["start_iterations"]
        measured["it_hook"] = {"PBE": new["hook_iterations"]}
    a = rows["k12_pbe_schA"]
    if a.get("hook_iterations_recorded"):
        measured["hook_sch_s"] = a["t_hook_s_median"]
        if a.get("status") == "ok":
            measured.setdefault("it_hook", {})["SCH-SI"] = a["hook_iterations"]
    if rows["k12_pbe"].get("t_hook_s_median") is not None:
        measured["hook_libxc_s"] = rows["k12_pbe"]["t_hook_s_median"]
    verdict = {"new_start_full_mesh": new["status"],
               "new_start_c_bands_warnings": None if not new["pw_out"] else new["pw_out"]["c_bands_warnings"],
               "old_start_full_mesh": old["status"],
               "old_start_iterations_started": None if not old["pw_out"] else old["pw_out"]["iterations_started"],
               "old_start_c_bands_warnings": None if not old["pw_out"] else old["pw_out"]["c_bands_warnings"],
               "stall_gone": new["status"] == "ok"}
    for n in ("k12_pbe", "k12_pbe_r8t4", "k28_pbe", "k28_pbe_r64"):
        verdict[f"qe_s_per_iteration_{n}"] = rows[n].get("qe_s_per_iteration_median")
    e = [rows[n].get("etot_ry") for n in ("k12_pbe", "k12_tpss", "k12_pbe_r8t4")]
    verdict["k12_etot_new_minus_old_ry"] = None if e[0] is None or e[1] is None else e[0] - e[1]
    verdict["k12_etot_r32_minus_r8t4_ry"] = None if e[0] is None or e[2] is None else e[0] - e[2]
    e = [rows[n].get("etot_ry") for n in ("k28_pbe", "k28_pbe_r64")]
    verdict["k28_etot_r32_minus_r64_ry"] = None if None in e else e[0] - e[1]
    out = {"what": "ionic lattice constants: start and layout diagnostic (LiF at a_ref, protocol cutoffs)", "rows": rows,
           "verdict": verdict, "cost_measured": measured}
    if "c_k_lif_s" in measured:
        c = dict(cost(), **{k: v for k, v in measured.items() if k != "it_hook"})
        c["it_hook"] = dict(COST["it_hook"], **measured.get("it_hook", {}))
        ts = [dict(_task(s_, f, n, STAGE[f]), ) for s_ in SOLIDS for f in FUNCTIONALS for n in BASE_STEPS]
        out["study_estimate"] = {"quarter_node_hours": {st: sum(estimate_s(t, c) for t in ts if t["stage"] == st) / 3600.0
                                                         for st in ("R", "M")},
                                 "per_point_s": {f"{s_}|{f}": estimate_s(_task(s_, f, 0, STAGE[f]), c) for s_ in SOLIDS for f in FUNCTIONALS}}
    return out


def diag_cost_json(summary):
    """The COST.json content of a diagnostic summary (None without a measured c_k)."""
    m = summary["cost_measured"]
    if "c_k_lif_s" not in m:
        return None
    return dict(m, source="diagnostic job (xcsb lc diag-summary): LiF 28^3 k, start pbe-scaffold, 32 ranks x 1 thread, npool 32",
                measured=sorted(m))


def render_diag(t) -> str:
    out = ["# Ionic lattice constants: start and layout diagnostic (LiF at a_ref, 96 / 768 Ry, 9 bands, PBE unless named)", "",
           "| run | start | k | irr. k | ranks x threads | k per pool | status | start it (boot + scaffold or carrier) | hook it | "
           "QE s / iteration (median) | s per k and iteration | hook s (median) | total s | pw.out: iterations started | "
           "c_bands warnings | E_tot (Ry) |", "|" + "---|" * 16]
    for n, r in t["rows"].items():
        pw = r["pw_out"] or {}
        out.append(f"| {n} ({r['functional']}) | {r['start']} | {r['kmesh']}^3 | {r['nk_irreducible']} | {r['ranks']} x {r['threads']} | "
                   f"{r['k_per_pool']} | {r['status']}{' ' + r['error'] if r.get('error') else ''} | "
                   f"{'' if r.get('start_iterations_recorded') is None else r['start_iterations_recorded']} | "
                   f"{'' if r.get('hook_iterations_recorded') is None else r['hook_iterations_recorded']} | "
                   f"{_f(r.get('qe_s_per_iteration_median'), '.1f')} | {_f(r.get('c_k_s'), '.3f')} | {_f(r.get('t_hook_s_median'), '.2f')} | "
                   f"{_f(r.get('seconds'), '.0f')} | {pw.get('iterations_started', '')} | {pw.get('c_bands_warnings', '')} | "
                   f"{_f(r.get('etot_ry'), '.8f')} |")
    out += ["", "Verdict: " + json.dumps(t["verdict"]), "", "Measured cost constants: " + json.dumps(t["cost_measured"])]
    for n, r in t["rows"].items():
        if r["pw_out"]:
            out.append(f"{n}: QE cpu s at the start of each iteration {r['pw_out']['cpu_s_at_iteration_start']}, last printed "
                       f"{r['pw_out']['last_cpu_s']}; c_bands warnings per iteration {r['pw_out']['c_bands_warnings_per_iteration']}")
        if r.get("start_t_iter_s") is not None:
            out.append(f"{n}: wall s per iteration: bootstrap {r.get('bootstrap_s')}, start {[round(x, 1) for x in r['start_t_iter_s'] if x is not None]}, "
                       f"hook {[round(x, 1) for x in r['hook_t_iter_s']]}")
    if "study_estimate" in t:
        q = t["study_estimate"]["quarter_node_hours"]
        out += ["", f"Full study from these constants (140 base points, no extensions): stage R {q['R']:.1f}, stage M {q['M']:.1f} "
                f"quarter-node h = {(q['R'] + q['M']) / 4:.1f} node-h."]
    return "\n".join(out)



# ---- the mixer test (job_ionic_lc_mix.sh) -------------------------------------------------------------------------------------------
# Coarse mesh (MIX_K^3 k, protocol cutoffs and bands), 32 ranks x 1 thread: the seven A/B points of the scan 61248332 that ended
# with a negative mixed density, and LiF at a_ref, each with the mixer reset between the scaffold and the functional ('reset',
# the start 'pbe-scaffold') and without it ('stale', the behaviour until 143eca6). Never a result.
MIX_POINTS = (("NaCl", -1, "SCH-SI-A:v3.2"), ("NaCl", -3, "SCH-SI-B:v3.2"), ("NaCl", -1, "SCH-SI-B:v3.2"), ("LiCl", 7, "SCH-SI-A:v3.2"),
              ("LiCl", 6, "SCH-SI-B:v3.2"), ("LiCl", 7, "SCH-SI-B:v3.2"), ("NaF", 7, "SCH-SI-A:v3.2"), ("LiF", 0, "SCH-SI-A:v3.2"),
              ("LiF", 0, "r2SCAN"))
MIX_FROZEN_DEXC = 1e-4                             # Ry: an iteration whose E_xc differs less from the first one's is 'frozen'


def mix_runs():
    return [(mix_case_name(s, st, MIX_K, m), f) for s, st, f in MIX_POINTS for m in ("reset", "stale")]


def mix_command(run, case, f):
    return ["scf", "--case", case, "--mode", "hooked", "--name", f, "--root", scf_root(run), "--npool", str(NPOOL)]


def mix_entry(run, case, f):
    from ..tiers import run_dir
    p = os.path.join(run_dir(scf_root(run), case, "hooked", f), "record.json")
    solid, step, k, st = parse_mix(case)
    out = {"case": case, "solid": solid, "step": step, "functional": f, "start": st, "status": "missing"}
    if not os.path.exists(p):
        return out
    r = json.load(open(p))
    h = r.get("history", [])
    exc = [x["e_xc_ry"] for x in h if x.get("e_xc_ry") is not None]
    frozen = 0
    for e in exc[1:]:
        if abs(e - exc[0]) >= MIX_FROZEN_DEXC:
            break
        frozen += 1
    e = [x["etot_ry"] for x in h]
    out.update(status=r.get("status"), error=(r.get("error") or "")[:160] or None, scaffold_iterations=r["carrier"].get("iterations"),
               iterations=len(h), frozen_after_first=frozen, etot_ry=e[-1] if e else None,
               etot_spread_last3_ry=(max(e[-3:]) - min(e[-3:])) if len(e) >= 3 else None,
               dr2_last=h[-1]["dr2"] if h else None, min_rho=min((x["min_rho"][0] for x in h if "min_rho" in x), default=None),
               seconds=r.get("seconds"))
    return out


def mix_summary(run):
    return {"what": f"ionic lattice constants: mixer test ({MIX_K}^3 k; reset = new SCF for the functional, stale = Broyden "
                    "history of the scaffold kept)", "rows": [mix_entry(run, c, f) for c, f in mix_runs()]}


def render_mix(t) -> str:
    g = lambda v, fmt: "" if v is None else format(v, fmt)      # noqa: E731
    out = [f"# {t['what']}", "", "| solid | step (%) | functional | start | status | scaffold it. | it. | frozen it. | E_tot (Ry) | "
           "spread of last 3 E (Ry) | last dr2 | min rho | s |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in t["rows"]:
        out.append(f"| {r['solid']} | {r['step']:+d} | {r['functional']} | {r['start']} | {r['status']}"
                   f"{'' if not r.get('error') else ': ' + r['error']} | {g(r.get('scaffold_iterations'), 'd')} | {g(r.get('iterations'), 'd')} | "
                   f"{g(r.get('frozen_after_first'), 'd')} | {g(r.get('etot_ry'), '.7f')} | {g(r.get('etot_spread_last3_ry'), '.1e')} | "
                   f"{g(r.get('dr2_last'), '.1e')} | {g(r.get('min_rho'), '.2e')} | {g(r.get('seconds'), '.0f')} |")
    return "\n".join(out) + "\n"
