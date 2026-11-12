"""GKS band gaps of AFM-II NiO and MnO (user 2026-11-11, v4 line): PBE, r2SCAN (reference functionals: computed once, banked),
r2SCAN-SB, r2SCAN-SB-kin (strongxc-v4 b2501727c, xchole line v4-sb). Reproduction of the Perdew group's statement that spin-
balanced r2SCAN improves the NiO and MnO gaps (private communication, no numbers). By spin-flip symmetry the term and its
potential vanish at spin-symmetric states, so only magnetic insulators can show it.

Cell: rock-salt parent, AFM-II order in the rhombohedral 4-atom cell (2 TM + 2 O; structures.rocksalt_afm2), the rhombohedral
distortion of the real low-temperature crystals IGNORED; nspin 2 with opposite starting moments on TM1 and TM2 (+-0.5, O 0),
fixed occupations with tot_magnetization 0 (an AFM insulator; QE needs it for fixed occupations with nspin 2), no Hubbard U,
nbnd = n_occ + N_EMPTY with diago_full_acc. SG15 Ni (18 e), Mn (15 e), O (6 e). Plain Broyden beta 0.3 (the global setting).

LATTICE: static-lattice experimental constants (Q10). The Zhang et al. NJP 2018 SM has neither NiO nor MnO: A_STATIC is EMPTY
and every case raises until sourced values are banked in the archive and entered here (open item).

Stage P: PBE on ecutwfc in SCAN_ECUT at k SCAN_K^3; E_c^PBE = the smallest cutoff from which every successive pair meets the
family criteria (F4: TM site moment 0.01 mu_B, GKS gap 0.02 eV, E_tot 1 meV per formula unit). Stage G (short, as MgO but not the
full grid, for cost): PBE at r E_c, r in RUNGS, at k SCAN_K^3, and at E_c with k in KMESHES (8^3 on a full node); selected: the
smallest r whose gap is within GAP_TOL of r = 2 (at k 4^3), and the smallest k within GAP_TOL of 8^3 (at r = 1).
Stage F: all four functionals at the selected settings (SETTINGS.json).

Reported per functional: the GKS gap (minimum over the SCF mesh: min CBM - max VBM over both spins, with the k points and spins
of the edges), the TM site moments (the integral of rho_up - rho_dn within R_SITE of each TM site, minimum image) and the cell's
absolute magnetization, SCF iterations."""
from __future__ import annotations

import json
import os

import numpy as np

from ..bench.structures import BOHR_PER_ANGSTROM, rocksalt_afm2
from ..host.qe import Case

A_STATIC = {}                   # {"NiO": (a_static Angstrom, archive entry), "MnO": (...)}: NOT SOURCED (open item)
TM = {"NiO": "Ni", "MnO": "Mn"}
ZVAL = {"Ni": 18, "Mn": 15, "O": 6}
N_EMPTY = 8
FUNCTIONALS = ("PBE", "r2SCAN", "r2SCAN-SB", "r2SCAN-SB-kin")
SCAN_ECUT = (60.0, 80.0, 100.0, 120.0, 140.0)
SCAN_K = 4
RUNGS = (1.0, 1.5, 2.0)
KMESHES = (4, 6, 8)
CRIT_MOMENT, CRIT_GAP, CRIT_DE_MEV = 0.01, 0.02, 1.0
GAP_TOL = 0.02
R_SITE = 2.0                    # bohr: the TM sphere of the site moments (stated; diagnostic)
RY_EV = 13.605693122994


def n_occ(mat):
    return (2 * ZVAL[TM[mat]] + 2 * ZVAL["O"]) // 2          # per spin (tot_magnetization 0)


def case_name(mat, ecut, k):
    return f"TMO-{mat}-{ecut:g}-k{int(k)}"


def parse_case(name):
    tag, mat, ecut, k = name.split("-")
    if tag != "TMO" or mat not in TM or not k.startswith("k"):
        raise KeyError(f"not a TMO case: {name!r}")
    return mat, float(ecut), int(k[1:])


def get_case(name) -> Case:
    mat, ecut, k = parse_case(name)
    if mat not in A_STATIC:
        raise KeyError(f"{mat}: no sourced static-lattice constant (Zhang et al. SM has no NiO/MnO; open item): not runnable")
    a, _ = A_STATIC[mat]
    tm = TM[mat]
    s = rocksalt_afm2(tm, "O", a * BOHR_PER_ANGSTROM)
    return Case(s, ecutwfc=ecut, kmesh=(k, k, k), occupations="fixed", nspin=2, tot_magnetization=0.0,
                starting_magnetization={tm + "1": 0.5, tm + "2": -0.5, "O": 0.0}, nbnd=n_occ(mat) + N_EMPTY, diago_full_acc=True,
                conv_class="bench")


# ---- gaps and site moments --------------------------------------------------------------------------------------------------
def gaps(rec, mat):
    """Minimum GKS gap over the mesh, both spins (QE lists the k points of spin up, then spin down), with the band edges."""
    et = np.asarray(rec["eigenvalues_ry"]) * RY_EV
    kf = np.asarray(rec["kpoints_frac"])
    nk = et.shape[1]
    if nk % 2 or kf.shape[0] != nk:
        raise ValueError(f"eigenvalues {et.shape}, k list {kf.shape}: expected the doubled LSDA list")
    no = n_occ(mat)
    vb, cb = et[no - 1], et[no]
    iv, ic = int(vb.argmax()), int(cb.argmin())
    spin = lambda i: "up" if i < nk // 2 else "down"      # noqa: E731
    return {"min_gap_ev": float(cb[ic] - vb[iv]), "vbm_ev": float(vb[iv]), "cbm_ev": float(cb[ic]),
            "vbm_k": kf[iv].tolist(), "vbm_spin": spin(iv), "cbm_k": kf[ic].tolist(), "cbm_spin": spin(ic),
            "direct_min_gap_ev": float(np.min(cb - vb))}


def site_moments(grid, rho, structure):
    """{label: integral of rho_up - rho_dn within R_SITE of the site} for the TM sites (minimum image)."""
    fr = np.stack(np.meshgrid(*[np.arange(n) / n for n in grid.shape], indexing="ij")).reshape(3, -1)
    m = (rho[0] - rho[1]).reshape(-1)
    out = {}
    for lab, f0 in zip(structure.labels, structure.frac):
        if lab[:-1] not in ("Ni", "Mn"):
            continue
        d = fr - np.asarray(f0)[:, None]
        best = np.full(d.shape[1], np.inf)
        for sh in np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij")).reshape(3, -1).T:
            c = (d - np.round(d) + sh[:, None]).T @ grid.lattice
            best = np.minimum(best, np.linalg.norm(c, axis=1))
        out[lab] = float(grid.weight * m[best <= R_SITE].sum())
    return {"r_site_bohr": R_SITE, "moments": out}


def post_moments(case_name_):
    def f(grid, rho, tau):
        return site_moments(grid, rho, get_case(case_name_).structure)
    return f


# ---- records and stages ------------------------------------------------------------------------------------------------------
def scf_root(run):
    return os.path.join(run, "scf")


def record(run, case, functional):
    from ..tiers import run_dir
    p = os.path.join(run_dir(scf_root(run), case, "hooked", functional), "record.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _ok(r):
    return r is not None and r.get("status") == "ok"


def _row(r, mat):
    sm = ((r.get("post") or {}).get("tmo") or {}).get("moments", {})
    return dict(gaps(r, mat), etot_ry=r["energies_ry"]["etot"], abs_m=r["magnetization"]["absolute"],
                m_total=r["magnetization"]["total"], site_moments=sm, iterations=r["iterations"],
                carrier_iterations=r["carrier"]["iterations"], seconds=r.get("seconds"))


def e_c_pbe(run, mat):
    rows = []
    for e in SCAN_ECUT:
        r = record(run, case_name(mat, e, SCAN_K), "PBE")
        if not _ok(r):
            return None, rows
        rows.append(dict(ecutwfc=e, **_row(r, mat)))
    ok = []
    for a, b in zip(rows[:-1], rows[1:]):
        dm = max(abs(abs(b["site_moments"][k]) - abs(a["site_moments"][k])) for k in a["site_moments"])
        ok.append(dm <= CRIT_MOMENT and abs(b["min_gap_ev"] - a["min_gap_ev"]) <= CRIT_GAP
                  and abs(b["etot_ry"] - a["etot_ry"]) * RY_EV * 1000 / 2 <= CRIT_DE_MEV)
    n = next((i for i in range(len(ok)) if all(ok[i:])), None)
    if n is None:
        raise RuntimeError(f"{mat}: the PBE scan never meets the F4 criteria between successive cutoffs")
    return rows[n]["ecutwfc"], rows


def grid_points():
    return sorted({(r_, SCAN_K) for r_ in RUNGS} | {(RUNGS[0], k) for k in KMESHES})


def selected(run, mat, ec):
    """(ecutwfc, k) from the PBE points of stage G, or None while incomplete."""
    g = {}
    for r_, k in grid_points():
        rec = record(run, case_name(mat, round(r_ * ec, 6), k), "PBE")
        if not _ok(rec):
            return None
        g[(r_, k)] = _row(rec, mat)["min_gap_ev"]
    r_sel = next(r_ for r_ in RUNGS if abs(g[(r_, SCAN_K)] - g[(RUNGS[-1], SCAN_K)]) <= GAP_TOL)
    k_sel = next(k for k in KMESHES if abs(g[(RUNGS[0], k)] - g[(RUNGS[0], KMESHES[-1])]) <= GAP_TOL)
    return {"ecutwfc": round(r_sel * ec, 6), "k": k_sel, "r": r_sel, "cutoff_converged": r_sel != RUNGS[-1],
            "k_converged": k_sel != KMESHES[-1], "grid_gaps_ev": {f"r{a:g}|k{b}": v for (a, b), v in g.items()}}


# cost (scheduling): QE ~32 s per SCF iteration at 100 Ry, 4^3 k (about 13 irreducible points), quarter node: the Fe 80 Ry
# timing (3.7 s, 29 k, 1 atom) scaled by ~14 for 4 atoms (plane waves x bands) and by the k count; ~100 iterations (carrier and
# hook; AFM is slow); the k count ~ (k/4)^3; a full node (8^3) ~3.5 x faster.
def _task(mat, ecut, k, f, stage):
    w = 4 if k >= 8 else 1
    est = 32.0 * 100 * (ecut / 100.0) ** 1.5 * (k / 4.0) ** 3 / (3.5 if w == 4 else 1.0)
    return {"id": f"{stage}__{case_name(mat, ecut, k)}__{f}", "case": case_name(mat, ecut, k), "functional": f, "width": w,
            "stage": stage, "est_s": est}


def tasks(run):
    t = []
    for mat in A_STATIC:
        t += [_task(mat, e, SCAN_K, "PBE", "P") for e in SCAN_ECUT]
        ec, _ = e_c_pbe(run, mat)
        if ec is None:
            continue
        t += [_task(mat, round(r_ * ec, 6), k, "PBE", "G") for r_, k in grid_points()
              if not (r_ == RUNGS[0] and k == SCAN_K)]
        p = os.path.join(run, f"SETTINGS_{mat}.json")
        if not os.path.exists(p):
            sel = selected(run, mat, ec)
            if sel is None:
                continue
            json.dump(dict(sel, ec_pbe=ec), open(p, "w"), indent=1)
        s = json.load(open(p))
        t += [_task(mat, s["ecutwfc"], s["k"], f, "F") for f in FUNCTIONALS]
    return t


def done(run, task):
    return _ok(record(run, task["case"], task["functional"]))


def failed(run, task):
    r = record(run, task["case"], task["functional"])
    return r is not None and r.get("status") == "failed"


def command(run, task):
    return ["scf", "--case", task["case"], "--mode", "hooked", "--name", task["functional"], "--root", scf_root(run),
            "--npool", "1", "--tmo-diag"]


def table(run):
    out = {"a_static": A_STATIC, "r_site_bohr": R_SITE, "materials": {}}
    for mat in A_STATIC:
        ec, scan = e_c_pbe(run, mat)
        p = os.path.join(run, f"SETTINGS_{mat}.json")
        sel = json.load(open(p)) if os.path.exists(p) else None
        res = {}
        if sel:
            for f in FUNCTIONALS:
                r = record(run, case_name(mat, sel["ecutwfc"], sel["k"]), f)
                res[f] = _row(r, mat) if _ok(r) else {"status": None if r is None else r.get("status")}
        out["materials"][mat] = {"ec_pbe": ec, "scan": scan, "settings": sel, "results": res}
    return out
