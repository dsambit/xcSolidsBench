"""M1 step 1: the cutoff ladder of bcc Fe and bcc V (DESIGN.md Sec. 6.2; user go 2026-11-11).

Cases (fixed experimental lattice, nspin 2 unless NM1, 20^3 Gamma-centred k, Gaussian degauss 0.01 Ry, conv_thr bench):

    M1-<El>-<state>-<ecutwfc>-d<dual>      El = Fe, V;  state = FM (starting_magnetization 0.5), NM (0.0, set explicitly),
                                            NM1 (nspin 1: exactly nonmagnetic; the E(FM) - E(NM) reference of Fe in the PBE scan)

Stage P (PBE alone): ecutwfc in SCAN_ECUT for Fe FM, Fe NM1, V NM, V FM; E_c^PBE per element = the smallest scan cutoff from
which every successive pair is within the F2/F3 criteria (moment 0.005 mu_B/atom, E(FM) - E(NM) 1 meV/atom).
Stage L: PBE, r2SCAN, SCH-SI-A:v3.1, SCH-SI-B:v3.1 on ecutwfc = r E_c^PBE, r in RUNGS, d = 4, plus d = 8 at r in RUNGS_D8;
states Fe FM, V NM, V FM. The top d = 4 rung of Fe FM and V NM carries the ripple-seeded restarts (SeedPlan): the rung's
own record the seeds of the high band, a second SCF of the same case (run-directory tag 'riplow') those of the low band, so
that each step (base SCF + 3 seeds) fits one 4 h allocation.

Diagnostics on every converged density (post): (b) the BR-concave electron fraction, sum_s int rho_s [z_s < Z_STAR] / N with
z_s = 3 Q_s / (2 pi^(2/3) rho_s^(5/3)), Q_s = [lap rho_s - 4 lam (tau_s - tau_W,s)] / 6 (Hartree tau), lam from the pinned
v3.1 records (theta.lam); a diagnostic of the density, not part of any model.

Ripple seeds (c), at the density level (Lemma 1 of the ripple note: a common orbital factor g, phi_i -> g phi_i):
    rho_s' = g^2 rho_s,   tau_s' = g^2 tau_s + (1/2) g grad g . grad rho_s + (1/2) rho_s |grad g|^2,
exact for a common factor (then tau_s' - tau_W,s' = g^2 (tau_s - tau_W,s)), both multiplied by N_s / N_s' (a constant factor
of the orbitals of spin s: the electron count of each spin is kept) and projected onto the ecutrho sphere (P);
    h(r) = sum_{G in star} cos(G . r),  g = 1 + c h,
the star = the largest full shell of reciprocal-lattice vectors with |G| <= kfrac sqrt(ecutwfc) (bohr^-1, ecutwfc in Ry; a
full shell is invariant under the point group, so QE's symmetrization keeps the seed), k = |G| of that shell, for two bands
kfrac in BANDS (high 0.8, low 0.4). Every seed has the peak modulation max |g - 1| = c N_star = eps, eps in EPS (user
2026-11-11), i.e. c = eps / N_star. (At one star the 'curvature' form c = A2/k^2 and the 'gradient' form c = A1/k give the
same g at equal peak modulation; the channels are separated by the wavenumber instead: the seed changes grad rho_s by O(eps k)
and lap rho_s by O(eps k^2), so the second-order energy of a seed of peak eps is eps^2 (a_c k^4 + a_g k^2 + O(k^0)) (the k^3 cross
term averages out: grad h and lap h are in quadrature), a_g from the gradient channel (von Weizsaecker term included), a_c
from the curvature channel; two wavenumbers at the same eps separate a_c and a_g, and the low band weights the gradient
channel four times more relative to the curvature one.)
"""
from __future__ import annotations

import glob
import itertools
import json
import os

import numpy as np

from ..bench.structures import BOHR_PER_ANGSTROM, bcc
from ..host.qe import Case

# static-lattice (zero-point-corrected) experimental lattice constants, a_static = a_measured - Delta_ZP (user 2026-11-11, Q10:
# every solids calculation): refdata archive entry solids/lattice_constants_static_2026-11-11, from the measured values
# (solids/lattice_constants_exp_2026-11-11: Fe 2.861, V 3.028) and the PBE zero-point expansions Delta_ZP = a(Corr.) - a(Uncorr.)
# (Fe 0.003, V 0.005) of Zhang, Reilly, Tkatchenko, Scheffler, New J. Phys. 20, 063020 (2018), Supplementary Material Table II
A_EXP_ANGSTROM = {"Fe": 2.858, "V": 3.023}
A_EXP_SOURCE = "refdata solids/lattice_constants_static_2026-11-11 (static lattice: measured - PBE zero-point expansion)"
KMESH = (20, 20, 20)
DEGAUSS = 0.01                   # Ry, Gaussian (settings.SMEARING)
START_M = {"FM": 0.5, "NM": 0.0}
FUNCTIONALS = ("PBE", "r2SCAN", "SCH-SI-A:v3.1", "SCH-SI-B:v3.1")
SCAN_ECUT = (40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 140.0)
SCAN_STATES = (("Fe", "FM"), ("Fe", "NM1"), ("V", "NM"), ("V", "FM"))
LADDER_STATES = (("Fe", "FM"), ("V", "NM"), ("V", "FM"))
RUNGS = (0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
RUNGS_D8 = (1.0, 2.0, 3.0)
RIPPLE_STATES = (("Fe", "FM"), ("V", "NM"))
CRIT_MOMENT = 0.005              # mu_B / atom
CRIT_DE_MEV = 1.0                # meV / atom
RY_MEV = 13605.693122994         # meV per Ry (CODATA 2018)
Z_STAR = -0.861
BANDS = {"high": 0.8, "low": 0.4}     # kfrac of the seed star: |G| <= kfrac sqrt(ecutwfc)
EPS = (1e-3, 1e-2, 5e-2)              # peak modulation max |g - 1|


# ---- cases ------------------------------------------------------------------------------------------------------------------
def fmt_ecut(e: float) -> str:
    return f"{e:g}"


def case_name(el, state, ecut, dual=4):
    return f"M1-{el}-{state}-{fmt_ecut(ecut)}-d{int(dual)}"


def parse_case(name):
    """(El, state, ecutwfc, dual, k): 'M1-Fe-FM-80-d4'; an optional suffix '-k<n>' (n^3 k; the phase-0 checks only)."""
    parts = name.split("-")
    k = KMESH[0]
    if len(parts) == 6 and parts[5].startswith("k"):
        k = int(parts[5][1:])
        parts = parts[:5]
    if len(parts) != 5:
        raise KeyError(f"not an M1 case name: {name!r}")
    tag, el, state, ecut, d = parts
    if tag != "M1" or el not in A_EXP_ANGSTROM or state not in ("FM", "NM", "NM1") or not d.startswith("d"):
        raise KeyError(f"not an M1 case name: {name!r}")
    return el, state, float(ecut), int(d[1:]), k


def get_case(name) -> Case:
    el, state, ecut, dual, k = parse_case(name)
    s = bcc(el, A_EXP_ANGSTROM[el] * BOHR_PER_ANGSTROM)
    if state == "NM1":
        return Case(s, ecutwfc=ecut, kmesh=(k, k, k), occupations="smearing", degauss=DEGAUSS, nspin=1, dual=dual,
                    conv_class="bench")
    return Case(s, ecutwfc=ecut, kmesh=(k, k, k), occupations="smearing", degauss=DEGAUSS, nspin=2, dual=dual,
                starting_magnetization={el: START_M[state]}, conv_class="bench")


# ---- (b) BR-concave fraction ---------------------------------------------------------------------------------------------------
def record_lambdas():
    """{set name: theta.lam} of the pinned v3.1 records (sha256-checked)."""
    import hashlib

    from ..functional.xchole import RECORD_SHA256, XcholePinError, import_pinned_xchole
    import_pinned_xchole()
    from xchole.sch_si import params
    out = {}
    for name, (fname, sha) in RECORD_SHA256.items():
        raw = open(os.path.join(params.PARAMS_DIR, fname), "rb").read()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise XcholePinError(f"{fname}: sha256 mismatch")
        out[name] = float(json.loads(raw)["best"]["theta"]["lam"])
    return out


def concave_fraction(grid, rho, tau, lam):
    """sum_s int rho_s [z_s < Z_STAR] / N and the per-spin parts, on the grid (rho_s > 0 required)."""
    if np.any(rho <= 0):
        raise ValueError("concave_fraction: rho_s <= 0 at some grid point")
    grad, lapl, _ = grid.derivatives(rho, laplacian=True, hessian=False)
    tw = np.sum(grad ** 2, axis=1) / (8.0 * rho)
    q = (lapl - 4.0 * lam * (tau - tw)) / 6.0
    z = 3.0 * q / (2.0 * np.pi ** (2.0 / 3.0) * rho ** (5.0 / 3.0))
    n = grid.integrate(rho.sum(0))
    part = [grid.integrate(rho[s] * (z[s] < Z_STAR)) / n for s in range(2)]
    return {"lam": lam, "z_star": Z_STAR, "fraction": float(sum(part)), "fraction_by_spin": part,
            "z_min_by_spin": [float(z[s].min()) for s in range(2)]}


def post_diag(grid, rho, tau, lams=None):
    lams = lams or record_lambdas()
    out = {"concave": {}}
    for lam in sorted(set(lams.values())):
        out["concave"][f"{lam:.12g}"] = dict(concave_fraction(grid, rho, tau, lam),
                                            sets=sorted(k for k, v in lams.items() if v == lam))
    return out


# ---- (c) ripple seeds ----------------------------------------------------------------------------------------------------------
def cubic_ops():
    ops = []
    for p in itertools.permutations(range(3)):
        for sg in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            for i in range(3):
                m[i, p[i]] = sg[i]
            ops.append(m)
    return ops


def star(grid, ecutwfc, kfrac):
    """The largest full shell of reciprocal-lattice vectors with 0 < |G| <= kfrac sqrt(ecutwfc): (Miller indices (N, 3), G (N, 3),
    k). Raises unless the shell is invariant under the cubic point group (M1 cells are cubic) and inside the ecutrho sphere."""
    kmax2 = kfrac ** 2 * ecutwfc
    g2 = grid.G2
    sel = (g2 > 0) & (g2 <= kmax2 * (1 + 1e-12))
    if not np.any(sel):
        raise ValueError(f"no reciprocal-lattice vector with |G| <= {np.sqrt(kmax2):.3f}")
    gmax2 = float(g2[sel].max())
    shell = np.argwhere(np.abs(g2 - gmax2) <= 1e-9 * gmax2)
    ints = [np.fft.fftfreq(n, 1.0 / n).astype(int) for n in grid.shape]
    miller = np.array([[ints[a][i[a]] for a in range(3)] for i in shell])
    gv = miller @ grid.recip
    key = {tuple(np.round(v, 8)) for v in gv}
    for op in cubic_ops():
        for v in gv:
            if tuple(np.round(op @ v, 8)) not in key:
                raise ValueError("the shell is not invariant under the cubic point group")
    if gmax2 > grid.ecutrho:
        raise ValueError("the star lies outside the ecutrho sphere")
    return miller, gv, float(np.sqrt(gmax2))


def ripple_h(grid, gv):
    """h = sum_G cos(G . r) and grad h = -sum_G G sin(G . r) on the grid."""
    r = grid.points()
    ph = np.tensordot(gv, r, axes=(1, 0))                   # (N, n1, n2, n3)
    h = np.cos(ph).sum(0)
    gh = -np.tensordot(gv.T, np.sin(ph), axes=(1, 0))       # (3, n1, n2, n3)
    return h, gh


def common_factor(grid, rho, tau, c, h, gh):
    """rho', tau' of phi_i -> g phi_i, g = 1 + c h (Lemma 1), before normalization and projection."""
    g = 1.0 + c * h
    dg = c * gh
    grad, _, _ = grid.derivatives(rho, laplacian=False, hessian=False)
    r1 = g ** 2 * rho
    t1 = g ** 2 * tau + 0.5 * g * np.sum(dg[None] * grad, axis=1) + 0.5 * rho * np.sum(dg ** 2, axis=0)[None]
    return r1, t1


def seeded_fields(grid, rho, tau, c, h, gh):
    """The seeded fields: common_factor, times N_s / N_s' per spin (the electron count of each spin kept), projected onto the
    ecutrho sphere."""
    r1, t1 = common_factor(grid, rho, tau, c, h, gh)
    for s in range(2):
        f = grid.integrate(rho[s]) / grid.integrate(r1[s])
        r1[s] *= f
        t1[s] *= f
    return grid.project(r1), grid.project(t1)


class SeedPlan:
    """The ripple seeds of one SCF: for each band its star at this ecutwfc, the peak modulations EPS."""

    def __init__(self, ecutwfc, eps=EPS, bands=tuple(BANDS)):
        self.ecutwfc, self.eps, self.bands = float(ecutwfc), tuple(eps), tuple(bands)
        for b in self.bands:
            if b not in BANDS:
                raise ValueError(f"band {b!r}: one of {sorted(BANDS)}")
        self._cache = {}

    def _star(self, grid, band):
        if band not in self._cache:
            miller, gv, k = star(grid, self.ecutwfc, BANDS[band])
            h, gh = ripple_h(grid, gv)
            self._cache[band] = (miller, gv, k, h, gh)
        return self._cache[band]

    def specs(self):
        return [{"label": f"{b}-{e:g}", "band": b, "eps": e} for b in self.bands for e in self.eps]

    def coefficient(self, grid, spec):
        """c of g = 1 + c h: max |g - 1| = c N_star = eps (h = N_star at r = 0)."""
        return spec["eps"] / len(self._star(grid, spec["band"])[1])

    def describe(self, grid):
        bands = {}
        for b in self.bands:
            miller, gv, k, _, _ = self._star(grid, b)
            bands[b] = {"kfrac": BANDS[b], "k_bohr_inv": k, "kmax_bohr_inv": BANDS[b] * np.sqrt(self.ecutwfc),
                        "n_star": int(len(gv)), "miller": miller.tolist()}
        return {"bands": bands, "seeds": [dict(sp, c=self.coefficient(grid, sp)) for sp in self.specs()],
                "g": "g = 1 + c sum_star cos(G.r), max |g - 1| = eps",
                "tau_rule": "tau' = g^2 tau + g grad g . grad rho / 2 + rho |grad g|^2 / 2 (exact for a common orbital factor)",
                "normalization": "rho', tau' times N_s / N_s' per spin, then projected onto the ecutrho sphere"}

    def apply(self, grid, rho, tau, spec):
        _, _, _, h, gh = self._star(grid, spec["band"])
        r1, t1 = seeded_fields(grid, rho, tau, self.coefficient(grid, spec), h, gh)
        if np.any(r1 <= 0):
            raise ValueError(f"seeded density not positive (min {r1.min():.3e})")
        return r1, t1

    def amplitudes(self, grid, base, seeded, final, band):
        """Star components of rho_s, lap rho_s and |grad rho_s| (Fourier coefficients, f = sum_G f_G e^{iG.r}): the
        star-mean |f_G| of the base, seeded and final fields, the rms seeded and final changes against the base, and
        ratio = Re <d_final, d_seed> / <d_seed, d_seed> (0: the seed decayed; 1: it persisted; > 1: it grew)."""
        miller = self._star(grid, band)[0]
        idx = tuple((miller % np.array(grid.shape)).T)

        def comps(rho):
            grad, lapl, _ = grid.derivatives(rho, laplacian=True, hessian=False)
            fields = {"rho": rho, "lapl_rho": lapl, "abs_grad_rho": np.sqrt(np.sum(grad ** 2, axis=1))}
            return {k: np.stack([grid.fft(v[s])[idx] / grid.npts for s in range(2)]) for k, v in fields.items()}

        cb, cs, cf = comps(base[0]), comps(seeded[0]), comps(final[0])
        out = {}
        for key in cb:
            ds, df = cs[key] - cb[key], cf[key] - cb[key]
            rows = []
            for s in range(2):
                den = float(np.sum(np.abs(ds[s]) ** 2))
                rows.append({"base": float(np.mean(np.abs(cb[key][s]))), "seeded": float(np.mean(np.abs(cs[key][s]))),
                             "final": float(np.mean(np.abs(cf[key][s]))),
                             "d_seed_rms": float(np.sqrt(np.mean(np.abs(ds[s]) ** 2))),
                             "d_final_rms": float(np.sqrt(np.mean(np.abs(df[s]) ** 2))),
                             "ratio": float(np.real(np.sum(df[s] * np.conj(ds[s]))) / den) if den > 0 else None})
            out[key] = rows
        return out


# ---- run directories and records -------------------------------------------------------------------------------------------------
def scf_root(run):
    return os.path.join(run, "scf")


def record(run, case, functional, tag=""):
    from ..tiers import run_dir
    p = os.path.join(run_dir(scf_root(run), case, "hooked", functional) + (f"__{tag}" if tag else ""), "record.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _ok(r):
    return r is not None and r.get("status") == "ok"


def summary(r):
    """E (Ry), moment, absolute magnetization, iterations of an ok record."""
    return {"etot_ry": r["energies_ry"]["etot"], "m": r["magnetization"]["total"], "abs_m": r["magnetization"]["absolute"],
            "iterations": r["iterations"], "carrier_iterations": r["carrier"]["iterations"], "seconds": r.get("seconds")}


# ---- stage P: E_c^PBE ------------------------------------------------------------------------------------------------------------
W4_PROBE = ("Fe", "FM", 80.0)          # stage P also runs this scan case on a full node (tag w4): the measured full-node speedup


def stage_p_tasks():
    t = [{"id": f"P__{case_name(el, st, e)}", "case": case_name(el, st, e), "functional": "PBE", "width": 1, "stage": "P",
          "ripple": None, "tag": ""} for e in SCAN_ECUT for el, st in SCAN_STATES]
    c = case_name(*W4_PROBE)
    return t + [{"id": f"P__{c}__w4", "case": c, "functional": "PBE", "width": 4, "stage": "P", "ripple": None, "tag": "w4"}]


def scan_table(run):
    """Per element and scan cutoff: moments and E(FM) - E(NM) (meV/atom); None where a record is missing."""
    rows = {}
    for el in A_EXP_ANGSTROM:
        nm_state = "NM1" if el == "Fe" else "NM"
        rr = []
        for e in SCAN_ECUT:
            fm, nm = record(run, case_name(el, "FM", e), "PBE"), record(run, case_name(el, nm_state, e), "PBE")
            row = {"ecutwfc": e, "fm": summary(fm) if _ok(fm) else None, "nm": summary(nm) if _ok(nm) else None}
            if row["fm"] and row["nm"]:
                row["de_fm_nm_mev"] = (row["fm"]["etot_ry"] - row["nm"]["etot_ry"]) * RY_MEV
            rr.append(row)
        rows[el] = {"nm_state": nm_state, "rows": rr}
    return rows


def e_c_pbe(run):
    """{El: E_c} from the stage-P records, or None for an element whose scan is incomplete. Raises if a complete scan never
    meets the criteria (then the ladder is not built: report)."""
    out = {}
    for el, t in scan_table(run).items():
        rows = t["rows"]
        if any(r["fm"] is None or r["nm"] is None for r in rows):
            out[el] = None
            continue
        ok = []
        for a, b in zip(rows[:-1], rows[1:]):
            dm = max(abs(b["fm"]["m"] - a["fm"]["m"]), abs(b["nm"]["m"] - a["nm"]["m"]))
            de = abs(b["de_fm_nm_mev"] - a["de_fm_nm_mev"])
            ok.append(dm <= CRIT_MOMENT and de <= CRIT_DE_MEV)
        n = next((i for i in range(len(ok)) if all(ok[i:])), None)
        if n is None:
            raise RuntimeError(f"{el}: the PBE scan {SCAN_ECUT} never meets the F2/F3 criteria between successive cutoffs")
        out[el] = rows[n]["ecutwfc"]
    return out


# ---- stage L: the ladder -----------------------------------------------------------------------------------------------------------
def stage_l_tasks(ec):
    """Ordered: d = 4 rungs ascending (r = 3: full node, with the ripple seeds for RIPPLE_STATES), then d = 8."""
    tasks = []
    for d, rungs in ((4, RUNGS), (8, RUNGS_D8)):
        for r in rungs:
            for el, st in LADDER_STATES:
                e = round(r * ec[el], 6)
                for f in FUNCTIONALS:
                    rip = d == 4 and r == RUNGS[-1] and (el, st) in RIPPLE_STATES
                    base = {"case": case_name(el, st, e, d), "functional": f, "width": 4 if r == RUNGS[-1] else 1,
                            "stage": "L", "rung": r}
                    tasks.append(dict(base, id=f"L__{case_name(el, st, e, d)}__{f}", ripple="high" if rip else None,
                                      tag=""))
                    if rip:
                        tasks.append(dict(base, id=f"L__{case_name(el, st, e, d)}__{f}__riplow", ripple="low",
                                          tag="riplow"))
    return tasks


# ---- cost model (scheduling only) --------------------------------------------------------------------------------------------------
# QE time per SCF iteration on a quarter node (8 ranks x 4 threads): measured 3.7 s at 80 Ry, 8^3 k (29 irreducible points),
# nspin 2, npool 1 (Fe80 SCH-SI runs, results/fe_mag80, t_iter - t_hook); scaled with (ecutwfc/80)^1.5 and the irreducible k
# count (256 for 20^3). The hook: SCH-SI 5.3 s per round of 16384-point chunks (one chunk per rank and round), libxc 0.3 s.
# Iterations: carrier 20, hook 15 (PBE, r2SCAN) or 25 (SCH-SI), 20 per seed. A full node: QE part / 3.5.
COST = {"qe_s_80": 3.7 * 256 / 29, "hook_sch_s": 5.3, "hook_libxc_s": 0.3, "it_carrier": 20, "it_libxc": 15, "it_sch": 25,
        "it_seed": 20, "full_node_speedup": 3.5}


def fft_points(el, ecut, dual):
    """QE's dense FFT grid of the M1 bcc cell (2 n_max + 1 rounded up to 2^a 3^b 5^c 7^d)."""
    a = A_EXP_ANGSTROM[el] * BOHR_PER_ANGSTROM
    length = a * np.sqrt(3.0) / 2.0
    n = 2 * int(np.sqrt(dual * ecut) * length / (2.0 * np.pi)) + 1
    while True:
        m = n
        for p in (2, 3, 5, 7):
            while m % p == 0:
                m //= p
        if m == 1:
            return n ** 3
        n += 1


def estimate_s(task, cost=COST):
    el, st, ecut, dual, _ = parse_case(task["case"])
    sch = task["functional"].startswith("SCH-SI")
    ranks = 8 * task["width"]
    qe = cost["qe_s_80"] * (ecut / 80.0) ** 1.5 * (0.5 if st == "NM1" else 1.0)
    qe = qe / (cost["full_node_speedup"] if task["width"] == 4 else 1.0)
    chunks = -(-fft_points(el, ecut, dual) // 16384)
    rounds = -(-chunks // ranks)
    hook = rounds * (cost["hook_sch_s"] if sch else cost["hook_libxc_s"])
    it_h = cost["it_sch"] if sch else cost["it_libxc"]
    t = cost["it_carrier"] * qe + it_h * (qe + hook)
    if task.get("ripple"):
        t += len(EPS) * len(ripple_bands(task["ripple"])) * cost["it_seed"] * (qe + hook)
    return float(t)


def _qe_s_per_it(r):
    return (r["seconds"] - sum(h["t_hook_s"] for h in r["history"])) / (r["carrier"]["iterations"] + r["iterations"])


def phase0_record(run, functional, tag):
    from ..tiers import run_dir
    p = os.path.join(run_dir(os.path.join(run, "phase0"), PHASE0_CASE, "hooked", functional) + f"__{tag}", "record.json")
    return json.load(open(p)) if os.path.exists(p) else None


PHASE0_CASE = "M1-Fe-FM-40-d4-k8"


def calibrated_cost(run):
    """COST refitted from measured records, each constant only where its record exists ('measured' lists which):
    qe_s_80 (QE s per iteration, quarter node, pools) and the PBE iteration counts from the stage-P scan; full_node_speedup
    from the w4 probe; hook_sch_s (one round of chunks) from phase 0's SCH-SI-B npool-8 run. it_sch and it_seed stay assumed."""
    cost, measured = dict(COST), []
    vals, itc, ith = [], [], []
    for t in stage_p_tasks():
        if t["tag"]:
            continue
        r = record(run, t["case"], "PBE")
        if not _ok(r):
            continue
        el, st, ecut, _, _ = parse_case(t["case"])
        vals.append(_qe_s_per_it(r) / ((ecut / 80.0) ** 1.5 * (0.5 if st == "NM1" else 1.0)))
        itc.append(r["carrier"]["iterations"])
        ith.append(r["iterations"])
    if vals:
        cost.update(qe_s_80=float(np.median(vals)), it_carrier=float(np.median(itc)), it_libxc=float(np.median(ith)))
        measured += ["qe_s_80", "it_carrier", "it_libxc"]
    c = case_name(*W4_PROBE)
    q, f = record(run, c, "PBE"), record(run, c, "PBE", "w4")
    if _ok(q) and _ok(f):
        cost["full_node_speedup"] = _qe_s_per_it(q) / _qe_s_per_it(f)
        measured.append("full_node_speedup")
    r = phase0_record(run, "SCH-SI-B:v3.1", "np8")
    if _ok(r):
        cost["hook_sch_s"] = float(np.median([h["t_hook_s"] for h in r["history"]]))
        measured.append("hook_sch_s")
    cost["measured"] = measured
    return cost


def stage_p_report(run):
    """After stage P: EC.json, the measured per-SCF times (pools), the calibrated cost model and the node-hour estimate of
    stage L (ladder + seeds) at the measured E_c^PBE."""
    rows = []
    for t in stage_p_tasks():
        r = record(run, t["case"], "PBE", t["tag"])
        row = {"task": t["id"], "status": None if r is None else r.get("status")}
        if _ok(r):
            hk = [h["t_hook_s"] for h in r["history"]]
            row.update(seconds=r["seconds"], carrier_iterations=r["carrier"]["iterations"], hook_iterations=r["iterations"],
                       qe_s_per_iteration=_qe_s_per_it(r), hook_s_per_iteration=float(np.mean(hk)),
                       ranks=r.get("parallel", {}).get("ranks"), npool=r.get("parallel", {}).get("npool"))
        rows.append(row)
    p0 = {}
    for f in ("PBE", "SCH-SI-B:v3.1"):
        for tag in ("np1", "np8", "rip"):
            r = phase0_record(run, f, tag)
            if _ok(r):
                p0[f"{f}|{tag}"] = {"seconds": r["seconds"], "etot_ry": r["energies_ry"]["etot"], "m": r["magnetization"]["total"],
                                    "iterations": r["iterations"], "qe_s_per_iteration": _qe_s_per_it(r),
                                    "hook_s_per_iteration": float(np.mean([h["t_hook_s"] for h in r["history"]]))}
    pe = os.path.join(run, "EC.json")
    ec = json.load(open(pe)) if os.path.exists(pe) else None
    cost = calibrated_cost(run)
    est = None
    if ec:
        tl = [dict(t, est_s=estimate_s(t, cost)) for t in stage_l_tasks(ec["ec_pbe"])]
        nh = lambda ts: sum(t["est_s"] * t["width"] / 4 for t in ts) / 3600
        est = {"node_h_total": nh(tl), "node_h_seed_tasks": nh([t for t in tl if t["ripple"]]),
               "node_h_d8": nh([t for t in tl if t["case"].endswith("d8")]), "n_tasks": len(tl),
               "longest_step_h": max(t["est_s"] for t in tl) / 3600,
               "per_task_h": {t["id"]: t["est_s"] / 3600 for t in tl}}
    return {"ec": ec, "scf_times": rows, "phase0": p0, "cost": cost, "stage_l_estimate": est}


# ---- the task list the campaign runner consumes ---------------------------------------------------------------------------------------
def tasks(run):
    """Stage P, then (once every scan is complete) stage L; each task with its scheduling estimate. EC.json is written once.
    (The job selects the stage: the campaign runner's --stages filter.)"""
    tp = stage_p_tasks()
    cost = calibrated_cost(run)
    out = [dict(t, est_s=estimate_s(t, cost)) for t in tp]
    p = os.path.join(run, "EC.json")
    if os.path.exists(p):
        ec = json.load(open(p))["ec_pbe"]
    else:
        ec = e_c_pbe(run)
        if any(v is None for v in ec.values()):
            return out
        json.dump({"ec_pbe": ec, "scan": scan_table(run), "criteria": {"moment_muB": CRIT_MOMENT, "de_mev": CRIT_DE_MEV}},
                  open(p, "w"), indent=1)
    return out + [dict(t, est_s=estimate_s(t, cost)) for t in stage_l_tasks(ec)]


def ripple_bands(ripple):
    """The --ripple value: 'all' or one band."""
    if ripple == "all":
        return tuple(BANDS)
    if ripple not in BANDS:
        raise ValueError(f"ripple {ripple!r}: 'all' or one of {sorted(BANDS)}")
    return (ripple,)


def ripple_complete(r, ripple):
    sd = (r.get("ripple") or {}).get("seeds", [])
    return len(sd) == len(EPS) * len(ripple_bands(ripple)) and all("status" in x for x in sd)


def done(run, task):
    """The SCF record is ok and, for a ripple task, every seed has ended (ok or failed)."""
    r = record(run, task["case"], task["functional"], task.get("tag", ""))
    return _ok(r) and (not task.get("ripple") or ripple_complete(r, task["ripple"]))


def failed(run, task):
    r = record(run, task["case"], task["functional"], task.get("tag", ""))
    return r is not None and r.get("status") == "failed"


def command(run, task):
    """The xcsb CLI arguments of one task (run under srun with 8 * width ranks, one rank per pool)."""
    args = ["scf", "--case", task["case"], "--mode", "hooked", "--name", task["functional"], "--root", scf_root(run),
            "--npool", str(8 * task["width"]), "--m1-diag"]
    return (args + (["--ripple", task["ripple"]] if task.get("ripple") else [])
            + (["--tag", task["tag"]] if task.get("tag") else []))


# ---- tables (analysis) ---------------------------------------------------------------------------------------------------------
def table(run):
    ec = json.load(open(os.path.join(run, "EC.json")))["ec_pbe"] if os.path.exists(os.path.join(run, "EC.json")) else None
    out = {"ec_pbe": ec, "scan": scan_table(run), "ladder": {}, "d8": {}, "v_fm_minus_nm_mev": {}, "ripple": {}}
    if ec is None:
        return out
    for el, st in LADDER_STATES:
        for f in FUNCTIONALS:
            rows, prev = [], None
            for r in RUNGS:
                e = round(r * ec[el], 6)
                rec = record(run, case_name(el, st, e), f)
                row = {"r": r, "ecutwfc": e, "status": None if rec is None else rec.get("status")}
                if _ok(rec):
                    row.update(summary(rec))
                    conc = (rec.get("post") or {}).get("m1", {}).get("concave", {})
                    row["concave_fraction"] = {k: v.get("fraction") for k, v in conc.items()}
                    row["de_prev_mev"] = None if prev is None else (row["etot_ry"] - prev["etot_ry"]) * RY_MEV
                    row["dm_prev"] = None if prev is None else row["m"] - prev["m"]
                    prev = row
                else:
                    prev = None
                rows.append(row)
            out["ladder"][f"{el}-{st}|{f}"] = rows
            d8 = []
            for r in RUNGS_D8:
                e = round(r * ec[el], 6)
                a, b = record(run, case_name(el, st, e, 4), f), record(run, case_name(el, st, e, 8), f)
                if _ok(a) and _ok(b):
                    d8.append({"r": r, "ecutwfc": e, "e_d8_minus_d4_mev": (b["energies_ry"]["etot"] - a["energies_ry"]["etot"]) * RY_MEV,
                               "m_d8_minus_d4": b["magnetization"]["total"] - a["magnetization"]["total"]})
            out["d8"][f"{el}-{st}|{f}"] = d8
    for f in FUNCTIONALS:
        rows = []
        for r in RUNGS:
            e = round(r * ec["V"], 6)
            a, b = record(run, case_name("V", "FM", e), f), record(run, case_name("V", "NM", e), f)
            if _ok(a) and _ok(b):
                rows.append({"r": r, "ecutwfc": e, "de_mev": (a["energies_ry"]["etot"] - b["energies_ry"]["etot"]) * RY_MEV,
                             "m_fm_start": a["magnetization"]["total"], "m_nm_start": b["magnetization"]["total"]})
        out["v_fm_minus_nm_mev"][f] = rows
    for el, st in RIPPLE_STATES:
        e = round(RUNGS[-1] * ec[el], 6)
        for f in FUNCTIONALS:
            recs = [x for x in (record(run, case_name(el, st, e), f), record(run, case_name(el, st, e), f, "riplow"))
                    if x is not None and "ripple" in x]
            if not recs:
                continue
            rec = recs[0]
            rows = []
            for sd in (x for r_ in recs for x in r_["ripple"]["seeds"]):
                row = {"seed": sd["seed"]["label"], "status": sd.get("status"), "iterations": sd.get("iterations")}
                if sd.get("status") == "ok":
                    row.update(de_mev=sd["delta_etot_ry"] * RY_MEV, de_xc_seed_mev=sd["delta_e_xc_seed_ry"] * RY_MEV,
                               m_before=sd["magnetization_before"]["total"], m_after=sd["magnetization_after"]["total"],
                               ratio={k: [x["ratio"] for x in v] for k, v in sd["amplitudes"].items()})
                else:
                    row["error"] = sd.get("error")
                rows.append(row)
            out["ripple"][f"{el}-{st}|{f}"] = {"bands": {b: {k: v[k] for k in ("k_bohr_inv", "n_star")}
                                                         for r_ in recs for b, v in r_["ripple"]["plan"]["bands"].items()},
                                               "rows": rows}
    return out
