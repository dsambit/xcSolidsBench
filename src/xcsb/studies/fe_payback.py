"""Diagnostic (user 2026-11-10): how much does set B's spin-polarization payback act in bcc Fe?

Smoke settings for every variant (relative comparisons only, never benchmark values): bcc Fe, a = 2.866 A, ecutwfc 100 Ry,
ecutrho 400 Ry, 8^3 Gamma-centred k, Gaussian smearing 0.02 Ry, the converged-carrier start, conv_thr CONV_THR_BENCH.

1. Frozen attribution on B's converged density (`attribution`): per grid point, with xchole's own functions (read-only):
   e_xc of B and of B with ups_p = 0 (the payback's energy density is their difference), r_s, zeta, P_x, the gate
   g_P = P_x^2/(P_x^2 + P0^2) (1 + d c)/(1 + d), the zeta window W, the density factor f(r_s) = r_s^2/(r_s^2 + r_s0^2),
   binned by r_s, zeta and P_x and weighted by |m| = |rho_up - rho_dn|.
2. SCF variants of B (diagnostic theta edits of norm-independent fields, xcsb.functional.xchole.parse_variant).
"""
from __future__ import annotations

import numpy as np

from ..bench.structures import BOHR_PER_ANGSTROM, bcc
from ..host.qe import Case

A_FE = 2.866 * BOHR_PER_ANGSTROM


def case_fe100(nspin=2):
    kw = dict(starting_magnetization={"Fe": 0.5}) if nspin == 2 else {}
    return Case(bcc("Fe", A_FE), ecutwfc=100.0, kmesh=(8, 8, 8), occupations="smearing", degauss=0.02, nspin=nspin,
                conv_class="bench", **kw)


CASES = {"Fe100": lambda: case_fe100(2), "Fe100-NM": lambda: case_fe100(1)}

RS_EDGES = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, np.inf)
ZETA_EDGES = (-1.0, -0.75, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 0.75, 1.0)
PX_EDGES = (-1.0, -0.5, -0.25, -0.1, -0.03, 0.03, 0.1, 0.25, 0.5, 1.0)


def _pointwise(f, x):
    """Per-point (e_x, e_c, rs, zeta, P_x, g_P, W, f_rs) of an XcholeFunctional f at PointInputs x (xchole internals,
    exactly as jax_sc.ec_density / jax_si.make_energy_si evaluate them)."""
    import jax
    import jax.numpy as jnp
    from xchole.core import jax_model as jm
    from xchole.sch_si import jax_sc
    from xchole.sch_si.jax_si import exc_si
    m = f._model
    P = m.P
    Pc = jm._cparams(P)
    S = dict(f._S, Px=P)
    si = {k: float(v) for k, v in f._si.items()}

    def kernel(ru, rd, gu, gd, tu, td, lu, ld, Hu, Hd):
        ex, ec = exc_si(ru, rd, gu, gd, tu, td, lu, ld, Hu, Hd, P, Pc, S, si)
        rho = ru + rd
        rh, k, p2, T, L, gvn = jm._pp(rho, gu + gd, tu + td, lu + ld)
        z = (ru - rd) / (rh * (1 + jm.DELTA_Z))
        rs = (3 / (4 * jnp.pi)) ** (1 / 3) / k
        xu = jax_sc.payback_x(ru, gu, tu, lu, rho, P["lam"], S.get("eps_px", jax_sc.EPS_PX))
        xd = jax_sc.payback_x(rd, gd, td, ld, rho, P["lam"], S.get("eps_px", jax_sc.EPS_PX))
        c2 = jax_sc.cos2_reg(ru, rd, gu, gd)
        gp = jax_sc.payback_gate(xu, xd, P, c2, S.get("d_cos", jax_sc.D_COS), S.get("p0", jax_sc.P0_P))
        xcu, xcd = (jm.soft_cap_stable(v, P["x_cap"], P["cap_k"]) for v in (xu, xd))
        px = (xcu - xcd) / (xcu + xcd)
        W = jax_sc.payback_window(z, S.get("pb_zc"), S.get("pb_w"), S.get("pb_wmin", 0.0))
        frs = jax_sc.payback_rs_factor(rs, S.get("rs0", 0.0))
        return ex, ec, rs, z, px, gp, W, frs

    out = jax.jit(kernel)(x.rho[0], x.rho[1], x.grad[0], x.grad[1], x.tau[0], x.tau[1], x.lapl[0], x.lapl[1],
                          x.hess[0], x.hess[1])
    return [np.asarray(a) for a in out]


def _bins(key, edges, weights):
    idx = np.digitize(key, edges[1:-1])
    return [{k: float(np.sum(v[idx == b])) for k, v in weights.items()} | {"lo": edges[b], "hi": edges[b + 1]}
            for b in range(len(edges) - 1)]


def attribution(functional_name, grid, rho, tau):
    """Frozen attribution on one density (root only). functional_name: B (installed) or a B variant."""
    from ..functional import get
    from ..host.evaluate import point_inputs
    fB = get(functional_name)
    base, edits = functional_name.partition("@")[0], dict(fB.provenance.get("theta_edits", {}))
    edits["ups_p"] = 0.0
    f0 = get(base + "@" + ",".join(f"{k}={v}" for k, v in edits.items()))
    x, _ = point_inputs(grid, rho, tau, True, True)
    exB, ecB, rs, z, px, gp, W, frs = _pointwise(fB, x)
    ex0, ec0, *_ = _pointwise(f0, x)
    w = grid.weight
    ups = float(fB._S["ups_p"])
    e_pb = (exB + ecB) - (ex0 + ec0)                         # the payback's energy density (exchange identical: dex = 0)
    m = np.abs(x.rho[0] - x.rho[1])
    act = ups * gp * W * frs                                 # the payback weight factor multiplying (1 - w_c)
    wts = {"abs_m": w * m, "rho": w * (x.rho[0] + x.rho[1]), "e_payback_ry": 2.0 * w * e_pb,
           "abs_m_x_f": w * m * frs, "abs_m_x_gP": w * m * gp, "abs_m_x_W": w * m * W, "abs_m_x_act": w * m * act}
    M = float(np.sum(w * m))
    out = {
        "functional": functional_name, "ups_p": ups, "rs0": float(fB._S["rs0"]), "p0": float(fB._S["p0"]),
        "d_cos": float(fB._S["d_cos"]),
        "E_xc_ry": 2.0 * w * float(np.sum(exB + ecB)), "E_xc_ups0_ry": 2.0 * w * float(np.sum(ex0 + ec0)),
        "E_payback_ry": 2.0 * w * float(np.sum(e_pb)), "d_Ex_ry": 2.0 * w * float(np.sum(exB - ex0)),
        "abs_magnetization": M, "total_magnetization": float(np.sum(w * (x.rho[0] - x.rho[1]))),
        "m_weighted_mean": {"f_rs": float(np.sum(w * m * frs)) / M, "g_P": float(np.sum(w * m * gp)) / M,
                            "W_zeta": float(np.sum(w * m * W)) / M, "ups_gP_W_f": float(np.sum(w * m * act)) / M,
                            "rs": float(np.sum(w * m * rs)) / M, "abs_Px": float(np.sum(w * m * np.abs(px))) / M},
        "bins_rs": _bins(rs, RS_EDGES, wts), "bins_zeta": _bins(z, ZETA_EDGES, wts), "bins_Px": _bins(px, PX_EDGES, wts),
        "quantiles_abs_m_weighted": {},
    }
    order = np.argsort(rs)
    cm = np.cumsum((w * m)[order]) / M
    out["quantiles_abs_m_weighted"]["rs"] = {str(q): float(rs[order][np.searchsorted(cm, q)]) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    return out


RY_TO_MEV = 13605.693122994

ROWS = ("SCH-SI-B:v3.1", "SCH-SI-B:v3.1@ups_p=0", "SCH-SI-B:v3.1@ups_p=0.5", "SCH-SI-B:v3.1@rs0_pb=0",
        "SCH-SI-B:v3.1@rs0_pb=0.375", "SCH-SI-B:v3.1@rs0_pb=1.5", "SCH-SI-A:v3.1", "PBE")


def table(root):
    """Rows (installed B first): moment per atom (|total|, absolute), E_tot, E(FM) - E(NM), iterations. Every B variant's
    NM reference is B's NM run: the payback vanishes exactly at zeta = 0 (g_P ~ P_x^2 = 0); one variant's NM run checks it."""
    import json
    import os

    from ..tiers import run_dir

    def load(case, name):
        p = os.path.join(run_dir(root, case, "hooked", name), "record.json")
        return json.load(open(p)) if os.path.exists(p) else {"status": "missing"}

    nm = {n: load("Fe100-NM", n) for n in ("SCH-SI-B:v3.1", "SCH-SI-A:v3.1", "PBE", "SCH-SI-B:v3.1@rs0_pb=0")}
    rows = []
    for name in ROWS:
        r = load("Fe100", name)
        ref = nm["PBE"] if name == "PBE" else nm["SCH-SI-A:v3.1"] if name.startswith("SCH-SI-A") else nm["SCH-SI-B:v3.1"]
        row = {"functional": name, "status": r.get("status")}
        if r.get("status") == "ok":
            mg = r["magnetization"]
            row.update(m_total=abs(mg["total"]), m_abs=mg["absolute"], etot_ry=r["energies_ry"]["etot"],
                       iterations=r["iterations"], carrier_iterations=r.get("carrier", {}).get("iterations"))
            if ref.get("status") == "ok":
                row["E_FM_minus_NM_meV"] = (r["energies_ry"]["etot"] - ref["energies_ry"]["etot"]) * RY_TO_MEV
        rows.append(row)
    chk = {}
    if nm["SCH-SI-B:v3.1"].get("status") == "ok" and nm["SCH-SI-B:v3.1@rs0_pb=0"].get("status") == "ok":
        chk = {"NM_B_minus_NM_B_rs0_0_ry": nm["SCH-SI-B:v3.1"]["energies_ry"]["etot"]
               - nm["SCH-SI-B:v3.1@rs0_pb=0"]["energies_ry"]["etot"]}
    attr = load("Fe100", "SCH-SI-B:v3.1").get("post")
    return {"rows": rows, "nm_check": chk, "nm": {k: v.get("energies_ry", {}).get("etot") for k, v in nm.items()},
            "attribution_B": attr}
