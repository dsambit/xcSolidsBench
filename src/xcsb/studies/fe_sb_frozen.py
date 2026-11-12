"""v4 Fe item 2 (user 2026-11-11): FROZEN spin-balance evaluation (strongxc-v4 E1) on the Fe FSM states, plus the spin-gradient
texture diagnostics. No SCF: the converged fields of every FSM point are read from its QE save (xcsb.host.qe_files) and checked
against the point's record before use (electrons per spin to 1e-8, the model's E_xc = sum w (e_x + e_c^SI) to 1e-7 Ry).

Reference function: strongxc-v4 dd8b5a517 explore/spin_balance/sb_model.e_xc_points (read-only), xchole pinned to the v4 line.
Variants (base | ref | omega | placement | R0):
    default   ueg_end, omega 90, base, R0 kin       (E1 as defined)
    PBE ref   PBE,     omega 90, base, R0 kin       (labelled row)
    R0 half   ueg_end, omega 90, base, R0 half      (labelled row)
    lam = 1   ueg_end, omega 0,  base, R0 kin       (no gate: diagnostic ceiling only; omega = 0 is excluded as a model)
Per point and variant: Delta E = sum w de (Ry), the frozen corrected E(m) = E_tot(m) + Delta E and E_xc(m) + Delta E; per curve: the
implied moment (spline minimum) and E'', E_xc'' at m = 0 (even fit over m <= 0.3) and at the implied moment; the gate diagnostics
(sb_model.GateAccumulator: |delta_m|-weighted lam by indicator bins, retained fractions, electron fraction with lam < 0.5) overall
and in the 3d region.

Texture (frozen, no model term; user 2026-11-11): density-weighted distributions, overall, in the 3d region and per density decade,
of zeta, the reduced spin gradients |grad zeta|/(2 k_F) and |grad zeta|/(2 k_s) (k_F = (3 pi^2 n)^(1/3), k_s = sqrt(4 k_F/pi), the
Thomas-Fermi wavevector), the alignment c_theta = grad zeta . grad n / (|grad zeta| |grad n|), and the iso correlation weight w_c'
(sb_model.indicators 'w_c_prime'), with the iso correlation pieces (He-pair, flat) along m from the records' fsm_parts.

3d region (diagnostic only): the points within R_3D of the Fe site (minimum image; bcc, atom at the origin)."""
from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np

VARIANTS = {"default": ("ueg_end", 90.0, "base", "kin"), "PBE_ref": ("PBE", 90.0, "base", "kin"),
            "R0_half": ("ueg_end", 90.0, "base", "half"), "lam1_no_gate": ("ueg_end", 0.0, "base", "kin")}
BASES = {"A": "SCH-SI-A_v3.1", "B": "SCH-SI-B_v3.1"}
R_3D = 1.3                     # bohr: radius of the 3d region around the Fe site (the 3d radial density peaks near 0.6 bohr)
DECADES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, np.inf)
TEXTURE_BINS = {
    "zeta": (-1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0),
    "t_zeta_kF": (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, np.inf),
    "t_zeta_ks": (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, np.inf),
    "c_theta": (-1.0, -0.9, -0.5, 0.0, 0.5, 0.9, 1.0),
    "w_c_prime": (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
}
TOL_N = 1e-8                   # electrons per spin, saved fields vs record
TOL_EXC_RY = 1e-7              # model E_xc on the saved fields vs the record
CHUNK = 4096                   # points per e_xc_points call (fixed shape: one compile per variant)
RY = 2.0


def sb_module():
    """Import the pinned v4 explore/spin_balance/sb_model (xchole pinned to the v4 line first)."""
    from ..functional.xchole import XCHOLE_LINE, import_pinned_xchole
    if XCHOLE_LINE != "v4":
        raise RuntimeError("the spin-balance reference needs XCSB_XCHOLE_LINE=v4")
    _, commit = import_pinned_xchole()
    root = os.path.dirname(os.path.abspath(os.environ["XCSB_XCHOLE_SRC"]))
    sys.path.insert(0, os.path.join(root, "explore", "spin_balance"))
    import sb_model
    return sb_model, commit, root


def fsm_points(run):
    """{base: {m: run dir}} of the converged FSM grid points (points with fsm_parts) of one run directory."""
    out = {b: {} for b in BASES}
    for d in sorted(os.listdir(run)):
        mm = re.match(r"Fe80-fsm-([0-9.]+)__hooked__(SCH-SI-[AB]_v3\.1)$", d)
        if not mm:
            continue
        p = os.path.join(run, d, "record.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        if r.get("status") != "ok" or "E_x_ry" not in (r.get("post") or {}).get("fsm_parts", {}):
            continue
        b = "A" if "SCH-SI-A" in mm.group(2) else "B"
        out[b][float(mm.group(1))] = os.path.join(run, d)
    return out


def _fields(d):
    from ..grid.spectral import SpectralGrid
    from ..host.evaluate import point_inputs
    from ..host.qe_files import saved_spin_fields
    r = json.load(open(os.path.join(d, "record.json")))
    info = {}
    rho, tau, shape = saved_spin_fields(d, info=info)
    g = SpectralGrid(np.asarray(r["case"]["structure"]["lattice_bohr"]), shape, r["case"]["ecutrho"])
    x, grad = point_inputs(g, rho, tau, True, True)
    n_el = [g.integrate(rho[s]) for s in range(2)]
    ref = r["density_stats"]["n_electrons"]
    if max(abs(a - b) for a, b in zip(n_el, ref)) > TOL_N:
        raise RuntimeError(f"{d}: saved fields have {n_el} electrons per spin, the record {ref}")
    return r, g, x, info


def _region(g):
    """Minimum-image distance of every grid point to the Fe site at the origin (bohr), (N,)."""
    fr = np.stack(np.meshgrid(*[np.arange(n) / n for n in g.shape], indexing="ij")).reshape(3, -1)
    best = np.full(fr.shape[1], np.inf)
    for sh in np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij")).reshape(3, -1).T:
        c = (fr - np.round(fr) + sh[:, None]).T @ g.lattice
        best = np.minimum(best, np.linalg.norm(c, axis=1))
    return best


def _hist(x, w, edges):
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    h = np.bincount(idx, weights=w, minlength=len(edges) - 1)
    return (h / h.sum()).tolist() if h.sum() else None


def _wquant(x, w, qs=(0.1, 0.5, 0.9)):
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w.sum()
    return [float(x[o][np.searchsorted(c, q)]) for q in qs]


def texture(g, x, wcp, r3d):
    n = x.rho.sum(0)
    gn = x.grad.sum(0)
    zeta = (x.rho[0] - x.rho[1]) / n
    gz = (x.grad[0] - x.grad[1]) / n - zeta[None] * gn / n
    az, agn = np.linalg.norm(gz, axis=0), np.linalg.norm(gn, axis=0)
    kf = (3 * np.pi ** 2 * n) ** (1 / 3)
    ks = np.sqrt(4 * kf / np.pi)
    with np.errstate(invalid="ignore", divide="ignore"):
        cth = np.where((az > 0) & (agn > 0), np.sum(gz * gn, axis=0) / (az * agn), 0.0)
    q = {"zeta": zeta, "t_zeta_kF": az / (2 * kf), "t_zeta_ks": az / (2 * ks), "c_theta": cth, "w_c_prime": wcp}
    w = g.weight * n
    regions = {"all": np.ones_like(n, bool), "3d": r3d <= R_3D}
    for lo, hi in zip(DECADES[:-1], DECADES[1:]):
        regions[f"n[{lo:g},{hi:g})"] = (n >= lo) & (n < hi)
    out = {}
    for name, sel in regions.items():
        if not np.any(sel):
            continue
        out[name] = {"electrons": float(w[sel].sum()),
                     **{k: {"edges": list(TEXTURE_BINS[k]), "density_weighted_share": _hist(v[sel], w[sel], TEXTURE_BINS[k]),
                            "q10_q50_q90": _wquant(v[sel], w[sel]), "mean": float(np.average(v[sel], weights=w[sel]))}
                        for k, v in q.items()}}
    return out


def evaluate_point(SB, base, d, variants=VARIANTS):
    """All variants on one FSM point: Delta E, the model check, gate diagnostics (all, 3d) and the texture."""
    r, g, x, info = _fields(d)
    N = g.npts
    r3d = _region(g)
    mask3d = r3d <= R_3D
    n = x.rho.sum(0)
    w = np.full(N, g.weight)
    res = {"dir": d, "m": r["magnetization"]["total"], "etot_ry": r["energies_ry"]["etot"], "e_xc_record_ry": r["e_xc_final_ry"],
           "rel_imag": info.get("rel_imag"), "variants": {}}
    pad = (-N) % CHUNK
    args = [x.rho[0], x.rho[1], x.grad[0], x.grad[1], x.tau[0], x.tau[1], x.lapl[0], x.lapl[1], x.hess[0], x.hess[1]]
    args = [np.concatenate([a, np.zeros(a.shape[:-1] + (pad,))], axis=-1) if pad else a for a in args]
    wcp = None
    for vname, (ref, om, pl, r0f) in variants.items():
        acc, acc3 = SB.GateAccumulator(), SB.GateAccumulator()
        ex = ec_si = de = 0.0
        for c0 in range(0, N, CHUNK):
            sl = slice(c0, c0 + CHUNK)
            o = SB.e_xc_points(*[a[..., sl] for a in args], base=base, ref=ref, omega=om, placement=pl, indicators=True, r0_form=r0f)
            k = min(CHUNK, N - c0)
            o = {key: (np.asarray(v)[..., :k]) for key, v in o.items()}
            ww, nn = w[c0:c0 + k], n[c0:c0 + k]
            ex += float(ww @ o["e_x"])
            ec_si += float(ww @ o["e_c_si"])
            de += float(ww @ o["de"])
            acc.add(o, ww, nn)
            acc3.add(o, ww, nn, mask=mask3d[c0:c0 + k])
            if vname == "default":
                wcp = o["w_c_prime"] if wcp is None else np.concatenate([wcp, o["w_c_prime"]])
        exc = RY * (ex + ec_si)
        if abs(exc - r["e_xc_final_ry"]) > TOL_EXC_RY:
            raise RuntimeError(f"{d}: e_x + e_c^SI on the saved fields = {exc:.10f} Ry, record {r['e_xc_final_ry']:.10f}")
        res["variants"][vname] = {"delta_E_ry": RY * de, "E_x_ry": RY * ex, "E_c_si_ry": RY * ec_si, "gate_all": acc.summary(),
                                  "gate_3d": acc3.summary()}
    res["model_check_d_exc_ry"] = exc - r["e_xc_final_ry"]
    res["texture"] = texture(g, x, wcp, r3d)
    fp = r["post"]["fsm_parts"]
    res["pieces_ry"] = fp.get("pieces_ry")
    res["E_x_ry"], res["E_c_ry"] = fp.get("E_x_ry"), fp.get("E_c_ry")
    return res


def curves(points):
    """Per variant: the frozen corrected E(m), E_xc(m), implied moment and curvatures (meV/mu_B^2) from evaluate_point results."""
    from .fsm_curves import curv0, curv_at, minimum
    RYMEV = 13605.693122994
    ms = np.array(sorted(points))
    E = np.array([points[m]["etot_ry"] for m in ms])
    Exc = np.array([points[m]["E_x_ry"] + points[m]["E_c_ry"] for m in ms])
    out = {"m": ms.tolist()}
    for v in list(points[ms[0]]["variants"]) + ["model"]:
        dE = np.zeros_like(E) if v == "model" else np.array([points[m]["variants"][v]["delta_E_ry"] for m in ms])
        Ev, Xv = E + dE, Exc + dE
        mi, ei = minimum(ms, Ev)
        out[v] = {"delta_E_meV": (dE * RYMEV).tolist(), "E_minus_E0_meV": ((Ev - Ev[0]) * RYMEV).tolist(), "implied_moment": mi,
                  "curvature_meV_per_muB2": {
                      "E_m0": None if curv0(ms, Ev) is None else curv0(ms, Ev) * RYMEV,
                      "Exc_m0": None if curv0(ms, Xv) is None else curv0(ms, Xv) * RYMEV,
                      "E_implied": curv_at(ms, Ev, mi) * RYMEV, "Exc_implied": curv_at(ms, Xv, mi) * RYMEV}}
    return out


def crosscheck(SB, root, out):
    """sb_model.e_xc_points on the v4 testpoints (20 points) against the stored outputs of every variant key present; if the file
    has not been regenerated with R0 kin for both sets, the keys present are compared and the missing ones listed."""
    p = os.path.join(root, "explore", "spin_balance", "results", "testpoints_sb.json")
    d = json.load(open(p))
    P = d["points"]
    a = lambda k: np.array([q[k] for q in P], float)                       # noqa: E731
    at = lambda k: np.array([q[k] for q in P], float).T                    # noqa: E731
    args = (a("rho_up"), a("rho_dn"), at("grad_up"), at("grad_dn"), a("tau_up"), a("tau_dn"), a("lap_up"), a("lap_dn"),
            at("hess_up"), at("hess_dn"))
    res = {"file": p, "keys": {}}
    for key, stored in d["outputs"].items():
        parts = key.split("|")
        base, ref, om, pl = parts[0], parts[1], float(parts[2].replace("omega", "")), parts[3]
        r0f = parts[4].split("=")[1] if len(parts) > 4 else "half"
        o = SB.e_xc_points(*args, base=base, ref=ref, omega=om, placement=pl, indicators=False, r0_form=r0f)
        dev = max(float(np.max(np.abs(np.asarray(o[k]) - np.asarray(v)) / np.maximum(np.abs(np.asarray(v)), 1e-300)))
                  for k, v in stored.items() if k in o and k != "v")
        res["keys"][key] = dev
    res["max_rel_dev"] = max(res["keys"].values())
    res["regenerated_with_kin_for_both"] = all(f"{b}|{r}|omega90|base|R0=kin" in d["outputs"] for b in "AB" for r in ("ueg_end", "PBE"))
    json.dump(res, open(os.path.join(out, "crosscheck.json"), "w"), indent=1)
    return res


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="xcsb.studies.fe_sb_frozen")
    ap.add_argument("--fsm", action="append", required=True, help="FSM run directory (repeatable)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    SB, commit, root = sb_module()
    t0 = time.time()
    cc = crosscheck(SB, root, a.out)
    print(json.dumps({"crosscheck_max_rel_dev": cc["max_rel_dev"], "regenerated": cc["regenerated_with_kin_for_both"],
                      "seconds": time.time() - t0}), flush=True)
    summary = {"xchole_commit": commit, "variants": {k: list(v) for k, v in VARIANTS.items()}, "r_3d_bohr": R_3D, "bases": {}}
    for run in a.fsm:
        for b, pts in fsm_points(run).items():
            summary["bases"].setdefault(b, {})
            for m, d in sorted(pts.items()):
                p = os.path.join(a.out, f"{b}_m{m:g}.json")
                if os.path.exists(p):
                    summary["bases"][b][m] = json.load(open(p))
                    continue
                t = time.time()
                res = evaluate_point(SB, b, d)
                json.dump(res, open(p, "w"), indent=1, default=float)
                summary["bases"][b][m] = res
                print(json.dumps({"base": b, "m": m, "d_exc_ry": res["model_check_d_exc_ry"],
                                  "delta_E_meV": {k: v["delta_E_ry"] * 13605.693 for k, v in res["variants"].items()},
                                  "seconds": time.time() - t}), flush=True)
    out = {"xchole_commit": commit, "crosscheck": cc, "curves": {b: curves(pts) for b, pts in summary["bases"].items() if len(pts) >= 4},
           "per_point": {b: {f"{m:g}": {"delta_E_ry": {k: v["delta_E_ry"] for k, v in res["variants"].items()},
                                       "gate_all": {k: v["gate_all"] for k, v in res["variants"].items()},
                                       "gate_3d": {k: v["gate_3d"] for k, v in res["variants"].items()},
                                       "texture": res["texture"], "pieces_ry": res["pieces_ry"]}
                             for m, res in pts.items()} for b, pts in summary["bases"].items()}}
    json.dump(out, open(os.path.join(a.out, "sb_frozen_summary.json"), "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
