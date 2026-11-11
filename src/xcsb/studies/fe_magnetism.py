"""Diagnostic (user 2026-11-10): which pieces of SCH-SI carry bcc Fe's magnetization energy?

Frozen split on a converged FM density (A or B): E_xc = sum of pieces, each evaluated with xchole's own functions (read-only),

  exchange (sum over spins, xchole.sch_si.exchange.spin_pieces):
    X_iso_BRHe  = w_x (1 - g_s) e_BR-He       the BR / He-family hole (lambda_eff, Q)
    X_iso_shell = w_x g_s e_shell             the floored Gaussian shell (eps, c_min, gates)
    X_UEG       = (1 - w_x) e_x^UEG
  correlation (xchole.sch_si.jax_sc.ec_density, base):
    C_iso_He    = w_c (1 - c) e_He            He-pair hole
    C_iso_flat  = w_c c e_flat                strictly correlated (flat) hole
    C_payback   = (w_c' - w_c) (e_iso - e_UEG)
    C_UEG       = (1 - w_c) e_c^UEG
  strong-coupling correction (xchole.sch_si.jax_si.exc_si):
    SI          = Lambda' (e_T - e_x - e_c)

and its magnetization dependence: Delta piece = piece[FM density] - piece[the same density depolarized: rho_s = rho/2,
grad rho_s, tau_s, lap rho_s, Hessian_s halved]. The pieces sum to e_xc of the model at every point (checked).
"""
from __future__ import annotations

import numpy as np

from ..bench.structures import BOHR_PER_ANGSTROM, bcc
from ..host.qe import Case

A_FE = 2.866 * BOHR_PER_ANGSTROM
PIECES = ("X_iso_BRHe", "X_iso_shell", "X_UEG", "C_iso_He", "C_iso_flat", "C_payback", "C_UEG", "SI")


def case_fe80(nspin=2):
    kw = dict(starting_magnetization={"Fe": 0.5}) if nspin == 2 else {}
    return Case(bcc("Fe", A_FE), ecutwfc=80.0, kmesh=(8, 8, 8), occupations="smearing", degauss=0.02, nspin=nspin,
                conv_class="bench", **kw)


CASES = {"Fe80": lambda: case_fe80(2)}


def case_fe80_fsm(m):
    """Fixed spin moment: QE's tot_magnetization = m (mu_B per cell; two Fermi levels), otherwise case_fe80."""
    return Case(bcc("Fe", A_FE), ecutwfc=80.0, kmesh=(8, 8, 8), occupations="smearing", degauss=0.02, nspin=2,
                starting_magnetization={"Fe": 0.5}, tot_magnetization=float(m), conv_class="bench")


def _kernel(f):
    import jax
    import jax.numpy as jnp
    from xchole.core import jax_model as jm
    from xchole.sch_si import jax_sc
    from xchole.sch_si.exchange import spin_pieces
    from xchole.sch_si.jax_si import exc_si, weights
    from xchole.sch_si.target import target_jax
    P = f._model.P
    Pc = jm._cparams(P)
    S = dict(f._S, Px=P)
    si = {k: float(v) for k, v in f._si.items()}

    def k(ru, rd, gu, gd, tu, td, lu, ld, Hu, Hd):
        out = {}
        xs = []
        for (r, g, t, l) in ((ru, gu, tu, lu), (rd, gd, td, ld)):
            ex_s, pc = spin_pieces(r, g, t, l, P)
            xs.append((ex_s, pc))
        for (_, pc), sp in zip(xs, ("up", "dn")):
            out[f"e_isox_{sp}"] = (1 - pc["gs"]) * pc["e_br"] + pc["gs"] * pc["e_sh"]
            out[f"e_defx_{sp}"] = pc["e_def"]
        out["X_iso_BRHe"] = sum(pc["w"] * (1 - pc["gs"]) * pc["e_br"] for _, pc in xs)
        out["X_iso_shell"] = sum(pc["w"] * pc["gs"] * pc["e_sh"] for _, pc in xs)
        out["X_UEG"] = sum((1 - pc["w"]) * pc["e_def"] for _, pc in xs)
        ex = sum(e for e, _ in xs)
        # base correlation, as jax_sc.ec_density
        w0, e_he, e_ueg = jax_sc.ueg_terms_reduced(ru, rd, gu, gd, tu, td, lu, ld, Pc)
        rho = ru + rd
        rh, kk, p2, T, L, gvn = jm._pp(rho, gu + gd, tu + td, lu + ld)
        z = (ru - rd) / (rh * (1 + jm.DELTA_Z))
        rs = (3 / (4 * jnp.pi)) ** (1 / 3) / kk
        su, _, _ = jax_sc.spin_switch_x(ru, gu, tu, lu, P)
        sd_, _, _ = jax_sc.spin_switch_x(rd, gd, td, ld, P)
        xu = jax_sc.payback_x(ru, gu, tu, lu, rho, P["lam"], S.get("eps_px", jax_sc.EPS_PX))
        xd = jax_sc.payback_x(rd, gd, td, ld, rho, P["lam"], S.get("eps_px", jax_sc.EPS_PX))
        c2 = jax_sc.cos2_reg(ru, rd, gu, gd)
        pb = S.get("ups_p", 0.0) * jax_sc.payback_gate(xu, xd, P, c2, S.get("d_cos", jax_sc.D_COS), S.get("p0", jax_sc.P0_P)) \
            * jax_sc.payback_window(z, S.get("pb_zc"), S.get("pb_w"), S.get("pb_wmin", 0.0)) \
            * jax_sc.payback_rs_factor(rs, S.get("rs0", 0.0))
        w1 = w0 + (1 - w0) * pb
        e_flat = rho * jax_sc.eps_rational(rs, z, *jax_sc.slab_B(kk, *jax_sc.hess_invariants(rh, gvn, Hu + Hd),
                                                                    jax_sc.HESS_EPS), jax_sc.KAPPA)
        q = 1 - 1 / jnp.sqrt(1 + rs / S["r_q"])
        qs = rs / (rs + S["r_q"])
        sc = (ru * su + rd * sd_) / rh
        cc = (1 - sc) * q + sc * qs
        e_iso_c = (1 - cc) * e_he + cc * e_flat
        out["C_iso_He"] = w0 * (1 - cc) * e_he
        out["C_iso_flat"] = w0 * cc * e_flat
        out["C_payback"] = (w1 - w0) * (e_iso_c - e_ueg)
        out["C_UEG"] = (1 - w0) * e_ueg
        ec = w1 * e_iso_c + (1 - w1) * e_ueg
        lam, _, wt, _ = weights(ru, rd, gu, gd, tu, td, lu, ld, Hu, Hd, P, si)
        eT = target_jax(ru, rd, gu, gd, lu, ld, Hu, Hd, wt, P, Pc, S, si)
        out["SI"] = lam * (eT - ex - ec)
        out.update(pb=pb, e_isoc=e_iso_c, e_uegc=e_ueg, eT=eT)
        exm, ecm = exc_si(ru, rd, gu, gd, tu, td, lu, ld, Hu, Hd, P, Pc, S, si)
        out["model_total"] = exm + ecm
        out["w_x_up"], out["w_x_dn"] = xs[0][1]["w"], xs[1][1]["w"]
        out["lam_eff_up"], out["lam_eff_dn"] = xs[0][1]["lam"], xs[1][1]["lam"]
        out["w_c"] = w0
        out["w_c_prime"] = w1
        out["Lambda_prime"] = lam
        out["rs"] = rs
        return out

    return jax.jit(k)


def _depolarized(x):
    from ..functional.contract import PointInputs
    h = lambda a: np.stack([0.5 * a.sum(0)] * 2)      # noqa: E731
    return PointInputs(h(x.rho), h(x.grad), h(x.tau), h(x.lapl), h(x.hess))


def split(functional_name, grid, rho, tau):
    from ..functional import get
    from ..host.evaluate import point_inputs
    f = get(functional_name)
    kern = _kernel(f)
    x, _ = point_inputs(grid, rho, tau, True, True)
    args = lambda y: (y.rho[0], y.rho[1], y.grad[0], y.grad[1], y.tau[0], y.tau[1], y.lapl[0], y.lapl[1],   # noqa: E731
                      y.hess[0], y.hess[1])
    fm = {k: np.asarray(v) for k, v in kern(*args(x)).items()}
    nm = {k: np.asarray(v) for k, v in kern(*args(_depolarized(x))).items()}
    w = grid.weight
    ry = lambda a: 2.0 * w * float(np.sum(a))          # noqa: E731
    m = np.abs(x.rho[0] - x.rho[1])
    M = float(np.sum(w * m))
    out = {"functional": functional_name, "abs_magnetization": M,
           "sum_pieces_minus_model_ry": ry(sum(fm[p] for p in PIECES) - fm["model_total"]),
           "E_ry": {p: ry(fm[p]) for p in PIECES + ("model_total",)},
           "E_depolarized_ry": {p: ry(nm[p]) for p in PIECES + ("model_total",)},
           "Delta_mag_ry": {p: ry(fm[p] - nm[p]) for p in PIECES + ("model_total",)},
           "m_weighted_mean": {k: float(np.sum(w * m * fm[k])) / M
                               for k in ("w_x_up", "w_x_dn", "lam_eff_up", "lam_eff_dn", "w_c", "w_c_prime", "Lambda_prime", "rs")}}
    out["descriptors"] = descriptors(x, fm, w)
    # magnetization energy by r_s region (where does each piece's Delta live)
    edges = (0.0, 0.5, 0.75, 1.0, 1.5, np.inf)
    idx = np.digitize(fm["rs"], edges[1:-1])
    out["Delta_mag_by_rs_ry"] = [{"lo": edges[b], "hi": edges[b + 1], "abs_m_share": float(np.sum(w * m[idx == b])) / M,
                                  **{p: ry((fm[p] - nm[p])[idx == b]) for p in PIECES}} for b in range(len(edges) - 1)]
    return out


def _wq(v, wt, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    v, wt = np.asarray(v).reshape(-1), np.asarray(wt).reshape(-1)
    o = np.argsort(v)
    cw = np.cumsum(wt[o]) / np.sum(wt)
    return {"mean": float(np.sum(wt * v) / np.sum(wt)),
            **{f"q{int(q * 100)}": float(v[o][min(np.searchsorted(cw, q), v.size - 1)]) for q in qs}}


def descriptors(x, fm, w):
    """Distributions (weighted by |m| and by rho) of r_s, s, s_sigma, q, alpha (total and per spin), w_x, w_c, zeta, and 2-D
    |m| shares. These are SG15 valence (+ semicore) pseudo-densities. Conventions: k_F = (3 pi^2 rho)^(1/3),
    s = |grad rho|/(2 k_F rho), q = lap rho/(4 k_F^2 rho); per spin with k_F,s = (6 pi^2 rho_s)^(1/3);
    alpha = (tau - tau_W)/tau_unif, tau_unif = (3/10)(3 pi^2)^(2/3) rho^(5/3) d_s(zeta) (SCAN convention), per spin
    alpha_s = (tau_s - tau_W,s)/((3/10)(6 pi^2)^(2/3) rho_s^(5/3))."""
    tiny = 1e-300
    r = x.rho.sum(0)
    g = x.grad.sum(0)
    t = x.tau.sum(0)
    lap = x.lapl.sum(0)
    z = (x.rho[0] - x.rho[1]) / np.maximum(r, tiny)
    kf = (3 * np.pi ** 2 * r) ** (1 / 3)
    gn = np.sqrt(np.sum(g ** 2, axis=0))
    d = {"r_s": (3 / (4 * np.pi * np.maximum(r, tiny))) ** (1 / 3),
         "s": gn / (2 * kf * np.maximum(r, tiny)),
         "q": lap / (4 * kf ** 2 * np.maximum(r, tiny)),
         "zeta": z}
    ds = 0.5 * ((1 + z) ** (5 / 3) + (1 - z) ** (5 / 3))
    tw = gn ** 2 / (8 * np.maximum(r, tiny))
    d["alpha"] = (t - tw) / ((3 / 10) * (3 * np.pi ** 2) ** (2 / 3) * np.maximum(r, tiny) ** (5 / 3) * ds)
    for sgn, sp in ((0, "up"), (1, "dn")):
        rs_ = np.maximum(x.rho[sgn], tiny)
        kfs = (6 * np.pi ** 2 * rs_) ** (1 / 3)
        gs = np.sqrt(np.sum(x.grad[sgn] ** 2, axis=0))
        d[f"s_{sp}"] = gs / (2 * kfs * rs_)
        d[f"alpha_{sp}"] = (x.tau[sgn] - gs ** 2 / (8 * rs_)) / ((3 / 10) * (6 * np.pi ** 2) ** (2 / 3) * rs_ ** (5 / 3))
    d["w_x_up"], d["w_x_dn"], d["w_c"], d["w_c_prime"] = fm["w_x_up"], fm["w_x_dn"], fm["w_c"], fm["w_c_prime"]
    m = np.abs(x.rho[0] - x.rho[1])
    wts = {"abs_m": w * m, "rho": w * r}
    out = {"note": "SG15 valence (+ semicore) pseudo-densities", "quantiles": {k: {wn: _wq(v, wv) for wn, wv in wts.items()}
                                                                              for k, v in d.items()}}
    M = float(np.sum(w * m))
    s_edges = (0.0, 0.25, 0.5, 1.0, 2.0, np.inf)
    a_edges = (-np.inf, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, np.inf)
    out["abs_m_share_rs_x_s"] = {f"rs{'<' if lo else '>='}0.75": [
        {"s_lo": s_edges[b], "s_hi": s_edges[b + 1],
         "share": float(np.sum((w * m)[((d['r_s'] < 0.75) == lo) & (d['s'] >= s_edges[b]) & (d['s'] < s_edges[b + 1])])) / M}
        for b in range(len(s_edges) - 1)] for lo in (True, False)}
    out["abs_m_share_alpha"] = [{"a_lo": a_edges[b], "a_hi": a_edges[b + 1],
                                 "share": float(np.sum((w * m)[(d['alpha'] >= a_edges[b]) & (d['alpha'] < a_edges[b + 1])])) / M,
                                 "rho_share": float(np.sum((w * r)[(d['alpha'] >= a_edges[b]) & (d['alpha'] < a_edges[b + 1])]))
                                 / float(np.sum(w * r))} for b in range(len(a_edges) - 1)]
    return out


def pairing(fm, rho_up, rho_dn):
    """Frozen pairing test on one density (per-point arrays fm from _kernel): E_xc density with the switches as in the model,
    with the correlation switch set to the exchange switch (w_c := w_x_bar = sum_s rho_s w_x,s / rho; the payback acts on
    top as in the model: w_c' = w + (1 - w) pb), and with the exchange switches set to the correlation switch (w_x,s := w_c).
    Lambda' (e_T - e_x - e_c) is recomputed with each pair; e_T is unchanged."""
    rt = np.maximum(rho_up + rho_dn, 1e-300)
    wxu, wxd, w0, pb, lam, eT = fm["w_x_up"], fm["w_x_dn"], fm["w_c"], fm["pb"], fm["Lambda_prime"], fm["eT"]

    def exc(wu, wd, wc):
        ex = wu * fm["e_isox_up"] + (1 - wu) * fm["e_defx_up"] + wd * fm["e_isox_dn"] + (1 - wd) * fm["e_defx_dn"]
        w1 = wc + (1 - wc) * pb
        ec = w1 * fm["e_isoc"] + (1 - w1) * fm["e_uegc"]
        return ex + ec + lam * (eT - ex - ec), ex, ec

    wbar = (rho_up * wxu + rho_dn * wxd) / rt
    return {"model": exc(wxu, wxd, w0) + (w0,), "wc_eq_wx": exc(wxu, wxd, wbar) + (wbar,), "wx_eq_wc": exc(w0, w0, w0) + (w0,)}


def fsm_parts(functional_name, grid, rho, tau):
    """Per FSM point: E_x, E_c (clean totals, Ry), the pieces by end (indicative), |m|-weighted switch means, and for the full
    SCH-SI sets the frozen pairing test."""
    from ..functional import LIBXC_CODES, LibxcFunctional, get
    from ..host.evaluate import point_inputs
    x, _ = point_inputs(grid, rho, tau, True, True)
    w = grid.weight
    ry = lambda a: 2.0 * w * float(np.sum(a))          # noqa: E731
    m = np.abs(x.rho[0] - x.rho[1])
    M = float(np.sum(w * m))
    out = {"functional": functional_name, "abs_magnetization": M, "total_magnetization": float(np.sum(w * (x.rho[0] - x.rho[1])))}
    if functional_name in LIBXC_CODES:
        xc, cc = LIBXC_CODES[functional_name].split(",")
        out["E_x_ry"] = 2.0 * w * LibxcFunctional(functional_name + "-x", xc + ",").evaluate(x).e_sum
        out["E_c_ry"] = 2.0 * w * LibxcFunctional(functional_name + "-c", "," + cc).evaluate(x).e_sum
        return out
    if functional_name.endswith("#ueg-end"):
        import jax
        from xchole.core import jax_model as jm
        from xchole.sch_si import ueg_end
        full = get(functional_name.split("#")[0])
        Px = full._model.P
        px, pc = ueg_end.params_P(Px), ueg_end.params_P(jm._cparams(Px))
        exf = jax.jit(lambda r, g, l: ueg_end.ex_spin(r, g, l, px))
        ecf = jax.jit(lambda ru, rd, gu, gd, lu, ld: ueg_end.ec(ru, rd, gu, gd, lu, ld, pc))
        out["E_x_ry"] = ry(np.asarray(exf(x.rho[0], x.grad[0], x.lapl[0])) + np.asarray(exf(x.rho[1], x.grad[1], x.lapl[1])))
        out["E_c_ry"] = ry(np.asarray(ecf(x.rho[0], x.rho[1], x.grad[0], x.grad[1], x.lapl[0], x.lapl[1])))
        out["pieces_ry"] = {"X_UEG": out["E_x_ry"], "C_UEG": out["E_c_ry"]}
        return out
    f = get(functional_name)
    k = _kernel(f)
    fm = {kk: np.asarray(v) for kk, v in k(x.rho[0], x.rho[1], x.grad[0], x.grad[1], x.tau[0], x.tau[1], x.lapl[0], x.lapl[1],
                                           x.hess[0], x.hess[1]).items()}
    out["pieces_ry"] = {p: ry(fm[p]) for p in PIECES}
    out["pieces_note"] = "indicative: the switch weights move with m"
    out["E_x_ry"] = sum(out["pieces_ry"][p] for p in PIECES if p.startswith("X_"))
    out["E_c_ry"] = sum(out["pieces_ry"][p] for p in PIECES if p.startswith("C_") or p == "SI")
    out["E_xc_model_ry"] = ry(fm["model_total"])
    out["m_weighted_mean"] = {kk: float(np.sum(w * m * fm[kk])) / M for kk in ("w_x_up", "w_x_dn", "w_c", "w_c_prime", "Lambda_prime")}
    out["rho_weighted_mean"] = {kk: float(np.sum(w * (x.rho[0] + x.rho[1]) * fm[kk])) / float(np.sum(w * (x.rho[0] + x.rho[1])))
                                for kk in ("w_x_up", "w_x_dn", "w_c")}
    pr = pairing(fm, x.rho[0], x.rho[1])
    pr.pop("wc_eq_wx")                         # P1 of explore/xc_pairing (below) is the one implementation of w_c := wbar_x
    out["pairing_ry"] = {kk: {"E_xc": ry(v[0]), "E_x": ry(v[1]), "E_c_base": ry(v[2]),
                              "m_weighted_w_c": float(np.sum(w * m * v[3])) / M} for kk, v in pr.items()}
    # P1, P2 (strongxc-v3 explore/xc_pairing, reference.e_xc_points; base = this set, its payback inside w'_c) at omega 0 and 1
    from ..functional.xchole import xc_pairing_reference
    ref = xc_pairing_reference()
    args = (x.rho[0], x.rho[1], x.grad[0], x.grad[1], x.tau[0], x.tau[1], x.lapl[0], x.lapl[1], x.hess[0], x.hess[1])
    for form in ("P1", "P2"):
        for om in (0.0, 1.0):
            o = ref.e_xc_points(*args, form=form, omega=om, base=functional_name)
            key = f"{form}_omega{om:g}"
            out["pairing_ry"][key] = {"E_xc": ry(o["e_xc"]), "E_x": ry(o["e_x"]), "E_c_base": ry(o["e_c_base"]),
                                      "m_weighted_wbar_x": float(np.sum(w * m * o["wbar_x"])) / M,
                                      "m_weighted_w_c_prime": float(np.sum(w * m * o["w_c_prime"])) / M}
    out["pairing_ry"]["P1_omega0.5"] = {"E_xc": 0.5 * (out["pairing_ry"]["P1_omega0"]["E_xc"] + out["pairing_ry"]["P1_omega1"]["E_xc"]),
                                        "note": "linear in omega at a frozen density"}
    out["pairing_ry"]["P2_omega0.5"] = {"E_xc": 0.5 * (out["pairing_ry"]["P2_omega0"]["E_xc"] + out["pairing_ry"]["P2_omega1"]["E_xc"]),
                                        "note": "linear in omega at a frozen density"}
    out["pairing_omega0_minus_model_ry"] = {f: out["pairing_ry"][f"{f}_omega0"]["E_xc"] - out["E_xc_model_ry"] for f in ("P1", "P2")}
    out["pairing_note"] = ("w_x,s = the per-spin OUTER exchange switch w of xchole spin_pieces (the weight of the iso-end exchange "
                           "hole, return factor included; the inner shell weight g_s only splits the iso end into BR-He and shell); "
                           "w_c = the base outer correlation switch of ueg_terms_reduced, the payback acting on top")
    out["pairing_check_ry"] = out["pairing_ry"]["model"]["E_xc"] - out["E_xc_model_ry"]
    return out
