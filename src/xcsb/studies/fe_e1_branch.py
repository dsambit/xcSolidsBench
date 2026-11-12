"""SCH-SI-B:v3.1 + E1 (half) on free bcc Fe: SCF branches between omega 90 and 150 (user go 2026-11-12). The omega scan gave B
2.156 / 2.160 / 2.127 / 2.093 mu_B at omega 10 / 30 / 60 / 90 and 2.450 / 2.678 at 150 / 300: a jump. SB-Fe settings (a 2.858 A,
80/320 Ry, 8^3 k, Gaussian 0.02, beta 0.3), xchole line v4-e1, --fsm-parts and --e1-diag on every run.

  free   omega in FREE_W, each from a cold carrier start, a warm start from the converged omega = 90 state and one from the
         omega = 150 state (their WAVEFUNCTIONS, mode 'wavefunctions', the source's band count; tags w90, w150)
  fsm    omega = FSM_W at fixed m in FSM_M (case SB-Fe-fsm-<m>): cold, then warm from the two free branch states (w90, w150)
Branches: converged runs at the same omega (resp. m) differing by more than BRANCH_TOL in E. Analysis: the ground state (lowest
E) per omega with its moment, the omega of its 2.22 crossing or of the branch switch; E(m) at FSM_W with its minima."""
from __future__ import annotations

import json
import math
import os

SET = "SCH-SI-B:v3.1"
FREE_W = (100.0, 110.0, 120.0, 130.0)
FSM_W = 110.0
FSM_M = (1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6)
SRC_W = (90.0, 150.0)
BRANCH_TOL = 1e-4
TARGET = 2.22
E1SC_RUN = os.environ.get("XCSB_E1SC_RUN", "/pscratch/sd/d/dsambit/xc_functional_dev/runs/xcsb/v4_2026-11-12_fe_e1_sc")


def fname(w):
    return f"{SET}+E1(omega={w:g})"


def source_dir(w):
    from ..tiers import run_dir
    return run_dir(E1SC_RUN, "SB-Fe", "hooked", fname(w))


def _t(case, w, start, est):
    tag = "" if start is None else f"w{start:g}"
    return {"id": f"{case}__{fname(w)}" + (f"__{tag}" if tag else ""), "case": case, "functional": fname(w), "tag": tag,
            "warm": start, "width": 1, "stage": "B", "est_s": est}


def tasks(run):
    t = [_t("SB-Fe", w, s, 600.0 if s is None else 400.0) for w in FREE_W for s in (None,) + SRC_W]
    t += [_t(f"SB-Fe-fsm-{m:.1f}", FSM_W, None, 600.0) for m in FSM_M]
    t += [_t(f"SB-Fe-fsm-{m:.1f}", FSM_W, s, 400.0) for s in SRC_W for m in FSM_M]
    return t


def record(run, task):
    from ..tiers import run_dir
    p = os.path.join(run_dir(run, task["case"], "hooked", task["functional"]) + (f"__{task['tag']}" if task["tag"] else ""),
                     "record.json")
    return json.load(open(p)) if os.path.exists(p) else None


def done(run, task):
    r = record(run, task)
    return r is not None and r.get("status") == "ok"


def failed(run, task):
    r = record(run, task)
    return r is not None and r.get("status") == "failed"


def command(run, task):
    a = ["scf", "--case", task["case"], "--mode", "hooked", "--name", task["functional"], "--root", run, "--fsm-parts", "--e1-diag"]
    if task["warm"] is not None:
        a += ["--tag", task["tag"], "--warm-from", source_dir(task["warm"]),
              "--warm-label", f"warm start from the omega = {task['warm']:g} state (wavefunctions)", "--warm-mode", "wavefunctions"]
    return a


def _row(r):
    if r is None:
        return {"status": "missing"}
    if r.get("status") != "ok":
        return {"status": r.get("status"), "error": r.get("error")}
    post = r.get("post") or {}
    mw = (post.get("fsm_parts") or {}).get("m_weighted_mean") or {}
    e1 = post.get("e1") or {}
    return {"status": "ok", "m": abs(r["magnetization"]["total"]), "abs_m": r["magnetization"]["absolute"],
            "etot_ry": r["energies_ry"]["etot"], "iterations": r["iterations"], "w_x_up": mw.get("w_x_up"),
            "w_x_dn": mw.get("w_x_dn"), "w_c": mw.get("w_c"),
            "delta_E_meV": None if "delta_E_ry" not in e1 else e1["delta_E_ry"] * 13605.693122994,
            "retained_3d": (e1.get("gate_3d") or {}).get("retained")}


def analysis(run, target=TARGET):
    from ..tiers import run_dir
    out = {"target": target, "branch_tol_ry": BRANCH_TOL, "free": {}, "fsm": {}}
    # free: e1sc points (cold) + this run's starts
    ws = sorted(set((10.0, 30.0, 60.0, 90.0, 150.0, 300.0) + FREE_W))
    ground = []
    for w in ws:
        starts = {}
        p = os.path.join(run_dir(E1SC_RUN, "SB-Fe", "hooked", fname(w)), "record.json")
        if os.path.exists(p):
            starts["cold (e1sc)"] = _row(json.load(open(p)))
        if w in FREE_W:
            for s in (None,) + SRC_W:
                starts["cold" if s is None else f"warm{s:g}"] = _row(record(run, _t("SB-Fe", w, s, 0)))
        ok = {k: v for k, v in starts.items() if v.get("status") == "ok"}
        row = {"starts": starts}
        if ok:
            k0 = min(ok, key=lambda k: ok[k]["etot_ry"])
            es = [v["etot_ry"] for v in ok.values()]
            row.update(ground=k0, ground_m=ok[k0]["m"], spread_ry=max(es) - min(es),
                       two_branches=(max(es) - min(es)) > BRANCH_TOL)
            ground.append((w, ok[k0]["m"], ok[k0]["etot_ry"]))
        out["free"][f"{w:g}"] = row
    out["ground_m_of_omega"] = ground
    cross = None
    for (w0, m0, _), (w1, m1, _) in zip(ground[:-1], ground[1:]):
        if (m0 - target) * (m1 - target) <= 0 and m0 != m1:
            if abs(m1 - m0) > 0.15:                  # a jump: the crossing lies at the branch switch inside (w0, w1)
                cross = {"kind": "branch switch", "between_omega": [w0, w1], "m": [m0, m1]}
            else:
                x = (target - m0) / (m1 - m0)
                cross = {"kind": "interpolated", "omega": math.exp(math.log(w0) + x * (math.log(w1) - math.log(w0)))}
            break
    out["target_crossing"] = cross
    # fsm at FSM_W
    em = []
    for m in FSM_M:
        starts = {("cold" if s is None else f"warm{s:g}"): _row(record(run, _t(f"SB-Fe-fsm-{m:.1f}", FSM_W, s, 0)))
                  for s in (None,) + SRC_W}
        ok = {k: v for k, v in starts.items() if v.get("status") == "ok"}
        row = {"starts": starts}
        if ok:
            k0 = min(ok, key=lambda k: ok[k]["etot_ry"])
            row.update(lowest=k0, E_ry=ok[k0]["etot_ry"], spread_ry=max(v["etot_ry"] for v in ok.values()) - ok[k0]["etot_ry"])
            em.append((m, ok[k0]["etot_ry"]))
        out["fsm"][f"{m:.1f}"] = row
    out["E_of_m_lowest"] = em
    out["local_minima_m"] = [em[i][0] for i in range(1, len(em) - 1) if em[i][1] < em[i - 1][1] and em[i][1] < em[i + 1][1]]
    return out
