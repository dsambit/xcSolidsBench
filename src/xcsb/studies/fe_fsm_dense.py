"""v4 Fe item 1 (user 2026-11-11): dense FSM scan and hysteresis on the 0.5-1.0 mu_B step of SCH-SI-A/B:v3.1.

In A and B, E(m) drops 620-720 meV between m = 0.5 and 1.0 (PBE 165), with the shell exchange -5254 meV, the UEG-gauge exchange
+4703, BR-He -1269 and w_x,dn 0.43 -> 0.30 inside the step: possibly a change of state. Same settings as the FSM run (Fe80-fsm-<m>:
80/320 Ry, 8^3 k, Gaussian 0.02, beta 0.3, conv_thr 1e-8, tot_magnetization m; 8 ranks, one pool; --fsm-parts):

    cold      m = 0.6, 0.7, 0.8, 0.9 for SCH-SI-A, SCH-SI-B, SCH-SI-A#ueg-end, PBE
    warm      m = 0.6-0.9 for A and B from the converged m = 0.5 and m = 1.0 states of the FSM run (their WAVEFUNCTIONS only:
              warm mode 'wavefunctions', run-directory tags warm0.5, warm1.0; labels 'warm start from m = 0.5 (wavefunctions)')

Hysteresis: two branches at m when two starts both converged and differ by more than BRANCH_TOL in E_tot. Reported per point and
start: E, iterations, m-weighted w_x per spin and w_c, the piece energies (fsm_parts); no orbital projections (the host has none)."""
from __future__ import annotations

import json
import os

DENSE_M = (0.6, 0.7, 0.8, 0.9)
COLD = ("SCH-SI-A:v3.1", "SCH-SI-B:v3.1", "SCH-SI-A:v3.1#ueg-end", "PBE")
WARM = ("SCH-SI-A:v3.1", "SCH-SI-B:v3.1")
WARM_FROM = (0.5, 1.0)
BRANCH_TOL = 1e-4                 # Ry
FSM_RUN = os.environ.get("XCSB_FSM_RUN", "/pscratch/sd/d/dsambit/xc_functional_dev/runs/xcsb/2026-11-11_fe_fsm")


def case(m):
    """One decimal, as the FSM run names its points (Fe80-fsm-1.0, not -1)."""
    return f"Fe80-fsm-{m:.1f}"


def _safe(f):
    return f.replace(":", "_").replace("@", "__").replace(",", "_").replace("=", "")


def source_dir(f, m0, fsm_run=FSM_RUN):
    return os.path.join(fsm_run, f"{case(m0)}__hooked__{_safe(f)}")


def tasks(run):
    t = [{"id": f"cold__{case(m)}__{_safe(f)}", "case": case(m), "functional": f, "tag": "", "warm": None, "width": 1,
          "stage": "D", "est_s": 900.0 if f.startswith("SCH") and "#" not in f else 300.0} for m in DENSE_M for f in COLD]
    t += [{"id": f"warm{m0:.1f}__{case(m)}__{_safe(f)}", "case": case(m), "functional": f, "tag": f"warm{m0:.1f}", "warm": m0,
           "width": 1, "stage": "D", "est_s": 900.0} for m in DENSE_M for f in WARM for m0 in WARM_FROM]
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
    a = ["scf", "--case", task["case"], "--mode", "hooked", "--name", task["functional"], "--root", run, "--fsm-parts"]
    if task["warm"] is not None:
        a += ["--tag", task["tag"], "--warm-from", source_dir(task["functional"], task["warm"]),
              "--warm-label", f"warm start from m = {task['warm']:.1f} (wavefunctions)", "--warm-mode", "wavefunctions"]
    return a


def _row(r):
    if r is None:
        return {"status": "missing"}
    out = {"status": r.get("status"), "start": r.get("start") if r.get("warm_start") is None else r["warm_start"]["label"]}
    if r.get("status") != "ok":
        h = r.get("history", [])
        out.update(error=r.get("error"), dr2_last=[x.get("dr2") for x in h[-10:]])
        return out
    fp = (r.get("post") or {}).get("fsm_parts", {})
    out.update(etot_ry=r["energies_ry"]["etot"], m=r["magnetization"]["total"], abs_m=r["magnetization"]["absolute"],
               iterations=r["iterations"], E_x_ry=fp.get("E_x_ry"), E_c_ry=fp.get("E_c_ry"), pieces_ry=fp.get("pieces_ry"),
               m_weighted=fp.get("m_weighted_mean"))
    return out


def table(run, fsm_run=FSM_RUN):
    """Per functional and m (0.5 and 1.0 from the FSM run, 0.6-0.9 here): every start's row; branches where two converged starts
    differ by more than BRANCH_TOL."""
    from ..tiers import run_dir
    out = {"branch_tol_ry": BRANCH_TOL, "rows": {}, "branches": []}
    for f in COLD:
        rows = {}
        for m in (0.5,) + DENSE_M + (1.0,):
            if m in WARM_FROM:
                p = os.path.join(run_dir(fsm_run, case(m), "hooked", f), "record.json")
                rows[f"{m:.1f}"] = {"cold (FSM run)": _row(json.load(open(p)) if os.path.exists(p) else None)}
                continue
            starts = {"cold": _row(record(run, {"case": case(m), "functional": f, "tag": ""}))}
            if f in WARM:
                for m0 in WARM_FROM:
                    starts[f"warm{m0:.1f}"] = _row(record(run, {"case": case(m), "functional": f, "tag": f"warm{m0:.1f}"}))
            ok = {k: v for k, v in starts.items() if v.get("status") == "ok"}
            if len(ok) >= 2:
                es = [v["etot_ry"] for v in ok.values()]
                spread = max(es) - min(es)
                if spread > BRANCH_TOL:
                    out["branches"].append({"functional": f, "m": m, "spread_ry": spread,
                                            "etot_ry": {k: v["etot_ry"] for k, v in ok.items()}})
                starts["spread_ry"] = spread
            rows[f"{m:.1f}"] = starts
        out["rows"][f] = rows
    return out


def render(t) -> str:
    lines = [f"# Fe FSM dense scan + hysteresis (v4; branch criterion: converged starts differing by > {t['branch_tol_ry']} Ry)", ""]
    lines.append("| functional | m | start | status | E (Ry) | it | w_x up | w_x dn | w_c | E_x (Ry) | E_c (Ry) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for f, rows in t["rows"].items():
        for m, starts in rows.items():
            for k, v in starts.items():
                if k == "spread_ry":
                    continue
                mw = v.get("m_weighted") or {}
                lines.append(f"| {f} | {m} | {k} | {v['status']} | {v.get('etot_ry', '')} | {v.get('iterations', '')} | "
                             f"{mw.get('w_x_up', '')} | {mw.get('w_x_dn', '')} | {mw.get('w_c', '')} | {v.get('E_x_ry', '')} | "
                             f"{v.get('E_c_ry', '')} |")
    lines.append("\nBranches: " + (json.dumps(t["branches"]) if t["branches"] else "none found"))
    return "\n".join(lines)
