"""Command line.

    xcsb scf --case NAME --mode native|hooked --name DFT_OR_FUNCTIONAL --root DIR [--host-check XC]   (under srun)
    xcsb compare v3|v4|v6 --root DIR                                                                  (serial)
    xcsb tier v5 --out DIR                                                                            (under srun)
    xcsb run CASE.toml --functional NAME --out DIR                                                    (under srun)
    xcsb m1 tasks|ec|table --run DIR                                                                  (serial)
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _case_from_toml(path):
    import tomllib

    from .bench import structures as S
    from .host.qe import Case
    c = tomllib.load(open(path, "rb"))
    st = c.pop("structure")
    builder = getattr(S, st.pop("kind"))
    a = st.pop("a_bohr")
    structure = builder(*st.pop("species"), a, **st)
    return Case(structure, **{k: (tuple(v) if isinstance(v, list) else v) for k, v in c.items()})


def main(argv=None):
    p = argparse.ArgumentParser(prog="xcsb")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scf", help="one SCF of a named verification case (one process group per SCF)")
    s.add_argument("--case", required=True)
    s.add_argument("--mode", choices=["native", "hooked"], required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--root", required=True)
    s.add_argument("--host-check")
    s.add_argument("--payback-diag", action="store_true", help="frozen payback attribution on the converged density")
    s.add_argument("--maxiter", type=int, help="diagnostic: MAXITER for this run (recorded as a deviation)")
    s.add_argument("--mixing-beta", type=float, help="diagnostic: Broyden beta for this run (recorded as a deviation)")
    s.add_argument("--tag", default="", help="suffix of the run directory (e.g. beta0.1)")
    s.add_argument("--mag-split", action="store_true", help="frozen split of E_xc and its magnetization dependence by piece")
    s.add_argument("--fsm-parts", action="store_true", help="E_x, E_c, pieces, switch means and the frozen pairing test")
    s.add_argument("--npool", type=int, default=1, help="QE k-point pools (-nk); must divide the rank count")
    s.add_argument("--ripple", choices=["all", "high", "low"],
                   help="M1: ripple-seeded restarts after convergence, both wavenumber bands or one (studies.m1_ladder)")
    s.add_argument("--m1-diag", action="store_true", help="M1: BR-concave electron fraction of the converged density")
    s.add_argument("--e1-diag", action="store_true", help="v4: the E1 term's parts and gate diagnostics on the converged density")
    s.add_argument("--warm-from", help="production warm start (user-approved per point): a converged run directory whose QE "
                   "outdir (wavefunctions, rho, tau) is read; the point's earlier failed record is kept")
    s.add_argument("--warm-label", help="the start label in the record and the tables, e.g. 'warm start from m = 2.6'")
    s.add_argument("--warm-mode", choices=["wavefunctions", "files"],
                   help="wavefunctions: QE reads the orbitals only and rho, tau are rebuilt from them; files: also rho, tau")
    m1 = sub.add_parser("m1", help="M1 cutoff ladder: task list, E_c^PBE, tables (serial; no QE)")
    m1.add_argument("what", choices=["tasks", "ec", "table", "report-p"])
    m1.add_argument("--run", required=True)
    mg = sub.add_parser("mgo", help="MgO band gap: task list, tables, the banked reference record (serial; no QE)")
    mg.add_argument("what", choices=["tasks", "table", "bank"])
    mg.add_argument("--run", required=True)
    pv = sub.add_parser("prep-variant", help="re-solve and cache the norms of norm-dependent xchole variants (serial)")
    pv.add_argument("names", nargs="+")
    c = sub.add_parser("compare", help="tier verdicts from the SCF records")
    c.add_argument("tier", choices=["v3", "v4", "v6"])
    c.add_argument("--root", required=True)
    t = sub.add_parser("tier", help="V5 (host-step bitwise MPI check)")
    t.add_argument("tier", choices=["v5"])
    t.add_argument("--out", required=True)
    pt = sub.add_parser("payback-table", help="the Fe payback diagnostic table (xcsb.studies.fe_payback)")
    pt.add_argument("--root", required=True)
    r = sub.add_parser("run", help="one hooked SCF from a TOML case file")
    r.add_argument("case")
    r.add_argument("--functional", required=True)
    r.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.cmd == "scf":
        from mpi4py import MPI

        from . import tiers
        out = tiers.run_dir(a.root, a.case, a.mode, a.name) + (f"__{a.tag}" if a.tag else "")
        rec = tiers.scf(a.case, a.mode, a.name, out, a.host_check, a.payback_diag, a.maxiter, a.mixing_beta, a.mag_split,
                        a.fsm_parts, a.npool, a.ripple, a.m1_diag, a.e1_diag, a.warm_from, a.warm_label, a.warm_mode)
        ok = MPI.COMM_WORLD.bcast(rec.get("status") == "ok", root=0)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(json.dumps({"case": a.case, "mode": a.mode, "name": a.name, "status": rec.get("status"),
                              "error": rec.get("error")}))
        return 0 if ok else 1
    if a.cmd == "m1":
        from .studies import m1_ladder as L
        if a.what == "tasks":
            for t in L.tasks(a.run):
                print(json.dumps(dict(t, done=L.done(a.run, t), failed=L.failed(a.run, t))))
        elif a.what == "report-p":
            res = L.stage_p_report(a.run)
            json.dump(res, open(os.path.join(a.run, "stage_p_report.json"), "w"), indent=1, default=float)
            from .studies.m1_report import render_stage_p
            print(render_stage_p(res))
        elif a.what == "ec":
            print(json.dumps({"ec_pbe": L.e_c_pbe(a.run), "scan": L.scan_table(a.run)}, indent=1, default=float))
        else:
            res = L.table(a.run)
            json.dump(res, open(os.path.join(a.run, "ladder_table.json"), "w"), indent=1, default=float)
            from .studies.m1_report import render
            print(render(res))
        return 0
    if a.cmd == "mgo":
        from .studies import mgo_gap as G
        if a.what == "tasks":
            for t in G.tasks(a.run):
                print(json.dumps(dict(t, done=G.done(a.run, t), failed=G.failed(a.run, t))))
        elif a.what == "table":
            res = G.table(a.run)
            json.dump(res, open(os.path.join(a.run, "gap_table.json"), "w"), indent=1, default=float)
            print(G.render(res))
        else:
            print(json.dumps(G.bank_reference(a.run), indent=1, default=float))
        return 0
    if a.cmd == "compare":
        from . import tiers
        res = getattr(tiers, "compare_" + a.tier)(a.root)
        json.dump(res, open(os.path.join(a.root, f"{a.tier.upper()}.json"), "w"), indent=1, default=float)
        print(json.dumps({"tier": a.tier, "pass": res["pass"]}))
        return 0 if res["pass"] else 1
    if a.cmd == "prep-variant":
        from .functional.xchole import prepare_variant
        bad = 0
        for n in a.names:
            try:
                print(json.dumps({"name": n, "cache": prepare_variant(n)}), flush=True)
            except Exception as exc:                        # out-of-range values are refused by xchole: reported, not run
                bad += 1
                print(json.dumps({"name": n, "refused": f"{type(exc).__name__}: {exc}"}), flush=True)
        return 0
    if a.cmd == "payback-table":
        from .studies import fe_payback
        res = fe_payback.table(a.root)
        json.dump(res, open(os.path.join(a.root, "payback_table.json"), "w"), indent=1, default=float)
        for row in res["rows"]:
            print(json.dumps(row, default=float))
        print(json.dumps(res["nm_check"]))
        return 0
    if a.cmd == "tier":
        from mpi4py import MPI

        from . import tiers
        res = tiers.v5(a.out)
        ok = MPI.COMM_WORLD.bcast(res.get("pass"), root=0)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(json.dumps({"tier": a.tier, "pass": ok}))
        return 0 if ok else 1
    if a.cmd == "run":
        from mpi4py import MPI

        from .scf.driver import run_hooked
        rec = run_hooked(_case_from_toml(a.case), a.functional, a.out, MPI.COMM_WORLD)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(json.dumps({"status": rec["status"], "etot_ry": rec["energies_ry"]["etot"]}))
        return 0


if __name__ == "__main__":
    sys.exit(main())
