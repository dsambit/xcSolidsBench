"""Command line.

    xcsb scf --case NAME --mode native|hooked --name DFT_OR_FUNCTIONAL --root DIR [--host-check XC]   (under srun)
    xcsb compare v3|v4|v6 --root DIR                                                                  (serial)
    xcsb tier v5 --out DIR                                                                            (under srun)
    xcsb run CASE.toml --functional NAME --out DIR                                                    (under srun)
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
        rec = tiers.scf(a.case, a.mode, a.name, out, a.host_check, a.payback_diag, a.maxiter, a.mixing_beta)
        ok = MPI.COMM_WORLD.bcast(rec.get("status") == "ok", root=0)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(json.dumps({"case": a.case, "mode": a.mode, "name": a.name, "status": rec.get("status"),
                              "error": rec.get("error")}))
        return 0 if ok else 1
    if a.cmd == "compare":
        from . import tiers
        res = getattr(tiers, "compare_" + a.tier)(a.root)
        json.dump(res, open(os.path.join(a.root, f"{a.tier.upper()}.json"), "w"), indent=1, default=float)
        print(json.dumps({"tier": a.tier, "pass": res["pass"]}))
        return 0 if res["pass"] else 1
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
