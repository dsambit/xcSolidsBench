"""Command line: xcsb tier {v3,v4,v5,v6} --out DIR   |   xcsb run CASE.toml --functional NAME --out DIR"""
from __future__ import annotations

import argparse
import json
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
    t = sub.add_parser("tier", help="run a node verification tier (under srun)")
    t.add_argument("tier", choices=["v3", "v4", "v5", "v6"])
    t.add_argument("--out", required=True)
    t.add_argument("--functionals", nargs="*")
    t.add_argument("--systems", nargs="*")
    r = sub.add_parser("run", help="one hooked SCF from a TOML case file")
    r.add_argument("case")
    r.add_argument("--functional", required=True)
    r.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.cmd == "tier":
        from . import tiers
        kw = {}
        if a.functionals:
            kw["names"] = tuple(a.functionals)
        if a.systems and a.tier == "v6":
            kw["systems"] = tuple(a.systems)
        res = getattr(tiers, a.tier)(a.out, **kw)
        from mpi4py import MPI
        ok = MPI.COMM_WORLD.bcast(res.get("pass"), root=0)           # the verdict is formed on the root
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(json.dumps({"tier": a.tier, "pass": ok}, default=float))
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
