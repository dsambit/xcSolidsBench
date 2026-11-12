"""v4 Fe: does the lower FSM branch (w_x,dn ~0.30) reach m -> 0? (user go 2026-11-12). Cold-start FSM points of SCH-SI-A/B:v3.1 at
m = 0.2-0.5 (A also 0.1) with three carrier starts, starting_magnetization in STARTS (the ONLY setting that changes between
starts: the carrier TPSS start from QE's random wavefunctions with that initial moment; everything else as the FSM run: bcc Fe
a = 2.866 A (the FSM run's lattice, so the points join the existing curves), 80/320 Ry, 8^3 k, Gaussian 0.02 Ry, beta 0.3,
conv_thr 1e-8, tot_magnetization m, 8 ranks, one pool, --fsm-parts; xchole line v3.1 as the FSM run). Cases
'Fe80-fsm-<m>-s<start>'."""
from __future__ import annotations

import json
import os

SETS = ("SCH-SI-A:v3.1", "SCH-SI-B:v3.1")
M = {"SCH-SI-A:v3.1": (0.1, 0.2, 0.3, 0.4, 0.5), "SCH-SI-B:v3.1": (0.2, 0.3, 0.4, 0.5)}
STARTS = (0.2, 0.5, 0.8)


def case(m, s):
    return f"Fe80-fsm-{m:.1f}-s{s:g}"


def tasks(run):
    return [{"id": f"L__{case(m, s)}__{f}", "case": case(m, s), "functional": f, "width": 1, "stage": "L", "est_s": 700.0}
            for f in SETS for m in M[f] for s in STARTS]


def record(run, task):
    from ..tiers import run_dir
    p = os.path.join(run_dir(run, task["case"], "hooked", task["functional"]), "record.json")
    return json.load(open(p)) if os.path.exists(p) else None


def done(run, task):
    r = record(run, task)
    return r is not None and r.get("status") == "ok"


def failed(run, task):
    r = record(run, task)
    return r is not None and r.get("status") == "failed"


def command(run, task):
    return ["scf", "--case", task["case"], "--mode", "hooked", "--name", task["functional"], "--root", run, "--fsm-parts"]
