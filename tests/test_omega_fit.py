"""The E1 omega fit's logic on the laptop (no SCF, no JAX): bracket from the frozen scan, omega* from three SC points with the
STOP rules, the H-atom inputs, the name parsing."""
import json
import os

import numpy as np
import pytest

from xcsb.functional.xchole import parse_e1
from xcsb.studies import fe_omega_fit as F


def _summary(mA):
    cur = {"lam1_no_gate": {"implied_moment": mA[0]}}
    for w, m in zip(F.SCAN_W, mA[1:]):
        cur["default" if w == 90.0 else f"omega{w:g}"] = {"implied_moment": m}
    return {"curves": {"A": cur, "B": cur}}


def test_bracket_and_stops():
    s = _summary([2.10, 2.30, 2.45, 2.60, 2.70, 2.74])
    assert F.bracket(s, "A", 2.381)[:2] == (10.0, 30.0)
    with pytest.raises(RuntimeError, match="below"):
        F.bracket(s, "A", 2.20)                             # between lam = 1 and omega 10
    with pytest.raises(RuntimeError, match="never reach"):
        F.bracket(s, "A", 2.0)


def _rec(run, f, m):
    from xcsb.tiers import run_dir
    d = run_dir(run, F.CASE, "hooked", f)
    os.makedirs(d, exist_ok=True)
    json.dump({"status": "ok", "magnetization": {"total": m}}, open(os.path.join(d, "record.json"), "w"))


def test_plan_and_omega_star(tmp_path, monkeypatch):
    run, tr, fr = str(tmp_path / "run"), str(tmp_path / "sb"), str(tmp_path / "frozen")
    os.makedirs(run), os.makedirs(fr)
    monkeypatch.setattr(F, "TARGET_RUN", tr)
    monkeypatch.setattr(F, "FROZEN", fr)
    _rec(tr, "PBE", 2.39)
    json.dump(_summary([2.10, 2.30, 2.45, 2.60, 2.70, 2.74]), open(os.path.join(fr, "sb_frozen_summary.json"), "w"))
    p = F.plan(run)
    assert p["target_moment"] == 2.39 and p["sets"]["SCH-SI-A:v3.1"]["omega"] == [10.0, pytest.approx(np.sqrt(10 * 30)), 30.0]
    t = F.tasks(run)
    assert len(t) == 6 and all(x["stage"] == "F" for x in t) and "--e1-diag" in F.command(run, t[0])
    for s in F.SETS:
        for w, m in zip(p["sets"][s]["omega"], (2.33, 2.40, 2.47)):
            _rec(run, F.fname(s, w), m)
    st = F.omega_star(run)
    w = st["SCH-SI-A:v3.1"]["omega_star"]
    assert p["sets"]["SCH-SI-A:v3.1"]["omega"][0] < w < p["sets"]["SCH-SI-A:v3.1"]["omega"][1]
    _rec(run, F.fname("SCH-SI-B:v3.1", 30.0), 2.30)          # non-monotone -> STOP
    with pytest.raises(RuntimeError, match="monotone"):
        F.omega_star(run)


def test_h_inputs_and_names():
    d, w = F.h_atom_inputs()
    assert abs(w @ d[0] - 1.0) < 1e-4 and np.all(d[1] == 0)
    assert parse_e1("SCH-SI-A:v3.1+E1(omega=30)") == ("SCH-SI-A:v3.1", 30.0, "half")
    assert parse_e1("SCH-SI-A:v3.1+E1kin(omega=30)") == ("SCH-SI-A:v3.1", 30.0, "kin")
    assert parse_e1(F.fname("SCH-SI-B:v3.1", 51.96152422706632)) == ("SCH-SI-B:v3.1", 51.9615, "half")
    with pytest.raises(ValueError):
        parse_e1("SCH-SI-A:v3.1+E1(omega=0)")
    with pytest.raises(KeyError):
        parse_e1("SCH-SI-A:v3.1+E1@omega=30")


def test_e1_sc_tasks():
    from xcsb.studies import fe_e1_sc as E
    n = E.names()
    assert len(n) == 2 + 2 * 2 * 2 and "SCH-SI-B:v3.1+E1kin(omega=30)" in n and "SCH-SI-A:v3.1" in n
    t = E.tasks("/r")
    assert "--e1-diag" in E.command("/r", t[2]) and "--e1-diag" not in E.command("/r", t[0])
    from xcsb.studies import r2scan_sb as S
    tt = [x for x in S.tasks("/r") if x["stage"] == "T"]
    assert len(tt) == 6 and {x["functional"] for x in tt} == {"r2SCAN-SB", "r2SCAN-SB-kin"}


def test_m_of_omega(tmp_path, monkeypatch):
    from xcsb.studies import fe_e1_sc as E
    run = str(tmp_path)
    ms = {10: 2.30, 30: 2.40, 60: 2.50, 90: 2.55, 150: 2.62, 300: 2.68}
    from xcsb.tiers import run_dir
    for s in E.SETS:
        for w, m in ms.items():
            d = run_dir(run, E.CASE, "hooked", f"{s}+E1(omega={w:g})")
            os.makedirs(d, exist_ok=True)
            json.dump({"status": "ok", "magnetization": {"total": -m, "absolute": m + 0.3}, "energies_ry": {"etot": -1.0},
                       "iterations": 9, "carrier": {"iterations": 9}, "post": {}}, open(os.path.join(d, "record.json"), "w"))
    r = E.m_of_omega(run, target=2.45)
    a = r["sets"]["SCH-SI-A:v3.1"]
    assert a["monotone"] and 30 < a["omega_at_target"] < 60 and 10 < a["omega_at_pbe"] < 30
    assert E.m_of_omega(run, target=None)["sets"]["SCH-SI-B:v3.1"]["omega_at_target"] is None
    assert E.TARGET_MOMENT == 2.22 and E.m_of_omega(run)["target"] == 2.22
