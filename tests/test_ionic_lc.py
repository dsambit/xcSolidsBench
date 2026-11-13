"""Ionic-solid lattice constants on the laptop (seconds, no QE): protocol (cutoffs, k meshes, bands), the task list, the
scan-extension logic on synthetic E(V), the table and the banked record from synthetic records."""
import json
import os

import numpy as np
import pytest

from xcsb.bench.eos import RY_BOHR3_TO_GPA, EOSError, bm3_fit
from xcsb.host.qe import pw_input
from xcsb.studies import ionic_lc as L
from xcsb.tiers import get_case, run_dir

PDIR = os.environ.get("XCSB_PSEUDO_DIR")
needs_pseudos = pytest.mark.skipif(not PDIR or not os.path.isdir(PDIR), reason="XCSB_PSEUDO_DIR not set")


def bm3(v, v0, b0_gpa=60.0, b0p=4.5, e0=-100.0):
    eta = (v0 / v) ** (2.0 / 3.0)
    return e0 + 9.0 * v0 * (b0_gpa / RY_BOHR3_TO_GPA) / 16.0 * ((eta - 1) ** 3 * b0p + (eta - 1) ** 2 * (6 - 4 * eta))


def energy(solid, step, ratio, **kw):
    """Exact BM3 energy of the scan point for a minimum at a0 = ratio a_ref."""
    return bm3(L.volume(solid, step), L.volume(solid, 0) * ratio ** 3, **kw)


def test_protocol():
    p = L.protocol()
    for s, v in p.items():
        print(f"{s}: a_ref {v['a_ref_angstrom']} A, ecutwfc {v['ecutwfc_ry']:g} Ry, ecutrho {v['ecutrho_ry']:g} Ry, k mesh "
              f"{v['kmesh']} Gamma-centred ({v['nk_irreducible']} irreducible), n_occ {v['n_occ']}, nbnd {v['nbnd']}")
    assert {s: v["ecutwfc_ry"] for s, v in p.items()} == {"LiF": 96, "LiCl": 82, "NaF": 96, "NaCl": 96, "MgO": 96}
    assert all(v["ecutrho_ry"] == 8 * v["ecutwfc_ry"] for v in p.values())
    assert {s: v["n_occ"] for s, v in p.items()} == {"LiF": 5, "LiCl": 5, "NaF": 8, "NaCl": 8, "MgO": 8}
    assert all(v["nbnd"] == v["n_occ"] + 4 for v in p.values())
    assert {s: v["a_ref_angstrom"] for s, v in p.items()} == {"LiF": 3.972, "LiCl": 5.070, "NaF": 4.582, "NaCl": 5.569, "MgO": 4.189}
    # fcc primitive cell: |b_i| = 2 pi sqrt(3) / a, the same for the three vectors
    for s, v in p.items():
        n = int(np.ceil(2 * np.pi * np.sqrt(3) / v["a_ref_angstrom"] / 0.1))
        assert v["kmesh"] == [n, n, n] and v["kshift"] == [0, 0, 0]
    assert {s: v["kmesh"][0] for s, v in p.items()} == {"LiF": 28, "LiCl": 22, "NaF": 24, "NaCl": 20, "MgO": 26}
    assert [L.nk_irreducible(n) for n in (6, 8, 10)] == [16, 29, 47]          # the known fcc counts (mgo_gap.NK_IBZ)


def test_cases():
    c = get_case("LC-NaCl-m3")
    assert c.nspin == 1 and c.occupations == "fixed" and c.nbnd == 12 and c.dual == 8 and c.kmesh == (20, 20, 20)
    assert c.kshift == (0, 0, 0) and c.ecutwfc == 96.0 and c.conv_thr == 1e-8 and c.mixing_beta == 0.3 and not c.diago_full_acc
    a = np.sqrt(2) * np.linalg.norm(c.structure.lattice[0]) / L.BOHR_PER_ANGSTROM
    assert a == pytest.approx(5.569 * 0.97, abs=1e-12) and c.structure.elements == {"Na": "Na", "Cl": "Cl"}
    assert c.as_dict()["ecutrho"] == 768.0
    # the k mesh and the cutoff come from the reference cell and do not move along the scan
    assert all(get_case(L.case_name("LiF", n)).kmesh == (28, 28, 28) for n in range(-8, 9))
    assert L.parse_case("LC-LiF-p0") == ("LiF", 0) and L.parse_case("LC-MgO-m8") == ("MgO", -8)
    for bad in ("LC-LiF-p9", "LC-KCl-p0", "LC-LiF-m0", "LC-LiF-3", "MgO-80-k8"):
        with pytest.raises(KeyError):
            L.parse_case(bad)
    p0 = get_case(L.PHASE0)
    assert p0.ecutwfc == 40.0 and p0.kmesh == (6, 6, 6) and p0.dual == 8 and p0.nbnd == 12


@needs_pseudos
def test_input():
    txt = pw_input(get_case("LC-LiCl-p2"), "TPSS", PDIR, "/tmp/out", "x")
    for s in ("occupations = 'fixed'", "nbnd = 9", "K_POINTS automatic\n  22 22 22 0 0 0", "nspin = 1", "ecutwfc = 82.000000",
              "ecutrho = 656.000000", "Li_ONCV_PBE-1.2.upf", "Cl_ONCV_PBE-1.2.upf", "mixing_beta = 0.300000"):
        assert s in txt, s


def test_tasks(tmp_path):
    ts = L.tasks(str(tmp_path))
    assert len(ts) == 5 * 4 * 7 and len({t["id"] for t in ts}) == len(ts)
    assert [t["stage"] for t in ts] == ["R"] * 70 + ["M"] * 70
    assert {t["functional"] for t in ts if t["stage"] == "R"} == set(L.REFERENCE) == {"PBE", "r2SCAN"}
    assert {t["functional"] for t in ts if t["stage"] == "M"} == set(L.MODELS) == {"SCH-SI-A:v3.2", "SCH-SI-B:v3.2"}
    assert [t["step"] for t in ts[:7]] == [-3, -2, -1, 0, 1, 2, 3] and ts[0]["case"] == "LC-LiF-m3" and ts[0]["functional"] == "PBE"
    assert all(t["est_s"] > 0 and t["width"] == 1 for t in ts)
    assert L.command(str(tmp_path), ts[0]) == ["scf", "--case", "LC-LiF-m3", "--mode", "hooked", "--name", "PBE", "--root",
                                               os.path.join(str(tmp_path), "scf"), "--npool", "8"]
    hours = {st: sum(t["est_s"] for t in ts if t["stage"] == st) / 3600 / 4 for st in ("R", "M")}
    print("estimated node-hours (uncalibrated):", {k: round(v, 1) for k, v in hours.items()})


def test_model_sets_are_the_v32_line():
    from xcsb.functional import XCHOLE_NAMES, xchole
    assert set(L.MODELS) == set(xchole.RECORDS_BY_VERSION["v3.2"]) and set(L.MODELS) <= set(XCHOLE_NAMES)
    assert xchole.XCHOLE_PINS[L.XCHOLE_LINE] == "019c8acd65a7ee2d71f59d750338415e73695ec1"
    assert xchole.LINE_VERSION[L.XCHOLE_LINE] == "v3.2" and set(xchole.LINE_VERSION) == set(xchole.XCHOLE_PINS)


def test_bm3_guard_parameter():
    V = np.linspace(100.0, 130.0, 9)
    E = bm3(V, 101.5)
    with pytest.raises(EOSError, match="inner two thirds"):
        bm3_fit(V, E)                                               # the default guard is unchanged
    fit = bm3_fit(V, E, guard=False)
    assert fit["V0_bohr3"] == pytest.approx(101.5, abs=1e-6) and not fit["v0_inner_two_thirds"] and fit["rms_ok"]
    assert bm3_fit(V, bm3(V, 115.0))["v0_inner_two_thirds"]


def _extend_until_done(solid, ratio):
    steps, n_rounds = list(L.BASE_STEPS), 0
    while True:
        d = L.extension(solid, steps, [energy(solid, n, ratio) for n in steps])
        if d["action"] != "extend":
            return steps, d, n_rounds
        assert d["add"] and not set(d["add"]) & set(steps) and all(abs(n) <= L.MAX_STEP for n in d["add"])
        steps, n_rounds = sorted(steps + d["add"]), n_rounds + 1
        assert steps == list(range(steps[0], steps[-1] + 1))


@pytest.mark.parametrize("solid", L.SOLIDS)
def test_extension(solid):
    # minimum well inside: no extension
    assert _extend_until_done(solid, 1.005) == (list(L.BASE_STEPS), {"action": "done", "V0_bohr3": pytest.approx(L.volume(solid, 0) * 1.005 ** 3)}, 0)
    # over-expanded (minimum at +2.8 %: inside the scan, outside its inner two thirds): points on the large-volume side only
    steps, d, n = _extend_until_done(solid, 1.028)
    lo, hi = L._inner([L.volume(solid, m) for m in steps])
    assert d["action"] == "done" and n == 1 and steps[0] == -3 and 3 < steps[-1] <= 8 and lo <= d["V0_bohr3"] <= hi
    # the smallest extension that puts V0 inside the inner two thirds (-3 ... +4: inner upper bound at a = 1.0290 a_ref)
    assert steps[-1] == 4 and d["V0_bohr3"] > L._inner([L.volume(solid, m) for m in steps[:-1]])[1]
    # minimum beyond the base scan (+4.5 %), still bracketed within +8 %
    steps, d, n = _extend_until_done(solid, 1.045)
    assert d["action"] == "done" and steps[0] == -3 and steps[-1] == 6       # -3 ... +6: inner upper bound at 1.0460 a_ref
    assert d["V0_bohr3"] > L._inner([L.volume(solid, m) for m in steps[:-1]])[1]
    # compressed side
    steps, d, n = _extend_until_done(solid, 0.965)
    assert d["action"] == "done" and steps[-1] == 3 and steps[0] == -5    # -5 ... +3: inner lower bound at 0.9643 a_ref
    # not bracketed within +-8 %: stop (no value), never a relaxed guard
    steps, d, n = _extend_until_done(solid, 1.075)
    assert d["action"] == "stop" and steps == list(range(-3, 9)) and "inner two thirds" in d["reason"]
    # no BM3 minimum near the scan (minimum far outside): one point at a time on the side of the lowest energy
    d = L.extension(solid, list(L.BASE_STEPS), [energy(solid, m, 1.6) for m in L.BASE_STEPS])
    assert d == {"action": "extend", "add": [4], "side": 1, "V0_bohr3": None}
    with pytest.raises(ValueError):
        L.extension(solid, [-3, -2, 0, 1, 2, 3], [0.0] * 6)


def _rec(run, solid, step, f, etot=None, status="ok", seconds=100.0):
    d = run_dir(L.scf_root(run), L.case_name(solid, step), "hooked", f)
    os.makedirs(d, exist_ok=True)
    r = {"status": status, "error": None if status == "ok" else "SCFError: synthetic"}
    if status == "ok":
        r.update(energies_ry={"etot": etot}, iterations=9, carrier={"iterations": 11}, seconds=seconds,
                 history=[{"t_hook_s": 1.0}] * 9)
    json.dump(r, open(os.path.join(d, "record.json"), "w"))


RATIO = {"PBE": {"LiF": 1.0227, "LiCl": 1.0161, "NaF": 1.0264, "NaCl": 1.0233, "MgO": 1.0159},     # ~ the collaborator's PBE
         "r2SCAN": {"LiF": 1.0054, "LiCl": 1.0100, "NaF": 1.0075, "NaCl": 1.0114, "MgO": 1.0064},
         "SCH-SI-A:v3.2": {"LiF": 1.01, "LiCl": 1.045, "NaF": 1.02, "NaCl": 1.03, "MgO": 0.99},
         "SCH-SI-B:v3.2": {"LiF": 1.0, "LiCl": 1.09, "NaF": 1.0, "NaCl": 1.0, "MgO": 1.0}}


def _fill(run, pairs):
    """Write the records every listed scan asks for until no task of these pairs is pending."""
    while True:
        todo = [t for t in L.tasks(run) if (t["solid"], t["functional"]) in pairs and not L.done(run, t)]
        if not todo:
            return
        for t in todo:
            _rec(run, t["solid"], t["step"], t["functional"], energy(t["solid"], t["step"], RATIO[t["functional"]][t["solid"]]))


def test_scan_table_bank(tmp_path):
    run = str(tmp_path)
    assert L.scan(run, "LiF", "PBE") == {"steps": list(L.BASE_STEPS), "status": "running", "rounds": []}
    # LiCl with set A: minimum at +4.5 % -> extension tasks (stage MX) of that pair only, after its base scan is complete
    for n in L.BASE_STEPS[:-1]:
        _rec(run, "LiCl", n, "SCH-SI-A:v3.2", energy("LiCl", n, 1.045))
    assert not [t for t in L.tasks(run) if t["stage"].endswith("X")]
    _rec(run, "LiCl", 3, "SCH-SI-A:v3.2", energy("LiCl", 3, 1.045))
    tx = [t for t in L.tasks(run) if t["stage"].endswith("X")]
    assert tx and {(t["stage"], t["solid"], t["functional"]) for t in tx} == {("MX", "LiCl", "SCH-SI-A:v3.2")}
    assert all(t["step"] > 3 and t["id"].startswith("MX__LC-LiCl-p") for t in tx)
    assert len(L.tasks(run)) == 140 + len(tx) and [t["stage"] for t in L.tasks(run)][-len(tx):] == ["MX"] * len(tx)
    with pytest.raises(RuntimeError, match="no final fit"):
        L.bank_reference(run)
    pairs = {(s, f) for s in L.SOLIDS for f in L.FUNCTIONALS} - {("NaF", "SCH-SI-B:v3.2")}
    _fill(run, pairs)
    _rec(run, "NaF", 0, "SCH-SI-B:v3.2", status="failed")
    t = L.table(run)
    rows = {(r["solid"], r["functional"]): r for r in t["rows"]}
    assert len(rows) == 20
    for (s, f), r in rows.items():
        if (s, f) == ("NaF", "SCH-SI-B:v3.2"):
            assert r["status"] == "failed" and r["failed_steps"] == [0] and r["fit"] is None and r["fit_pm3"] is None
            continue
        if (s, f) == ("LiCl", "SCH-SI-B:v3.2"):                      # minimum at +9 %: not bracketed within +-8 %
            assert r["status"] == "stop" and r["fit"] is None and "inner two thirds" in r["fit_error"] and r["steps"][-1] == 8
            assert r["fit_pm3"] is None or not r["fit_pm3"]["v0_inner_two_thirds"]
            continue
        a0 = RATIO[f][s] * L.A_REF[s]
        assert r["status"] == "done" and r["fit"]["a0_angstrom"] == pytest.approx(a0, abs=1e-6)
        assert r["fit"]["B0_GPa"] == pytest.approx(60.0, rel=1e-5) and r["fit"]["B0_prime"] == pytest.approx(4.5, rel=1e-5)
        assert r["fit"]["a0_minus_ref_angstrom"] == pytest.approx(a0 - L.A_REF[s], abs=1e-6)
        assert r["fit"]["n_points"] == len(r["steps"]) and r["extended"] == (len(r["steps"]) > 7)
        assert r["fit_pm3"]["a0_angstrom"] == pytest.approx(a0, abs=1e-6) and r["fit_pm3"]["n_points"] == 7
        assert r["fit_pm3"]["v0_inner_two_thirds"] == (not r["extended"])
    r = rows[("LiCl", "SCH-SI-A:v3.2")]
    assert r["extended"] and r["steps"] == list(range(-3, 7)) and r["collaborator_a0_angstrom"] is None
    assert rows[("NaF", "PBE")]["extended"] and rows[("NaCl", "SCH-SI-A:v3.2")]["extended"]
    r = rows[("MgO", "r2SCAN")]
    assert r["collaborator_a0_angstrom"] == 4.2159 and r["collaborator_B0_GPa"] == 165.72
    assert r["fit_pm3"]["a0_minus_collaborator_angstrom"] == pytest.approx(1.0064 * 4.189 - 4.2159, abs=1e-6)
    assert rows[("MgO", "PBE")]["collaborator_B0_GPa"] is None and rows[("LiF", "PBE")]["collaborator_a0_angstrom"] == 4.0622
    # the synthetic PBE / r2SCAN minima sit at the collaborator's values to 1e-3 A
    assert all(abs(rows[(s, f)]["fit_pm3"]["a0_minus_collaborator_angstrom"]) < 1e-3 for s in L.SOLIDS for f in L.REFERENCE)
    sm = t["summary"]
    err = np.array([(RATIO["PBE"][s] - 1) * L.A_REF[s] for s in L.SOLIDS])
    assert sm["PBE"]["fit"]["complete"] and sm["PBE"]["fit"]["ME_angstrom"] == pytest.approx(err.mean(), abs=1e-6)
    assert sm["PBE"]["fit"]["MAE_angstrom"] == pytest.approx(np.abs(err).mean(), abs=1e-6)
    assert sm["SCH-SI-A:v3.2"]["fit"]["n"] == 5 and sm["SCH-SI-A:v3.2"]["fit"]["ME_angstrom"] != sm["SCH-SI-A:v3.2"]["fit"]["MAE_angstrom"]
    assert sm["SCH-SI-B:v3.2"]["fit"] == {**sm["SCH-SI-B:v3.2"]["fit"], "n": 3, "complete": False}
    txt = L.render(t)
    print(txt)
    for s in ("-3 ... +6 (extended)", "| PBE | LiF | 3.972 | 4.0622 | +0.0902 |", "(incomplete)", "failed steps [0]", "28x28x28",
              "collaborator run 2026-11-05", "ZPAE-corrected"):
        assert s in txt, s
    bank = L.bank_reference(run)
    assert set(bank["results"]) == {"PBE", "r2SCAN"} and set(bank["results"]["PBE"]) == set(L.SOLIDS)
    assert bank["results"]["r2SCAN"]["LiF"]["fit"]["a0_angstrom"] == pytest.approx(1.0054 * 3.972, abs=1e-6)
    assert bank["protocol"]["per_solid"]["LiCl"]["kmesh"] == [22, 22, 22] and bank["results"]["PBE"]["NaF"]["extended"]
    json.dumps(t, default=float), json.dumps(bank, default=float)
    # scheduling: the estimates follow the measured times once enough points are done (100 s each here)
    ts = L.tasks(run)
    k = L.cost_scale(run, ts)
    assert 0 < k < 1 and ts[0]["est_s"] == pytest.approx(k * L.estimate_s(ts[0])) and t["timing"]["cost_scale"] == pytest.approx(k)
