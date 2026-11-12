"""MgO band-gap test on the laptop (seconds, no QE): case and input, the special points on the SCF meshes, the gap
extraction, E_c^PBE and the setting selection from synthetic records."""
import itertools
import json
import os

import numpy as np
import pytest

from xcsb.host.qe import pw_input
from xcsb.studies import mgo_gap as G
from xcsb.tiers import run_dir

PDIR = os.environ.get("XCSB_PSEUDO_DIR")
needs_pseudos = pytest.mark.skipif(not PDIR or not os.path.isdir(PDIR), reason="XCSB_PSEUDO_DIR not set")


def test_case():
    c = G.get_case("MgO-90-k8")
    assert c.nspin == 1 and c.occupations == "fixed" and c.nbnd == 12 and c.diago_full_acc and c.kmesh == (8, 8, 8)
    a = np.sqrt(2) * np.linalg.norm(c.structure.lattice[0]) / G.BOHR_PER_ANGSTROM
    assert abs(a - 4.189) < 1e-12 and c.structure.elements == {"Mg": "Mg", "O": "O"}


@needs_pseudos
def test_input():
    txt = pw_input(G.get_case("MgO-90-k10"), "TPSS", PDIR, "/tmp/out", "x")
    for s in ("occupations = 'fixed'", "nbnd = 12", "diago_full_acc = .true.", "K_POINTS automatic\n  10 10 10 0 0 0",
              "nspin = 1", "ntyp = 2", "Mg_ONCV_PBE-1.2.upf", "O_ONCV_PBE-1.2.upf", "ecutrho = 360.000000"):
        assert s in txt, s
    assert "smearing" not in txt and "degauss" not in txt


def _mesh(n):
    return np.array(list(itertools.product(range(n), repeat=3)), dtype=float) / n


def _fake_eigs(lat, kf, gap_gamma, shift=0.0):
    """Bands: valence top at Gamma, conduction bottom at Gamma (parabolic in |k|), eV -> Ry."""
    b = G._recip(lat)
    kc = (kf - np.round(kf)) @ b
    k2 = np.sum(kc ** 2, axis=1)
    et = np.zeros((12, len(kf)))
    for i in range(12):
        et[i] = -10.0 + i * 0.5 - 0.2 * k2
    et[8:] = gap_gamma + shift + 0.3 * k2 + np.arange(4)[:, None] * 0.5
    et[7] = -0.2 * k2
    return et / G.RY_EV


@pytest.mark.parametrize("n", G.KMESHES)
def test_special_points_on_mesh(n):
    lat = G._lattice()
    kf = _mesh(n)
    for name, t in G.special_points(lat).items():
        assert G.equivalent(lat, kf, t).any(), name
    sp = G.special_points(lat)
    assert G.equivalent(lat, [0.5, 0.0, 0.0], sp["L"]) and not G.equivalent(lat, [0.5, 0.0, 0.0], sp["X"])
    assert G.equivalent(lat, [0.0, 0.5, 0.5], sp["X"]) and G.equivalent(lat, sp["X"], sp["X"])


def test_gaps():
    lat = G._lattice()
    kf = _mesh(6)
    r = {"eigenvalues_ry": _fake_eigs(lat, kf, 4.5).tolist(), "kpoints_frac": kf.tolist()}
    g = G.gaps(r, lat)
    assert g["gap_Gamma_ev"] == pytest.approx(4.5) and g["min_gap_ev"] == pytest.approx(4.5)
    assert g["gap_X_ev"] > g["gap_Gamma_ev"] and g["gap_L_ev"] > g["gap_Gamma_ev"]


def _rec(run, case, f, gap, etot):
    lat = G._lattice()
    ecut, k = G.parse_case(case)
    kf = _mesh(k)
    d = run_dir(G.scf_root(run), case, "hooked", f)
    os.makedirs(d, exist_ok=True)
    json.dump({"status": "ok", "energies_ry": {"etot": etot}, "iterations": 9, "carrier": {"iterations": 11}, "seconds": 50.0,
               "eigenvalues_ry": _fake_eigs(lat, kf, gap).tolist(), "kpoints_frac": kf.tolist()},
              open(os.path.join(d, "record.json"), "w"))


def test_ec_selection(tmp_path):
    run = str(tmp_path)
    assert [t for t in G.tasks(run) if t["stage"] == "G"] == []
    for e in G.SCAN_ECUT:
        x = np.exp(-e / 12.0)
        _rec(run, G.case_name(e, G.SCAN_K), "PBE", 4.45 + 2 * x, -150.0 - 3 * x)
    # E_c is the first cutoff from which every successive pair meets both criteria, and the pair before it does not
    ec = G.e_c_pbe(run)
    rows = G.scan_table(run)
    i = [r["ecutwfc"] for r in rows].index(ec)
    for a, b in zip(rows[i:-1], rows[i + 1:]):
        assert abs(b["etot_ry"] - a["etot_ry"]) * G.RY_MEV / 2 <= 1.0 and abs(b["gap_Gamma_ev"] - a["gap_Gamma_ev"]) <= 0.02
    a, b = rows[i - 1], rows[i]
    assert abs(b["etot_ry"] - a["etot_ry"]) * G.RY_MEV / 2 > 1.0 or abs(b["gap_Gamma_ev"] - a["gap_Gamma_ev"]) > 0.02
    ts = G.tasks(run)
    tg = [t for t in ts if t["stage"] == "G"]
    assert len(tg) == 3 * 3 * 4 and all(t["est_s"] > 0 for t in ts) and G.command(run, tg[0])[-2:] == ["--npool", "1"]
    # the grid: SCH-SI drifts with the cutoff by 0.03 eV per rung; k converged from 8^3
    gap0 = {"PBE": 4.45, "r2SCAN": 5.6, "SCH-SI-A:v3.1": 5.0, "SCH-SI-B:v3.1": 5.1}
    for t in tg:
        f, r, k = t["functional"], t["rung"], t["k"]
        drift = 0.03 * (r - 1) / 0.5 if f.startswith("SCH") else 0.0
        _rec(run, t["case"], f, gap0[f] - drift + (0.05 if k == 6 else 0.0), -150.0)
    tab = G.table(run)
    s = tab["selected"]
    assert s["r"] == 2.0 and not s["cutoff_converged"] and s["k"] == 8 and s["k_converged"]
    assert s["per_functional"]["r2SCAN"]["minus_pbe_gap_Gamma_ev"] == pytest.approx(5.6 - 4.45)
    assert "Selected" in G.render(tab)
    bank = G.bank_reference(run)
    assert set(bank["results"]) == {"PBE", "r2SCAN"} and bank["protocol"]["kmesh"] == 8
