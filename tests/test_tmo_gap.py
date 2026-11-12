"""AFM-II NiO / MnO gap test on the laptop (seconds, no QE): the lattice guard, the input, the LSDA gap extraction with the band
edges, the TM site moments, the setting selection."""
import os

import numpy as np
import pytest

from xcsb.grid.spectral import SpectralGrid
from xcsb.host.qe import pw_input
from xcsb.studies import tmo_gap as T

PDIR = os.environ.get("XCSB_PSEUDO_DIR")


def test_cells():
    for mat, tm in (("NiO", "Ni"), ("MnO", "Mn")):
        st = T.structure(mat)
        L = st.lattice / T.BOHR_PER_ANGSTROM
        assert st.labels[:2] == [f"{tm}1", f"{tm}2"] and st.elements[f"{tm}2"] == tm and st.nat == 4
        lens = sorted(np.round(np.linalg.norm(L, axis=1), 2))
        assert lens == ([2.94, 2.94, 5.87] if mat == "NiO" else [3.15, 3.15, 6.19])
    assert T.kmesh("NiO", 6) == (6, 3, 6) and T.kmesh("MnO", 6) == (6, 6, 3)


@pytest.fixture
def lattice():
    return None


@pytest.mark.skipif(not PDIR or not os.path.isdir(PDIR), reason="XCSB_PSEUDO_DIR not set")
def test_inputs(lattice):
    for mat, tm, nb in (("NiO", "Ni", 24 + 8), ("MnO", "Mn", 21 + 8)):
        txt = pw_input(T.get_case(f"TMO-{mat}-100-k4"), "TPSS", PDIR, "/tmp/o", "x")
        assert "K_POINTS automatic" in txt
        for s in ("occupations = 'fixed'", "tot_magnetization = 0.000000", f"nbnd = {nb}", "diago_full_acc = .true.",
                  "starting_magnetization(1) = 0.500000", "starting_magnetization(2) = -0.500000", f"{tm}1 1.0 {tm}_ONCV_PBE-1.2.upf",
                  "nat = 4", "ntyp = 3"):
            assert s in txt, (mat, s)
        assert "smearing" not in txt


def test_gaps_lsda():
    nk = 5
    kf = np.random.default_rng(0).random((2 * nk, 3))
    et = np.zeros((32, 2 * nk))
    for i in range(32):
        et[i] = (i - 23.5) * 0.05
    et[23] = -0.1 - 0.01 * np.arange(2 * nk)
    et[24] = 0.2 + 0.01 * np.arange(2 * nk)[::-1]
    g = T.gaps({"eigenvalues_ry": et.tolist(), "kpoints_frac": kf.tolist()}, "NiO")
    assert g["min_gap_ev"] == pytest.approx((0.2 - (-0.1)) * T.RY_EV) and g["vbm_spin"] == "up" and g["cbm_spin"] == "down"


def test_site_moments(lattice):
    s = T.get_case("TMO-NiO-100-k4").structure
    g = SpectralGrid(s.lattice, (24, 48, 24), 60.0)
    rho = np.full((2,) + g.shape, 0.05)
    r = g.points()
    import itertools
    for lab, f0, sgn in zip(s.labels, s.frac, (1, -1, 0, 0)):
        if sgn == 0:
            continue
        for sh in itertools.product((-1, 0, 1), repeat=3):          # periodic images
            c = (np.asarray(f0) + np.array(sh)) @ s.lattice
            gss = 0.02 * np.exp(-np.sum((r - c[:, None, None, None]) ** 2, axis=0))
            rho[0 if sgn > 0 else 1] += gss
    sm = T.site_moments(g, rho, s)
    m = sm["moments"]
    br = sm["by_radius"]
    assert set(br) == {"1.6", "2", "2.4"} and br["1.6"]["Ni1"] < br["2"]["Ni1"] < br["2.4"]["Ni1"] and br["2"]["Ni1"] == m["Ni1"]
    assert set(m) == {"Ni1", "Ni2"} and m["Ni1"] > 0 and m["Ni2"] < 0 and abs(m["Ni1"] + m["Ni2"]) < 1e-3 * abs(m["Ni1"])


def test_grid_points():
    pts = T.grid_points()
    assert (1.0, 4) in pts and (2.0, 4) in pts and (1.0, 8) in pts and len(pts) == 5
