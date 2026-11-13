"""v4 Fe items on the laptop (seconds, no QE, no JAX compile of the spin-balance term): the QE density-file reader, the dense-scan
task list, and the frozen evaluation's texture, region, point discovery and curves."""
import json
import os

import numpy as np
import pytest

from xcsb.grid.spectral import SpectralGrid
from xcsb.host import qe_files
from xcsb.studies import fe_fsm_dense as D
from xcsb.studies import fe_sb_frozen as S
from xcsb.studies.fe_magnetism import case_fe80_fsm


def _write_rhog(path, mill, comps):
    def rec(b):
        n = np.int32(len(b)).tobytes()
        return n + b + n
    with open(path, "wb") as fh:
        fh.write(rec(np.array([0, len(mill), len(comps)], np.int32).tobytes()))
        fh.write(rec(np.zeros(9).tobytes()))
        fh.write(rec(np.asarray(mill, np.int32).tobytes()))
        for c in comps:
            fh.write(rec(np.asarray(c, np.complex128).tobytes()))


def test_reader_roundtrip(tmp_path):
    lat = case_fe80_fsm(1.0).structure.lattice
    g = SpectralGrid(lat, (12, 12, 12), 40.0)
    rng = np.random.default_rng(0)
    f = np.stack([1.0 + 0.1 * g.project(rng.normal(size=g.shape)) for _ in range(2)])
    ints = [np.fft.fftfreq(n, 1.0 / n).astype(int) for n in g.shape]
    sel = np.argwhere(g.mask)
    mill = np.array([[ints[a][i[a]] for a in range(3)] for i in sel])
    comps = [g.fft(f[s])[tuple(sel.T)] / g.npts for s in range(2)]
    p = str(tmp_path / "x.dat")
    _write_rhog(p, mill, comps)
    m2, c2 = qe_files.read_rhog(p)
    info = {}
    back = qe_files.to_grid(m2, c2, g.shape, info)
    assert np.allclose(back, f, atol=1e-13) and max(info["rel_imag"]) < 1e-12


def test_dense_tasks():
    t = D.tasks("/run")
    assert len(t) == 4 * 4 + 4 * 2 * 2 and len({x["id"] for x in t}) == len(t)
    w = [x for x in t if x["warm"] == 1.0 and x["functional"] == "SCH-SI-B:v3.1" and x["case"] == "Fe80-fsm-0.7"][0]
    c = D.command("/run", w)
    assert c[c.index("--warm-mode") + 1] == "wavefunctions" and c[c.index("--tag") + 1] == "warm1.0"
    assert c[c.index("--warm-from") + 1].endswith("Fe80-fsm-1.0__hooked__SCH-SI-B_v3.1")
    assert c[c.index("--warm-label") + 1] == "warm start from m = 1.0 (wavefunctions)"
    assert case_fe80_fsm(0.7).tot_magnetization == 0.7


def test_source_dir_names_match_fsm_run():
    from xcsb.tiers import run_dir
    for f in D.WARM:
        for m0 in D.WARM_FROM:
            assert D.source_dir(f, m0, "/R") == run_dir("/R", D.case(m0), "hooked", f)


def test_texture_and_region():
    lat = case_fe80_fsm(1.0).structure.lattice
    g = SpectralGrid(lat, (16, 16, 16), 60.0)
    r = g.points()
    k = g.recip[0]
    n = 0.1 + 0.05 * np.cos(np.tensordot(k, r, axes=(0, 0)))
    zeta = 0.2 * np.sin(np.tensordot(k, r, axes=(0, 0)))
    rho = np.stack([0.5 * n * (1 + zeta), 0.5 * n * (1 - zeta)])
    tau = np.ones_like(rho)
    from xcsb.host.evaluate import point_inputs
    x, _ = point_inputs(g, g.project(rho), tau, True, True)
    r3d = S._region(g)
    assert r3d.min() == 0.0 and r3d.max() < np.linalg.norm(lat[0])
    out = S.texture(g, x, np.full(g.npts, 0.3), r3d)
    assert set(["all", "3d"]) <= set(out)
    a = out["all"]
    assert abs(sum(a["zeta"]["density_weighted_share"]) - 1) < 1e-12 and a["w_c_prime"]["mean"] == pytest.approx(0.3)
    assert -1 <= a["c_theta"]["q10_q50_q90"][0] <= a["c_theta"]["q10_q50_q90"][2] <= 1
    assert a["t_zeta_kF"]["mean"] > 0 and a["t_zeta_ks"]["mean"] > 0


def test_points_and_curves(tmp_path):
    run = str(tmp_path)
    for m in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 2.7400):
        for b in ("A", "B"):
            d = os.path.join(run, f"Fe80-fsm-{m:g}__hooked__SCH-SI-{b}_v3.1")
            os.makedirs(d)
            post = {} if m == 2.74 else {"fsm_parts": {"E_x_ry": -23.0, "E_c_ry": -1.6}}
            json.dump({"status": "ok", "post": post}, open(os.path.join(d, "record.json"), "w"))
    os.makedirs(os.path.join(run, "Fe80-fsm-0.7__hooked__SCH-SI-A_v3.1__warm0.5"))
    pts = S.fsm_points(run)
    assert sorted(pts["A"]) == [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
    fake = {m: {"etot_ry": -236.0 - 0.02 * m ** 2 + 0.01 * m ** 4, "E_x_ry": -23.0 - 0.02 * m ** 2, "E_c_ry": -1.6,
                "variants": {"default": {"delta_E_ry": 0.001 * m ** 2}}} for m in sorted(pts["A"])}
    c = S.curves(fake)
    assert c["model"]["implied_moment"] == pytest.approx(1.0, abs=5e-2)   # spline on a sparse grid
    assert c["default"]["implied_moment"] < c["model"]["implied_moment"]
    assert c["model"]["curvature_meV_per_muB2"]["E_m0"] == pytest.approx(-0.04 * 13605.693122994, rel=1e-6)


def test_pin_table():
    from xcsb.functional import xchole
    assert set(xchole.XCHOLE_PINS) == {"v3.1", "v4", "v4-sb", "v4-e1", "v3.2"} and xchole.XCHOLE_PIN == xchole.XCHOLE_PINS[xchole.XCHOLE_LINE]


def test_lowm_cases():
    from xcsb.studies import fe_fsm_lowm as L
    from xcsb.tiers import get_case
    t = L.tasks("/r")
    assert len(t) == (5 + 4) * 3 and len({x["id"] for x in t}) == len(t)
    c = get_case("Fe80-fsm-0.3-s0.8")
    assert c.tot_magnetization == 0.3 and c.starting_magnetization == {"Fe": 0.8}
    assert get_case("Fe80-fsm-0.3").starting_magnetization == {"Fe": 0.5}
    assert abs(np.linalg.norm(c.structure.lattice[0]) * 2 / np.sqrt(3) / 1.8897261246257702 - 2.866) < 1e-9
