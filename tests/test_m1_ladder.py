"""M1 step 1 (cutoff ladder) on the laptop, seconds, no QE: case and input generation, the ripple seeds (both channels) on a
tiny grid against explicit orbitals, the seed amplitude bookkeeping, the BR-concave fraction, E_c^PBE selection and the task
list from synthetic records, and the campaign runner with a fake step."""
import json
import os
import sys
import time

import numpy as np
import pytest

from xcsb import campaign
from xcsb.grid.spectral import SpectralGrid
from xcsb.host.qe import pw_input
from xcsb.studies import m1_ladder as L
from xcsb.tiers import run_dir

PDIR = os.environ.get("XCSB_PSEUDO_DIR")
needs_pseudos = pytest.mark.skipif(not PDIR or not os.path.isdir(PDIR), reason="XCSB_PSEUDO_DIR not set")


# ---- cases ---------------------------------------------------------------------------------------------------------------------
def test_case_names():
    assert L.parse_case("M1-Fe-FM-87.5-d8") == ("Fe", "FM", 87.5, 8, 20)
    assert L.parse_case("M1-Fe-FM-40-d4-k8") == ("Fe", "FM", 40.0, 4, 8)
    assert L.case_name("V", "NM", 0.6 * 70) == "M1-V-NM-42-d4"
    for bad in ("M1-Co-FM-80-d4", "M1-Fe-AFM-80-d4", "Fe80"):
        with pytest.raises(KeyError):
            L.parse_case(bad)
    c = L.get_case("M1-V-NM-60-d4")
    assert c.nspin == 2 and c.starting_magnetization == {"V": 0.0} and c.kmesh == (20, 20, 20) and c.degauss == 0.01
    assert L.get_case("M1-Fe-NM1-60-d4").nspin == 1
    a = np.linalg.norm(L.get_case("M1-Fe-FM-60-d4").structure.lattice[0]) * 2 / np.sqrt(3) / L.BOHR_PER_ANGSTROM
    assert abs(a - 2.858) < 1e-12                                       # static lattice (user 2026-11-11, Q10)
    assert L.A_EXP_ANGSTROM == {"Fe": 2.858, "V": 3.023}


@needs_pseudos
def test_inputs():
    for name in ("M1-Fe-FM-80-d4", "M1-Fe-NM1-80-d4", "M1-V-NM-48-d4", "M1-V-FM-240-d8"):
        txt = pw_input(L.get_case(name), "TPSS", PDIR, "/tmp/out", "x")
        assert "K_POINTS automatic\n  20 20 20 0 0 0" in txt and "degauss = 0.01000000" in txt and "smearing = 'gaussian'" in txt
        assert "conv_thr = 1.000e-08" in txt
    assert "starting_magnetization(1) = 0.000000" in pw_input(L.get_case("M1-V-NM-48-d4"), "TPSS", PDIR, "/tmp/o", "x")
    assert "nspin = 1" in pw_input(L.get_case("M1-Fe-NM1-80-d4"), "TPSS", PDIR, "/tmp/o", "x")
    assert "ecutrho = 1920.000000" in pw_input(L.get_case("M1-V-FM-240-d8"), "TPSS", PDIR, "/tmp/o", "x")


# ---- ripple seeds ----------------------------------------------------------------------------------------------------------------
def _tiny(ecutwfc=6.0, dual=4):
    s = L.get_case(f"M1-Fe-FM-{ecutwfc:g}-d{dual}").structure
    n = round(L.fft_points("Fe", ecutwfc, dual) ** (1 / 3))
    return SpectralGrid(s.lattice, (n, n, n), dual * ecutwfc), ecutwfc


def _orbitals(grid, e_orb, n_orb=3, seed=1):
    """Real band-limited orbitals (|G|^2 <= e_orb) per spin, and rho, tau, grad of each orbital (spectral, exact)."""
    rng = np.random.default_rng(seed)
    m = grid.G2 <= e_orb
    phis = []
    for s in range(2):
        ps = []
        for i in range(n_orb):
            c = (rng.normal(size=grid.shape) + 1j * rng.normal(size=grid.shape)) * m
            c[0, 0, 0] = 0.0
            f = grid.ifft_real(c)
            ps.append(0.1 * (1.0 + 0.3 * i) + 0.03 * f / f.std())
        phis.append(ps)
    return phis


def _rho_tau(grid, phis):
    rho = np.stack([sum(p ** 2 for p in ps) for ps in phis])
    tau = np.zeros_like(rho)
    for s, ps in enumerate(phis):
        for p in ps:
            gp, _, _ = grid.derivatives(p, laplacian=False, hessian=False)
            tau[s] += 0.5 * np.sum(gp ** 2, axis=0)
    return rho, tau


@pytest.mark.parametrize("band", list(L.BANDS))
def test_star_symmetric_and_bounded(band):
    grid, e = _tiny(20.0)
    miller, gv, k = L.star(grid, e, L.BANDS[band])
    assert k <= L.BANDS[band] * np.sqrt(e) + 1e-12 and len(gv) % 2 == 0
    assert {tuple(x) for x in miller} == {tuple(-x) for x in miller}
    assert np.allclose(np.linalg.norm(gv, axis=1), k)


def test_two_bands_distinct_production():
    """At a production top rung (Fe, 240 Ry) the two bands are different shells, k_low about half of k_high."""
    e = 240.0
    s = L.get_case(L.case_name("Fe", "FM", e)).structure
    n = round(L.fft_points("Fe", e, 4) ** (1 / 3))
    grid = SpectralGrid(s.lattice, (n, n, n), 4 * e)
    kh, kl = L.star(grid, e, L.BANDS["high"])[2], L.star(grid, e, L.BANDS["low"])[2]
    assert kh <= 0.8 * np.sqrt(e) and kl <= 0.4 * np.sqrt(e) and 0.4 < kl / kh < 0.55


@pytest.mark.parametrize("band", list(L.BANDS))
def test_seed_common_factor_exact(band):
    """rho', tau' of the density-level seed equal those of the explicit orbitals g phi_i; tau' - tau_W' = g^2 (tau - tau_W)."""
    s = L.get_case("M1-Fe-FM-20-d4").structure
    grid = SpectralGrid(s.lattice, (40, 40, 40), 160.0)       # orbitals |G|^2 <= 6, stars |G| <= 0.8 sqrt(20): products exact
    phis = _orbitals(grid, 6.0)
    rho, tau = _rho_tau(grid, phis)
    plan = L.SeedPlan(20.0)
    spec = {"label": f"{band}-0.05", "band": band, "eps": 0.05}
    c = plan.coefficient(grid, spec)
    _, gv, k, h, gh = plan._star(grid, band)
    r1, t1 = L.common_factor(grid, rho, tau, c, h, gh)
    g = 1.0 + c * h
    r2, t2 = _rho_tau(grid, [[g * p for p in ps] for ps in phis])
    assert np.max(np.abs(r1 - r2)) <= 1e-12 * np.max(rho) and np.max(np.abs(t1 - t2)) <= 1e-11 * np.max(tau)
    tw = lambda r: np.sum(grid.derivatives(r, laplacian=False, hessian=False)[0] ** 2, axis=1) / (8 * r)
    assert np.max(np.abs((t1 - tw(r1)) - g ** 2 * (tau - tw(rho)))) <= 1e-10 * np.max(tau)
    assert np.isclose(np.max(np.abs(g - 1)), 0.05, rtol=1e-12)              # the peak modulation eps, in both bands
    # the seeded fields: the electron count of each spin kept, inside the ecutrho sphere, positive
    rs, ts = plan.apply(grid, rho, tau, spec)
    for sp in range(2):
        assert np.isclose(grid.integrate(rs[sp]), grid.integrate(rho[sp]), rtol=1e-12)
    assert np.allclose(grid.project(rs), rs, atol=1e-14 * np.max(rho)) and rs.min() > 0


def test_amplitudes_ratio():
    s = L.get_case("M1-Fe-FM-20-d4").structure
    grid = SpectralGrid(s.lattice, (40, 40, 40), 160.0)
    rho, tau = _rho_tau(grid, _orbitals(grid, 6.0))
    plan = L.SeedPlan(20.0)
    for spec in plan.specs():
        r1, t1 = plan.apply(grid, rho, tau, spec)
        back = plan.amplitudes(grid, (rho, tau), (r1, t1), (rho, tau), spec["band"])
        kept = plan.amplitudes(grid, (rho, tau), (r1, t1), (r1, t1), spec["band"])
        for key in ("rho", "lapl_rho", "abs_grad_rho"):
            for sp in range(2):
                assert abs(back[key][sp]["ratio"]) < 1e-12 and abs(kept[key][sp]["ratio"] - 1) < 1e-12
                assert kept[key][sp]["d_seed_rms"] > 0
    d = plan.describe(grid)
    assert all(b["n_star"] > 0 for b in d["bands"].values()) and len(d["seeds"]) == len(L.EPS) * len(L.BANDS)
    assert d["bands"]["low"]["k_bohr_inv"] < d["bands"]["high"]["k_bohr_inv"]


# ---- (b) concave fraction --------------------------------------------------------------------------------------------------------
def test_concave_fraction_uniform():
    grid, _ = _tiny()
    r = 0.02
    rho = np.full((2,) + grid.shape, r)
    tau = np.full((2,) + grid.shape, 0.3 * (6 * np.pi ** 2) ** (2 / 3) * r ** (5 / 3))
    z = -0.35 * 0.3 * 6 ** (2 / 3) * np.pi ** (2 / 3)                         # -0.744 > z* = -0.861
    out = L.concave_fraction(grid, rho, tau, 0.35)
    assert out["fraction"] == 0.0 and np.isclose(out["z_min_by_spin"][0], z, rtol=1e-12)
    assert L.concave_fraction(grid, rho, tau, 0.5)["fraction"] == pytest.approx(1.0, rel=1e-12)


# ---- stage P -> E_c -> stage L, with synthetic records -----------------------------------------------------------------------------
def _fake_record(run, case, func, etot, m, ripple=None):
    d = run_dir(L.scf_root(run), case, "hooked", func)
    os.makedirs(d, exist_ok=True)
    rec = {"status": "ok", "energies_ry": {"etot": etot}, "magnetization": {"total": m, "absolute": abs(m)},
           "iterations": 10, "carrier": {"iterations": 12}, "seconds": 100.0,
           "history": [{"t_hook_s": 0.1}] * 10}
    if ripple is not None:
        rec["ripple"] = ripple
    json.dump(rec, open(os.path.join(d, "record.json"), "w"))


def test_ec_and_tasks(tmp_path):
    run = str(tmp_path)
    assert [t for t in L.tasks(run) if t["stage"] == "L"] == []
    for e in L.SCAN_ECUT:
        x = np.exp(-e / 15.0)                                             # converges: successive changes shrink
        _fake_record(run, L.case_name("Fe", "FM", e), "PBE", -236.0 - x, 2.3 + 0.5 * x)
        _fake_record(run, L.case_name("Fe", "NM1", e), "PBE", -235.97 - x, 0.0)
        _fake_record(run, L.case_name("V", "FM", e), "PBE", -130.0 - x, 0.0)
        _fake_record(run, L.case_name("V", "NM", e), "PBE", -130.0 - x, 0.0)
    ec = L.e_c_pbe(run)
    # Fe: |dm| = 0.5 |dx| = 0.0045 <= 0.005 from 60 -> 70 on (50 -> 60: 0.0087); V: every pair within (E_FM = E_NM, m = 0)
    assert ec == {"Fe": 60.0, "V": 40.0}
    ts = L.tasks(run)
    tl = [t for t in ts if t["stage"] == "L"]
    assert len(tl) == len(L.RUNGS) * 3 * 4 + len(L.RUNGS_D8) * 3 * 4 + 8
    assert len([t for t in ts if t["stage"] == "P"]) == len(L.SCAN_ECUT) * 4 + 1
    rip = [t for t in tl if t["ripple"]]
    assert len(rip) == 16 and sum(t["tag"] == "riplow" for t in rip) == 8 and all(t["case"] in ("M1-Fe-FM-180-d4", "M1-V-NM-120-d4") for t in rip)
    assert all(t["width"] == 4 for t in tl if t["rung"] == 3.0) and all(t["width"] == 1 for t in tl if t["rung"] < 3.0)
    assert os.path.exists(os.path.join(run, "EC.json")) and all(t["est_s"] > 0 for t in ts)
    assert L.command(run, rip[0])[-2:] == ["--ripple", "high"] and L.command(run, rip[1])[-4:] == [
        "--ripple", "low", "--tag", "riplow"] and "--npool" in L.command(run, rip[0])
    w4 = [t for t in ts if t["stage"] == "P" and t["tag"] == "w4"]
    assert len(w4) == 1 and w4[0]["width"] == 4 and L.command(run, w4[0])[-2:] == ["--tag", "w4"]
    # the stage-P report: measured constants from the records, the stage-L estimate at the E_c found
    from xcsb.tiers import run_dir as _rd
    d = _rd(L.scf_root(run), w4[0]["case"], "hooked", "PBE") + "__w4"
    os.makedirs(d, exist_ok=True)
    json.dump({"status": "ok", "energies_ry": {"etot": -236.0}, "magnetization": {"total": 2.3, "absolute": 2.3},
               "iterations": 10, "carrier": {"iterations": 12}, "seconds": 30.0, "history": [{"t_hook_s": 0.1}] * 10},
              open(os.path.join(d, "record.json"), "w"))
    rep = L.stage_p_report(run)
    assert rep["ec"]["ec_pbe"] == ec and "full_node_speedup" in rep["cost"]["measured"]
    assert rep["cost"]["full_node_speedup"] == pytest.approx((100 - 1.0) / (30 - 1.0))
    assert rep["stage_l_estimate"]["node_h_total"] > 0
    from xcsb.studies.m1_report import render_stage_p
    assert "Stage L estimate" in render_stage_p(rep)
    # a ripple task is done only once every seed has ended
    t = rip[0]
    plan = {"bands": {"high": {"k_bohr_inv": 9.0, "n_star": 12}}, "seeds": []}
    sd = {"seed": {"label": "curvature-0.001"}, "status": "failed", "error": "test"}
    _fake_record(run, t["case"], t["functional"], -1.0, 0.0, ripple={"plan": plan, "seeds": [sd] * 2 + [{"seed": sd["seed"]}]})
    assert not L.done(run, t)
    _fake_record(run, t["case"], t["functional"], -1.0, 0.0, ripple={"plan": plan, "seeds": [sd] * 3})
    assert L.done(run, t)
    from xcsb.studies.m1_report import render
    assert "E_c^PBE" in render(L.table(run))


def test_estimates_monotone():
    e = [L.estimate_s({"case": L.case_name("Fe", "FM", r * 80), "functional": "SCH-SI-B:v3.1", "width": 1}) for r in L.RUNGS]
    assert all(b > a for a, b in zip(e[:-1], e[1:]))


# ---- the campaign runner -----------------------------------------------------------------------------------------------------------
class FakeStudy:
    """Tasks a..e: a, b, c succeed (write a record), d exits 3 without a record, e never fits the remaining time."""

    def __init__(self):
        self.ts = [dict(id=x, width=w, est_s=s) for x, w, s in (("a", 1, 1), ("b", 4, 1), ("c", 1, 1), ("d", 1, 1), ("e", 1, 1e6))]

    def tasks(self, run):
        return self.ts

    def _p(self, run, t):
        return os.path.join(run, f"{t['id']}.done")

    def done(self, run, t):
        return os.path.exists(self._p(run, t))

    def failed(self, run, t):
        return False

    def command(self, run, t):
        return []


def test_runner(tmp_path):
    import subprocess
    run, st = str(tmp_path), FakeStudy()

    def launch(t, seconds, log):
        code = "import sys; sys.exit(3)" if t["id"] == "d" else f"open({st._p(run, t)!r}, 'w').write('ok')"
        return subprocess.Popen([sys.executable, "-c", code], stdout=open(log, "w"), stderr=subprocess.STDOUT)

    for t in st.ts:
        t["stage"] = "P"
    st.ts.append(dict(id="f", width=1, est_s=1, stage="L"))              # another stage: never started
    r = campaign.Runner(st, run, time.time() + 60, slots=4, jobid="J1", launch=launch, alive=lambda: {"J1"}, poll=0.05,
                        stages=["P"])
    os.makedirs(os.path.join(run, "claims", "c"))                       # a stale claim of a dead job
    open(os.path.join(run, "claims", "c", "job"), "w").write("J0")
    r.loop()
    assert all(st.done(run, t) for t in st.ts if t["id"] in "abc")
    assert os.path.exists(os.path.join(run, "claims", "d", "failed")) and not os.path.exists(os.path.join(run, "claims", "e"))
    assert not os.path.exists(os.path.join(run, "claims", "f")) and not st.done(run, st.ts[-1])
    ev = [json.loads(x) for x in open(os.path.join(run, "campaign_J1.jsonl"))]
    assert ev[-1]["kind"] == "exit" and ev[-1]["pending_fitting"] == 0
    assert any(e["kind"] == "stale_claim_removed" and e["task"] == "c" for e in ev)
