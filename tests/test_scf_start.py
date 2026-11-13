"""The cold starts of a hooked SCF (scf.driver.run_hooked) with a scripted stand-in for the QE session (laptop, no QE): the
order of the host calls of 'tpss-carrier' (unchanged) and of 'pbe-scaffold' (XC-free step with mixing 1, the scaffold functional
through the hooks to convergence, then the functional), and what the record keeps."""
import json
import types

import numpy as np
import pytest

from xcsb.bench.structures import rocksalt
from xcsb.host.qe import Case
from xcsb.parallel.mpi import SerialComm
from xcsb.scf import driver as D

SHAPE = (4, 4, 4)


class FakeSession:
    """Converges (conv_elec) at every third mix after a potential change; logs every call."""

    def __init__(self, case, input_dft, workdir, comm, iterative, npool=1, restart=""):
        self.case, self.input_dft, self.shape, self.grid, self.log, self.n, self.conv = case, input_dft, SHAPE, object(), LOG, 0, False
        LOG.append(("session", input_dft, npool, restart))
        s = self
        cf = types.SimpleNamespace(get_conv_elec=lambda: s.conv)
        self.driver = types.SimpleNamespace(diagonalize=lambda: LOG.append(("diagonalize",)), mix=self._mix,
                                            get_scf_error=lambda: 10.0 ** -s.n, end_scf=lambda: LOG.append(("end_scf",)),
                                            qepy_modules=types.SimpleNamespace(control_flags=cf),
                                            embed=types.SimpleNamespace(etotal=-1.0, finish=True))

    def _mix(self, mix_coef):
        self.n += 1
        self.conv = self.n % 3 == 0
        LOG.append(("mix", mix_coef))

    def spin_densities(self):
        return np.ones((2,) + SHAPE), np.ones((2,) + SHAPE)

    def set_potentials(self, e_xc, v, v_tau):
        LOG.append(("set_potentials", e_xc, float(np.abs(v).max()), float(np.abs(v_tau).max())))

    def magnetization(self):
        return {"total": 0.0, "absolute": 0.0}

    def energies(self):
        return {"etot": -1.0}

    def fermi_level(self):
        return 0.0

    def stop(self):
        LOG.append(("stop",))


LOG = []


@pytest.fixture
def fake(monkeypatch):
    LOG.clear()
    monkeypatch.setattr(D, "QESession", FakeSession)
    monkeypatch.setattr(D, "get_functional", lambda name: types.SimpleNamespace(name=name, provenance={"name": name}))

    def evaluate(f, grid, rho, tau, comm, chunk):
        LOG.append(("evaluate", f.name))
        return types.SimpleNamespace(e_xc=-0.5, v=np.full((2,) + SHAPE, -0.3), v_tau=np.full((2,) + SHAPE, 0.1),
                                     stats={"min_rho": [1.0, 1.0], "frac_points_tau_below_tauW": [0.0, 0.0]})
    monkeypatch.setattr(D, "evaluate_on_grid", evaluate)


def _case(start):
    return Case(rocksalt("Li", "F", 7.5), ecutwfc=96.0, occupations="fixed", nspin=1, nbnd=9, dual=8, start=start)


HOOK = lambda f: [("evaluate", f), ("set_potentials", -0.5, 0.3, 0.1), ("diagonalize",), ("mix", 0.3)]      # noqa: E731


def test_tpss_carrier_unchanged(fake, tmp_path):
    rec = D.run_hooked(_case("tpss-carrier"), "r2SCAN", str(tmp_path), SerialComm(), npool=4)
    assert LOG == ([("session", "TPSS", 4, "")] + [("diagonalize",), ("mix", 0.3)] * 3 + HOOK("r2SCAN") * 3
                   + [("evaluate", "r2SCAN"), ("end_scf",), ("stop",)])
    assert rec["status"] == "ok" and rec["carrier"]["kind"] == "tpss-carrier" and rec["carrier"]["iterations"] == 3
    assert rec["iterations"] == 3 and "converged TPSS" in rec["start"] and all("t_iter_s" in h for h in rec["carrier"]["history"])


@pytest.mark.parametrize("name", ["r2SCAN", "PBE", "SCH-SI-A:v3.2"])
def test_pbe_scaffold(fake, tmp_path, name):
    rec = D.run_hooked(_case("pbe-scaffold"), name, str(tmp_path), SerialComm(), npool=4)
    # QE's own XC is never applied: the first diagonalization already runs with the XC-free external potential
    boot = [("set_potentials", 0.0, 0.0, 0.0), ("diagonalize",), ("mix", 1.0)]
    assert LOG == ([("session", "TPSS", 4, "")] + boot + HOOK("PBE") * 2 + HOOK(name) * 3 + [("evaluate", name), ("end_scf",), ("stop",)])
    car = rec["carrier"]
    assert rec["status"] == "ok" and car["kind"] == "pbe-scaffold" and car["functional"] == "PBE" and car["iterations"] == 2
    assert car["bootstrap"]["mix_coef"] == 1.0 and car["mixing_beta"] == 0.3 and len(car["history"]) == 2 and rec["iterations"] == 3
    assert all(h["t_iter_s"] >= h["t_hook_s"] >= 0 for h in car["history"] + rec["history"]) and "XC-free" in rec["start"]
    assert rec["case"]["start"] == "pbe-scaffold"
    assert json.load(open(tmp_path / "record.json"))["carrier"]["bootstrap"]["potential"].startswith("XC off")


def test_pbe_scaffold_refuses_carrier_beta(fake, tmp_path):
    with pytest.raises(ValueError, match="carrier_beta"):
        D.run_hooked(_case("pbe-scaffold"), "PBE", str(tmp_path), SerialComm(), carrier_beta=0.1)
