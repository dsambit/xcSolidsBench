"""The external-functional adapter (xcsb.functional.external) on the laptop with a stand-in module function (the real
r2SCAN-SB lives in the pinned v4 tree; its own FD check is tests/test_external_fd.py on the node): output checks, and the
energy-vs-potential FD check through the adapter."""
import numpy as np
import pytest

from test_v2_potential_fd import fd_spin, setup
from xcsb.functional.external import ExternalFunctional


def standin(name, rho, grad, tau):
    """A smooth (rho, grad rho, tau) functional with exact derivatives: e = sum_s -c rho_s^(4/3) (1 + 0.1 x_s) + 0.05 tau_s rho_s^(1/3),
    x_s = |grad rho_s|^2 / rho_s^(8/3)."""
    g2 = np.sum(grad ** 2, axis=1)
    r = np.where(rho > 0, rho, 1.0)
    x = np.where(rho > 0, g2 / r ** (8 / 3), 0.0)
    c = 0.93
    e_s = np.where(rho > 0, -c * r ** (4 / 3) * (1 + 0.1 * x) + 0.05 * tau * r ** (1 / 3), 0.0)
    d_rho = np.where(rho > 0, -c * (4 / 3) * r ** (1 / 3) * (1 + 0.1 * x) - c * r ** (4 / 3) * 0.1 * (-8 / 3) * x / r
                     + 0.05 * tau / 3 * r ** (-2 / 3), 0.0)
    d_g2 = np.where(rho > 0, -c * r ** (4 / 3) * 0.1 / r ** (8 / 3), 0.0)
    d_grad = 2.0 * d_g2[:, None, :] * grad
    d_tau = np.where(rho > 0, 0.05 * r ** (1 / 3), 0.0)
    return {"e": e_s.sum(0), "d_rho": d_rho, "d_grad": d_grad, "d_tau": d_tau}


def _adapter(fn):
    f = ExternalFunctional.__new__(ExternalFunctional)
    f.name, f._f, f._prov = "standin", fn, {"kind": "external", "name": "standin"}
    return f


def test_adapter_fd():
    g, rho, tau = setup(polarized=True)
    fd_spin(_adapter(standin), g, rho, tau)


def test_adapter_checks_outputs():
    from xcsb.functional.contract import PointInputs
    n = 8
    x = PointInputs(rho=np.full((2, n), 0.1), grad=np.zeros((2, 3, n)), tau=np.full((2, n), 0.2), lapl=np.zeros((2, n)),
                    hess=np.zeros((2, 6, n)))
    bad = lambda *a: dict(standin(*a), d_tau=np.zeros((2, n - 1)))          # noqa: E731
    with pytest.raises(ValueError):
        _adapter(bad).evaluate(x)
    nan = lambda *a: dict(standin(*a), e=np.full(n, np.nan))                 # noqa: E731
    with pytest.raises(FloatingPointError):
        _adapter(nan).evaluate(x)
    o = _adapter(standin).evaluate(x)
    assert np.isfinite(o.e_sum) and o.d_lapl.shape == (2, n)


def test_sb_cases():
    from xcsb.studies import r2scan_sb as S
    c = S.get_case("SB-Ni")
    a = np.sqrt(2) * np.linalg.norm(c.structure.lattice[0]) / S.BOHR_PER_ANGSTROM
    assert abs(a - 3.507) < 1e-12 and c.kmesh == (8, 8, 8) and c.degauss == 0.02 and c.ecutwfc == 80.0
    f = S.get_case("SB-Fe-fsm-2.4")
    assert f.tot_magnetization == 2.4 and abs(np.linalg.norm(f.structure.lattice[0]) * 2 / np.sqrt(3) / S.BOHR_PER_ANGSTROM - 2.858) < 1e-12
    assert len(S.tasks("/r")) == 2 * 4 + 3 * 4
