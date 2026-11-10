"""V2a: the assembled potentials are the derivatives of the grid energy. Five-point central differences of
E_grid(rho_sigma + eps delta) and E_grid(tau_sigma + eps delta) along random sphere-limited directions, each spin channel
separately (error relative to the L1 pairing int |v delta|), and through the QE layouts of nspin = 1 and nspin = 2 (laptop for libxc and the test functional; node for
SCH-SI because of the JAX compile)."""
import numpy as np
import pytest

from xcsb.functional import get
from xcsb.host.evaluate import evaluate_on_grid
from xcsb.host.layout import grid_to_qe, qe_potentials, spin_channels
from xcsb.testing import BandLimitedDensity, cells, grid_for, sphere_limited_random, tau_above_tw

ECUT = 24.0


def setup(cell="triclinic", polarized=True, seed=5):
    g = grid_for(cells()[cell], ECUT)
    d = BandLimitedDensity(g, nterms=10, c=(0.25, 0.12) if polarized else (0.2, 0.2), amp=0.008, seed=seed)
    rho = d.rho()
    if not polarized:
        rho[1] = rho[0]
    tau = tau_above_tw(g, rho)
    return g, rho, tau


def energy(f, g, rho, tau):
    return evaluate_on_grid(f, g, rho, tau, chunk=4096).e_xc


def d5(E, base, delta, amp):
    """Five-point central difference of E(base + t amp delta) at t = 0, per unit amplitude (truncation O(amp^4))."""
    e = {k: E(base + k * amp * delta) for k in (-2, -1, 1, 2)}
    return (8.0 * (e[1] - e[-1]) - (e[2] - e[-2])) / (12.0 * amp)


def fd_spin(f, g, rho, tau, eps=(2e-3, 1e-2), tol=1e-7):
    res = evaluate_on_grid(f, g, rho, tau, chunk=4096)
    worst = 0.0
    for s in range(2):
        for which in ("rho", "tau"):
            delta = np.zeros((2,) + g.shape)
            delta[s] = sphere_limited_random(g, seed=17 + 3 * s + (which == "tau"))
            base = rho if which == "rho" else tau
            amp = eps[which == "tau"] * float(np.mean(base[s]))
            if which == "rho":
                fd = d5(lambda r: energy(f, g, r, tau), rho, delta, amp)
                an = g.integrate(res.v[s] * delta[s])
                scale = g.integrate(np.abs(res.v[s] * delta[s]))
            else:
                fd = d5(lambda t: energy(f, g, rho, t), tau, delta, amp)
                an = g.integrate(res.v_tau[s] * delta[s])
                scale = g.integrate(np.abs(res.v_tau[s] * delta[s]))
            if scale == 0.0:                                 # a tau-free functional: v_tau = 0 exactly
                assert abs(fd) < 1e-12, (f.name, which, s, fd)
                continue
            err = abs(fd - an) / scale                       # relative to the L1 pairing (cancellation-safe)
            worst = max(worst, err)
            assert err < tol, (f.name, which, s, fd, an, err)
    return worst


@pytest.mark.parametrize("name", ["model:test", "PBE", "r2SCAN"])
@pytest.mark.parametrize("cell", ["triclinic", "bcc"])
def test_potential_fd_per_spin(name, cell):
    g, rho, tau = setup(cell)
    fd_spin(get(name), g, rho, tau)


@pytest.mark.parametrize("name", ["model:test", "r2SCAN"])
def test_layout_nspin1_chain_rule(name):
    """nspin = 1: QE's extpot/extkin are the derivatives with respect to the TOTAL rho and tau (Rydberg)."""
    f = get(name)
    g, rho2, tau2 = setup(polarized=False)
    rho_t, tau_t = rho2.sum(0), tau2.sum(0)
    kin_qe = (2.0 * grid_to_qe(tau_t))[:, None]                          # QE kin_r = 2 tau_Ha
    rho_s, tau_s = spin_channels(1, grid_to_qe(rho_t)[:, None], kin_qe, g.shape)
    assert np.allclose(rho_s, rho2) and np.allclose(tau_s, tau2)
    res = evaluate_on_grid(f, g, rho_s, tau_s)
    extpot, extkin, extene = qe_potentials(1, res.e_xc, res.v, res.v_tau)
    for which, pot in (("rho", extpot[:, 0] / 2.0), ("tau", extkin[:, 0])):  # Ry -> Ha for the multiplicative part
        delta = sphere_limited_random(g, seed=3 if which == "rho" else 4)
        base = rho_t if which == "rho" else tau_t
        amp = 1e-2 * float(np.mean(base))
        def E(b):
            r, t = (b, tau_t) if which == "rho" else (rho_t, b)
            return evaluate_on_grid(f, g, np.stack([r / 2, r / 2]), np.stack([t / 2, t / 2])).e_xc
        fd = d5(E, base, delta, amp)
        an = g.weight * float(np.sum(pot * grid_to_qe(delta)))
        scale = g.weight * float(np.sum(np.abs(pot * grid_to_qe(delta))))
        assert abs(fd - an) < 1e-7 * scale, (which, fd, an)
    assert extene == 2.0 * res.e_xc


def test_layout_nspin2_roundtrip():
    g, rho, tau = setup()
    rho_qe = np.stack([grid_to_qe(rho[s]) for s in range(2)], axis=1)
    kin_qe = np.stack([2.0 * grid_to_qe(tau[s]) for s in range(2)], axis=1)
    r, t = spin_channels(2, rho_qe, kin_qe, g.shape)
    assert np.array_equal(r, rho) and np.allclose(t, tau, rtol=0, atol=1e-15)
    res = evaluate_on_grid(get("PBE"), g, r, t)
    extpot, extkin, _ = qe_potentials(2, res.e_xc, res.v, res.v_tau)
    assert np.allclose(extpot[:, 1], 2.0 * grid_to_qe(res.v[1]), rtol=0, atol=0)


@pytest.mark.node
@pytest.mark.parametrize("name", ["SCH-SI-A:v3.1", "SCH-SI-B:v3.1"])
def test_potential_fd_xchole(name):
    g, rho, tau = setup("triclinic")
    fd_spin(get(name), g, rho, tau)
