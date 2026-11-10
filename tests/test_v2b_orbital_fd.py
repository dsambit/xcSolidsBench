"""V2b: orbital-level check of the GKS operator (laptop for the test functional and r2SCAN; node for SCH-SI).

A plane-wave toy: Bloch orbitals psi = e^{ik.r} u with u inside the wavefunction sphere |k+G|^2 <= ecutwfc = ecutrho/4,
several k-points, two spin channels with different occupations; rho_sigma = rho_c + sum_k w_k sum_i |psi|^2 and
tau_sigma = tau_c + (1/2) sum w_k |grad psi|^2 (constant backgrounds keep rho, tau - tau_W away from zero). Along an
occupied-virtual rotation psi_i -> cos(t) psi_i + sin(t) phi_a within one spin channel and one k-point,

    dE_xc/dt = 2 w_k Re <phi_a | H_xc,sigma | psi_i>,   H_xc,sigma psi = v_sigma psi - (1/2) div(v_tau,sigma grad psi)   (Ha),

with v, v_tau from the host (xcsb.host.evaluate). Checked against five-point differences of E_xc(t). This validates the
factor 1/2 in tau and the form of the tau operator, per spin."""
import numpy as np
import pytest

from xcsb.functional import get
from xcsb.host.evaluate import evaluate_on_grid
from xcsb.testing import cells, grid_for

ECUTWFC = 6.0


class Toy:
    def __init__(self, seed=0, nocc=(2, 1), nvirt=2):
        self.g = grid_for(cells()["triclinic"], 4 * ECUTWFC)
        g = self.g
        rng = np.random.default_rng(seed)
        self.kpts = [np.zeros(3), 0.5 * g.recip[0] + 0.25 * g.recip[2]]
        self.wk = [0.5, 0.5]
        n = g.shape
        nyq = np.ones(n, dtype=bool)
        for ax, m in enumerate(n):
            if m % 2 == 0:
                idx = [slice(None)] * 3
                idx[ax] = m // 2
                nyq[tuple(idx)] = False
        self.full = nyq                                    # full-grid spectral derivative (Nyquist zeroed)
        self.orb = {}                                      # (spin, ik) -> (occupied u, virtual u)
        for s in range(2):
            for ik, k in enumerate(self.kpts):
                kg = g.G + k[:, None, None, None]
                sph = np.sum(kg ** 2, axis=0) <= ECUTWFC
                m = nocc[s] + nvirt
                C = (rng.standard_normal((m,) + n) + 1j * rng.standard_normal((m,) + n)) * sph
                u = np.fft.ifftn(C, axes=(-3, -2, -1))
                Q, _ = np.linalg.qr(u.reshape(m, -1).T)        # orthonormal in the grid inner product (up to w)
                u = (Q.T.reshape((m,) + n)) / np.sqrt(g.weight)
                self.orb[(s, ik)] = (u[: nocc[s]], u[nocc[s]:])
        self.rho_c, self.tau_c = 0.05, 0.02

    def grad_u(self, u, ik):
        g, k = self.g, self.kpts[ik]
        uG = np.fft.fftn(u, axes=(-3, -2, -1))
        return np.stack([np.fft.ifftn(1j * (g.G[i] + k[i]) * self.full * uG, axes=(-3, -2, -1)) for i in range(3)])

    def div(self, F, ik):
        g, k = self.g, self.kpts[ik]
        return sum(np.fft.ifftn(1j * (g.G[i] + k[i]) * self.full * np.fft.fftn(F[i]), axes=(-3, -2, -1)) for i in range(3))

    def densities(self, orb):
        rho = np.full((2,) + self.g.shape, self.rho_c)
        tau = np.full((2,) + self.g.shape, self.tau_c)
        for (s, ik), (occ, _) in orb.items():
            for u in occ:
                rho[s] += self.wk[ik] * np.abs(u) ** 2
                tau[s] += 0.5 * self.wk[ik] * np.sum(np.abs(self.grad_u(u, ik)) ** 2, axis=0)
        return rho, tau

    def rotated(self, s, ik, i, a, t):
        orb = {key: (occ.copy(), vir) for key, (occ, vir) in self.orb.items()}
        occ, vir = orb[(s, ik)]
        occ[i] = np.cos(t) * self.orb[(s, ik)][0][i] + np.sin(t) * vir[a]
        return orb


def check(fname, tol=1e-7, h=2e-3):
    f = get(fname)
    toy = Toy()
    g = toy.g
    rho, tau = toy.densities(toy.orb)
    res = evaluate_on_grid(f, g, rho, tau)
    worst = 0.0
    for s in range(2):
        for ik in range(2):
            i, a = 0, 1
            psi, phi = toy.orb[(s, ik)][0][i], toy.orb[(s, ik)][1][a]
            gpsi = toy.grad_u(psi, ik)
            Hpsi = res.v[s] * psi - 0.5 * toy.div(res.v_tau[s] * gpsi, ik)
            an = 2.0 * toy.wk[ik] * np.real(g.weight * np.sum(np.conj(phi) * Hpsi))
            # the same pairing without integrating by parts (identity check of the operator form)
            gphi = toy.grad_u(phi, ik)
            an2 = 2.0 * toy.wk[ik] * g.weight * np.sum(res.v[s] * np.real(np.conj(phi) * psi)) \
                + toy.wk[ik] * g.weight * np.sum(res.v_tau[s] * np.real(np.sum(np.conj(gphi) * gpsi, axis=0)))
            E = {t: evaluate_on_grid(f, g, *toy.densities(toy.rotated(s, ik, i, a, t * h))).e_xc for t in (-2, -1, 1, 2)}
            fd = (8.0 * (E[1] - E[-1]) - (E[2] - E[-2])) / (12.0 * h)
            scale = abs(an) + 1e-3 * toy.wk[ik] * g.weight * np.sum(np.abs(res.v[s] * psi * phi))
            err = abs(fd - an) / scale
            worst = max(worst, err)
            assert abs(an - an2) < 1e-12 * scale, (s, ik, an, an2)
            assert err < tol, (fname, s, ik, fd, an, err)
    return worst


@pytest.mark.parametrize("name", ["model:test", "r2SCAN", "TPSS"])
def test_orbital_rotation(name):
    check(name)


@pytest.mark.node
@pytest.mark.parametrize("name", ["SCH-SI-A:v3.1", "SCH-SI-B:v3.1"])
def test_orbital_rotation_xchole(name):
    check(name)
