"""Shared fixtures for the verification tiers: test cells and band-limited periodic densities with closed-form derivatives."""
from __future__ import annotations

import numpy as np

from .functional.contract import HESS_PAIRS
from .grid.spectral import SpectralGrid

BOHR_PER_ANG = 1.0 / 0.529177210903


def cells():
    """Lattices (rows, bohr) for the tests: triclinic, bcc, fcc, hcp and the rocksalt AFM-II rhombohedral cell."""
    a = 5.0
    return {
        "triclinic": np.array([[5.1, 0.2, -0.3], [0.7, 5.6, 0.1], [-0.4, 0.9, 6.2]]),
        "bcc": 0.5 * a * np.array([[-1.0, 1, 1], [1, -1, 1], [1, 1, -1]]),
        "fcc": 0.5 * a * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]]),
        "hcp": np.array([[a, 0, 0], [-0.5 * a, 0.5 * np.sqrt(3) * a, 0], [0, 0, 1.63 * a]]),
        "afm2": 4.2 * np.array([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]]),
    }


def grid_for(lattice, ecutrho, margin=1):
    """The smallest grid with nr_i = 2 max|n_i| + 1 (+ margin), as QE's rule (without the good-FFT rounding)."""
    lat = np.asarray(lattice)
    nmax = [int(np.sqrt(ecutrho) * np.linalg.norm(lat[i]) / (2 * np.pi)) + 1 for i in range(3)]
    return SpectralGrid(lat, [2 * n + 1 + margin for n in nmax], ecutrho)


class BandLimitedDensity:
    """rho_sigma(r) = c_sigma + sum_m a_sigma,m cos(G_m . r + phi_sigma,m) with G_m inside the ecutrho sphere: every
    derivative is a finite Fourier series, computed in closed form."""

    def __init__(self, grid: SpectralGrid, nterms=8, c=(0.3, 0.2), amp=0.02, seed=0):
        rng = np.random.default_rng(seed)
        idx = np.argwhere(grid.mask & (grid.G2 > 0) & (grid.G2 < 0.6 * grid.ecutrho))
        pick = idx[rng.choice(len(idx), size=nterms, replace=False)]
        self.G = np.array([grid.G[:, i, j, k] for i, j, k in pick])             # (m, 3)
        self.c = np.asarray(c, dtype=float)
        self.a = amp * rng.uniform(0.5, 1.0, size=(2, nterms))
        self.phi = rng.uniform(0, 2 * np.pi, size=(2, nterms))
        self.r = grid.points()

    def _phase(self, s):
        return np.einsum("mi,i...->m...", self.G, self.r) + self.phi[s][:, None, None, None]

    def rho(self):
        return np.stack([self.c[s] + np.einsum("m,m...->...", self.a[s], np.cos(self._phase(s))) for s in range(2)])

    def grad(self):
        out = []
        for s in range(2):
            sn = np.sin(self._phase(s))
            out.append(np.stack([-np.einsum("m,m...->...", self.a[s] * self.G[:, i], sn) for i in range(3)]))
        return np.stack(out)

    def hess(self):
        out = []
        for s in range(2):
            cs = np.cos(self._phase(s))
            out.append(np.stack([-np.einsum("m,m...->...", self.a[s] * self.G[:, i] * self.G[:, j], cs)
                                 for i, j in HESS_PAIRS]))
        return np.stack(out)

    def lapl(self):
        h = self.hess()
        return h[:, 0] + h[:, 3] + h[:, 5]


def tau_above_tw(grid, rho, factor=1.3, extra=0.01, seed=1):
    """A smooth tau_sigma >= tau_W,sigma: factor * tau_W + extra * rho^(5/3) * (1 + small band-limited ripple)."""
    grad, _, _ = grid.derivatives(rho, laplacian=False, hessian=False)
    tw = np.sum(grad ** 2, axis=1) / (8 * rho)
    return factor * tw + extra * rho ** (5.0 / 3.0)


def sphere_limited_random(grid: SpectralGrid, seed=0, frac=0.8):
    """A random real grid function inside the sphere |G|^2 <= frac * ecutrho, unit max norm."""
    rng = np.random.default_rng(seed)
    f = grid.ifft_real((grid.G2 <= frac * grid.ecutrho) * grid.fft(rng.standard_normal(grid.shape)))
    return f / np.abs(f).max()
