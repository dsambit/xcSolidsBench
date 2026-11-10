"""Spectral derivatives on a periodic FFT grid with the ecutrho-sphere truncation, and the adjoint potential assembly
(DESIGN.md Sec. 4.3).

Grid: N = n1 n2 n3 points r = (i/n1) a1 + (j/n2) a2 + (k/n3) a3, arrays indexed [..., i, j, k]. Lattice rows a_i in bohr.
G = n1 b1 + n2 b2 + n3 b3 with signed integers n (numpy fftfreq order), b_i the reciprocal basis (a_i . b_j = 2 pi delta_ij).
Sphere mask M(G) = [|G|^2 <= ecutrho] (Rydberg units: |G|^2 in bohr^-2 equals the energy in Ry, as in QE's ggen).

    D_i = F^-1 M (i G_i) F,   lap = F^-1 M (-|G|^2) F,   H_ij = F^-1 M (-G_i G_j) F,   P = F^-1 M F.

D_i is real and M even in G with the Nyquist index excluded (checked), so D_i^T = -D_i and the second-order operators are
symmetric. The derivative of E_grid = w sum_k e(inputs at r_k) with respect to the grid values of rho_sigma, divided by
w = Omega/N, restricted to sphere-limited variations, is

    v = P [ d_rho - sum_i D_i d_grad_i + lap d_lapl + sum_{i<=j} (D_i D_j) d_hess_ij ],   v_tau = P [ d_tau ].
"""
from __future__ import annotations

import numpy as np

from ..functional.contract import HESS_PAIRS


class GridError(ValueError):
    pass


class SpectralGrid:
    def __init__(self, lattice, shape, ecutrho_ry: float):
        self.lattice = np.asarray(lattice, dtype=np.float64).reshape(3, 3)
        self.shape = tuple(int(n) for n in shape)
        self.ecutrho = float(ecutrho_ry)
        self.volume = abs(float(np.linalg.det(self.lattice)))
        self.npts = int(np.prod(self.shape))
        self.weight = self.volume / self.npts
        self.recip = 2.0 * np.pi * np.linalg.inv(self.lattice).T               # rows b_i
        ints = [np.fft.fftfreq(n, 1.0 / n) for n in self.shape]                 # signed integers
        n1, n2, n3 = np.meshgrid(*ints, indexing="ij")
        self.G = (n1[..., None] * self.recip[0] + n2[..., None] * self.recip[1] + n3[..., None] * self.recip[2])
        self.G = np.moveaxis(self.G, -1, 0)                                      # (3, n1, n2, n3)
        self.G2 = np.sum(self.G ** 2, axis=0)
        self.mask = self.G2 <= self.ecutrho
        for ax, (n, k) in enumerate(zip(self.shape, (n1, n2, n3))):
            if n % 2 == 0 and np.any(self.mask & (np.abs(k) == n // 2)):
                raise GridError(f"the ecutrho sphere reaches the Nyquist index along axis {ax} (n = {n})")
        self.ngm = int(np.count_nonzero(self.mask))

    # -- transforms ------------------------------------------------------------------------------------------------
    def fft(self, f):
        return np.fft.fftn(f, axes=(-3, -2, -1))

    def ifft_real(self, fG):
        return np.real(np.fft.ifftn(fG, axes=(-3, -2, -1)))

    def project(self, f):
        """P f: the component of a real grid function inside the ecutrho sphere."""
        return self.ifft_real(self.mask * self.fft(f))

    # -- derivatives -----------------------------------------------------------------------------------------------
    def derivatives(self, f, laplacian=True, hessian=True):
        """grad (..., 3, n1, n2, n3), lapl (..., n1, n2, n3) or None, hess (..., 6, n1, n2, n3) or None of real f."""
        fG = self.mask * self.fft(f)
        grad = np.stack([self.ifft_real(1j * self.G[i] * fG) for i in range(3)], axis=-4)
        lapl = self.ifft_real(-self.G2 * fG) if laplacian else None
        hess = (np.stack([self.ifft_real(-self.G[i] * self.G[j] * fG) for i, j in HESS_PAIRS], axis=-4)
                if hessian else None)
        return grad, lapl, hess

    def assemble(self, d_rho, d_grad, d_lapl=None, d_hess=None):
        """v = P[d_rho - sum_i D_i d_grad_i + lap d_lapl + sum_{i<=j} D_iD_j d_hess_ij] for one spin channel."""
        vG = self.fft(d_rho)
        for i in range(3):
            vG = vG - 1j * self.G[i] * self.fft(d_grad[i])
        if d_lapl is not None:
            vG = vG - self.G2 * self.fft(d_lapl)
        if d_hess is not None:
            for c, (i, j) in enumerate(HESS_PAIRS):
                vG = vG - self.G[i] * self.G[j] * self.fft(d_hess[c])
        return self.ifft_real(self.mask * vG)

    # -- helpers ---------------------------------------------------------------------------------------------------
    def points(self):
        """Cartesian coordinates of the grid points, (3, n1, n2, n3), bohr."""
        fr = np.meshgrid(*[np.arange(n) / n for n in self.shape], indexing="ij")
        return sum(fr[i][None] * self.lattice[i][:, None, None, None] for i in range(3))

    def integrate(self, f):
        return self.weight * float(np.sum(f))
