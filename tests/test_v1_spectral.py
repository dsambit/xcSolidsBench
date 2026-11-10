"""V1: spectral derivatives vs closed forms, adjointness, sphere bookkeeping (laptop, seconds)."""
import itertools

import numpy as np
import pytest

from xcsb.grid.spectral import GridError, SpectralGrid
from xcsb.testing import BandLimitedDensity, cells, grid_for

ECUT = 30.0          # Ry; small grids, exact arithmetic up to rounding
CELLS = cells()


def rel(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


@pytest.mark.parametrize("name", sorted(CELLS))
def test_derivatives_closed_form(name):
    g = grid_for(CELLS[name], ECUT)
    d = BandLimitedDensity(g, seed=3)
    rho = d.rho()
    grad, lapl, hess = g.derivatives(rho)
    assert rel(grad, d.grad()) < 1e-12
    assert rel(hess, d.hess()) < 1e-12
    assert rel(lapl, d.lapl()) < 1e-12
    assert rel(lapl, hess[:, 0] + hess[:, 3] + hess[:, 5]) < 1e-12
    assert rel(g.project(rho), rho) < 1e-13        # band-limited: P is the identity on it


@pytest.mark.parametrize("name", sorted(CELLS))
def test_adjointness(name):
    g = grid_for(CELLS[name], ECUT)
    rng = np.random.default_rng(7)
    f, h = rng.standard_normal(g.shape), rng.standard_normal(g.shape)
    gf, lf, hf = g.derivatives(f)
    gh, lh, hh = g.derivatives(h)
    scale = np.sqrt(np.sum(gf ** 2) * np.sum(h ** 2))
    for i in range(3):                                            # <D_i f, h> = -<f, D_i h>
        assert abs(np.sum(gf[i] * h) + np.sum(f * gh[i])) < 1e-12 * scale
    assert abs(np.sum(lf * h) - np.sum(f * lh)) < 1e-12 * np.sqrt(np.sum(lf ** 2) * np.sum(h ** 2))
    for c in range(6):                                            # D_iD_j symmetric
        assert abs(np.sum(hf[c] * h) - np.sum(f * hh[c])) < 1e-12 * np.sqrt(np.sum(hf[c] ** 2) * np.sum(h ** 2))
    # outputs are real: the imaginary parts dropped in ifft_real are at rounding level
    fG = g.mask * g.fft(f)
    im = np.max(np.abs(np.imag(np.fft.ifftn(1j * g.G[0] * fG))))
    assert im < 1e-13 * np.max(np.abs(gf[0]))


@pytest.mark.parametrize("name", sorted(CELLS))
def test_assemble_is_the_adjoint(name):
    """assemble(d_rho, d_grad, d_lapl, d_hess) . delta = sum of the pairings with the derivatives of delta (sphere-limited)."""
    g = grid_for(CELLS[name], ECUT)
    rng = np.random.default_rng(11)
    dr, dg, dl, dh = (rng.standard_normal(g.shape), rng.standard_normal((3,) + g.shape),
                      rng.standard_normal(g.shape), rng.standard_normal((6,) + g.shape))
    delta = g.project(rng.standard_normal(g.shape))
    gd, ld, hd = g.derivatives(delta)
    lhs = np.sum(g.assemble(dr, dg, dl, dh) * delta)
    rhs = np.sum(dr * delta) + np.sum(dg * gd) + np.sum(dl * ld) + np.sum(dh * hd)
    assert abs(lhs - rhs) < 1e-11 * (abs(rhs) + np.sqrt(np.sum(dr ** 2) * np.sum(delta ** 2)))


@pytest.mark.parametrize("name", sorted(CELLS))
def test_sphere_count_brute_force(name):
    """Sigma M equals an independent enumeration of the integer vectors with |G|^2 <= ecutrho."""
    g = grid_for(CELLS[name], ECUT)
    B = g.recip
    nmax = [int(np.sqrt(ECUT) * np.linalg.norm(g.lattice[i]) / (2 * np.pi)) + 2 for i in range(3)]
    count = 0
    for n in itertools.product(*[range(-m, m + 1) for m in nmax]):
        G = n[0] * B[0] + n[1] * B[1] + n[2] * B[2]
        count += G @ G <= ECUT
    assert g.ngm == count


def test_nyquist_raises():
    with pytest.raises(GridError):                       # an even grid much smaller than 2 max|n| + 1:
        SpectralGrid(CELLS["fcc"], [4, 4, 4], ECUT)      # its Nyquist plane lies inside the sphere
