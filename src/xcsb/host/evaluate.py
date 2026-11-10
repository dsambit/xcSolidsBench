"""The host step without QE (DESIGN.md Sec. 4.3-4.6): spin densities and tau on the grid -> E_xc, v_sigma, v_tau_sigma.

Collective over comm: the grid arrays are given on the root (None elsewhere); the result is returned on the root.
Units: Hartree atomic units throughout (QE layout and Rydberg conversion live in host/qe.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..functional.contract import NegativeDensityError, PointFunctional, PointInputs
from ..grid.spectral import SpectralGrid
from ..parallel.mpi import CHUNK, SerialComm, evaluate_points


@dataclass
class GridResult:
    e_xc: float                      # Ha
    v: np.ndarray                    # (2, n1, n2, n3) Ha, multiplicative potential per spin
    v_tau: np.ndarray                # (2, n1, n2, n3) dimensionless, d e / d tau_sigma (projected)
    stats: dict = field(default_factory=dict)


def density_stats(grid: SpectralGrid, rho, tau, grad):
    tw = np.where(rho > 0, np.sum(grad ** 2, axis=1) / (8.0 * np.where(rho > 0, rho, 1.0)), 0.0)
    below = tau < tw
    n_el = [grid.integrate(rho[s]) for s in range(2)]
    return {
        "n_electrons": n_el,
        "min_rho": [float(rho[s].min()) for s in range(2)],
        "frac_points_tau_below_tauW": [float(np.mean(below[s])) for s in range(2)],
        "min_tau_minus_tauW": [float((tau[s] - tw[s]).min()) for s in range(2)],
    }


def point_inputs(grid: SpectralGrid, rho, tau, need_lapl: bool, need_hess: bool):
    """PointInputs (2, N) from grid arrays rho, tau (2, n1, n2, n3); also returns grad (2, 3, n1, n2, n3)."""
    grad, lapl, hess = grid.derivatives(rho, laplacian=need_lapl, hessian=need_hess)
    n = grid.npts
    x = PointInputs(
        rho=np.ascontiguousarray(rho.reshape(2, n)),
        grad=np.ascontiguousarray(grad.reshape(2, 3, n)),
        tau=np.ascontiguousarray(tau.reshape(2, n)),
        lapl=np.zeros((2, n)) if lapl is None else np.ascontiguousarray(lapl.reshape(2, n)),
        hess=np.zeros((2, 6, n)) if hess is None else np.ascontiguousarray(hess.reshape(2, 6, n)),
    )
    return x, grad


def evaluate_on_grid(functional: PointFunctional, grid: SpectralGrid | None, rho, tau, comm=None,
                     chunk: int = CHUNK) -> GridResult | None:
    comm = comm or SerialComm()
    root = comm.Get_rank() == 0
    x = None
    stats = None
    if root:
        rho = np.asarray(rho, dtype=np.float64)
        tau = np.asarray(tau, dtype=np.float64)
        if rho.shape != (2,) + grid.shape or tau.shape != (2,) + grid.shape:
            raise ValueError(f"rho {rho.shape}, tau {tau.shape}: expected (2,) + {grid.shape}")
        x, grad = point_inputs(grid, rho, tau, functional.uses_laplacian, functional.uses_hessian)
        stats = density_stats(grid, rho, tau, grad)
        bad = None
        if np.any(rho < 0):
            neg = rho < 0
            bad = (f"negative spin density at {int(np.count_nonzero(neg))} grid points: min {rho.min():.3e}, "
                   f"integrated negative charge {grid.weight * float(-rho[neg].sum()):.3e}")
    else:
        bad = None
    bad = comm.bcast(bad, root=0)
    if bad:
        raise NegativeDensityError(bad)
    out = evaluate_points(functional, x, comm, chunk)
    if not root:
        return None
    shp = grid.shape
    v = np.empty((2,) + shp)
    v_tau = np.empty((2,) + shp)
    for s in range(2):
        v[s] = grid.assemble(out.d_rho[s].reshape(shp), out.d_grad[s].reshape((3,) + shp),
                             out.d_lapl[s].reshape(shp) if functional.uses_laplacian else None,
                             out.d_hess[s].reshape((6,) + shp) if functional.uses_hessian else None)
        v_tau[s] = grid.project(out.d_tau[s].reshape(shp))
    return GridResult(e_xc=grid.weight * out.e_sum, v=v, v_tau=v_tau, stats=stats)
