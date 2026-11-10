"""QE <-> contract layouts and units (DESIGN.md Sec. 4.2), without QE: pure numpy, testable on the laptop.

QE side (gathered arrays, Rydberg units), as returned by QEpy:
    rho_qe  (npts, nspin): nspin = 1: total density; nspin = 2: (up, down)       [qepy_get_rho converts (tot, mag) -> (up, down)]
    kin_qe  (npts, nspin): rho%kin_r = sum_i w_i |grad psi_i|^2 per spin (= 2 tau_Ha); nspin = 1: both spins together
Grid order: QE's real-space order, x fastest (Fortran order of (nr1, nr2, nr3)).

Contract side: (2, n1, n2, n3) arrays, Hartree units, tau with the 1/2.

Back to QE: extpot (Ry) = 2 v_sigma (nspin = 2) or v_up + v_down (nspin = 1, the exact chain rule for rho_sigma = rho/2);
extkin = v_tau_sigma (nspin = 2) or (v_tau_up + v_tau_down)/2 (nspin = 1); extene (Ry) = 2 E_xc.
"""
from __future__ import annotations

import numpy as np

RY_PER_HA = 2.0


def qe_to_grid(a: np.ndarray, shape) -> np.ndarray:
    """(npts,) QE order -> (n1, n2, n3)."""
    return np.asarray(a, dtype=np.float64).reshape(tuple(shape), order="F")


def grid_to_qe(f: np.ndarray) -> np.ndarray:
    """(n1, n2, n3) -> (npts,) QE order."""
    return np.asarray(f, dtype=np.float64).reshape(-1, order="F")


def spin_channels(nspin: int, rho_qe: np.ndarray, kin_qe: np.ndarray, shape):
    """(rho (2, ...), tau (2, ...)) in Hartree units from QE's gathered arrays."""
    if nspin == 1:
        rho = qe_to_grid(rho_qe[:, 0], shape)
        tau = qe_to_grid(kin_qe[:, 0], shape) / RY_PER_HA            # total tau_Ha
        return np.stack([0.5 * rho, 0.5 * rho]), np.stack([0.5 * tau, 0.5 * tau])
    if nspin == 2:
        rho = np.stack([qe_to_grid(rho_qe[:, s], shape) for s in range(2)])
        tau = np.stack([qe_to_grid(kin_qe[:, s], shape) for s in range(2)]) / RY_PER_HA
        return rho, tau
    raise ValueError(f"nspin = {nspin}: only collinear nspin 1 or 2 (non-collinear is out of scope)")


def qe_potentials(nspin: int, e_xc_ha: float, v: np.ndarray, v_tau: np.ndarray):
    """(extpot (npts, nspin) Ry, extkin (npts, nspin), extene Ry) from contract-side results."""
    if nspin == 1:
        extpot = grid_to_qe(v[0] + v[1])[:, None]
        extkin = grid_to_qe(0.5 * (v_tau[0] + v_tau[1]))[:, None]
    elif nspin == 2:
        extpot = np.stack([RY_PER_HA * grid_to_qe(v[s]) for s in range(2)], axis=1)
        extkin = np.stack([grid_to_qe(v_tau[s]) for s in range(2)], axis=1)
    else:
        raise ValueError(f"nspin = {nspin}: only collinear nspin 1 or 2")
    return np.asfortranarray(extpot), np.asfortranarray(extkin), RY_PER_HA * float(e_xc_ha)
