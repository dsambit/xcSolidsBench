"""Reading QE's saved densities (no QE needed): charge-density.dat and ekin-density.dat of a <prefix>.save directory, written by
QE 7.6 Modules/io_base.f90 write_rhog as Fortran unformatted sequential records (no HDF5 in our build):

    1: gamma_only (logical), ngm_g, nspin (int32)        2: b1, b2, b3 (9 float64, 2 pi/alat units)
    3: Miller indices (3, ngm_g) int32                   4..3+nspin: rho_g(1:ngm_g) complex128, one record per component

charge-density: rho%of_g, for nspin = 2 (total, magnetization) (QE's rhoz layout; HDF5 dataset names rhotot_g, rhodiff_g);
ekin-density: rho%kin_g per spin (up, down), kin = sum_i w_i |grad psi_i|^2 = 2 tau_Ha (PW/src/io_rho_xml.f90).
Fields on QE's dense grid: f(r) = sum_G f_G exp(i G.r), i.e. f = N ifftn(F) with F[mill mod n] = f_G.
Used only for frozen diagnostics of converged states (never to start an SCF from)."""
from __future__ import annotations

import os
import re

import numpy as np

from .layout import RY_PER_HA


def _records(path):
    raw = open(path, "rb").read()
    out, i = [], 0
    while i < len(raw):
        n = int(np.frombuffer(raw, np.int32, 1, i)[0])
        out.append(raw[i + 4:i + 4 + n])
        if int(np.frombuffer(raw, np.int32, 1, i + 4 + n)[0]) != n:
            raise ValueError(f"{path}: corrupt Fortran record at byte {i}")
        i += 8 + n
    return out


def read_rhog(path):
    """(mill (ngm, 3) int, rho_g (nspin, ngm) complex) of one QE density file."""
    r = _records(path)
    gamma = bool(np.frombuffer(r[0], np.int32, 1, 0)[0])
    ngm, nspin = (int(x) for x in np.frombuffer(r[0], np.int32, 2, 4))
    if gamma:
        raise ValueError(f"{path}: gamma_only files are not supported (half of the G sphere)")
    mill = np.frombuffer(r[2], np.int32).reshape(ngm, 3)
    if len(r) != 3 + nspin:
        raise ValueError(f"{path}: {len(r)} records for nspin {nspin}")
    rho = np.stack([np.frombuffer(r[3 + s], np.complex128) for s in range(nspin)])
    if rho.shape != (nspin, ngm):
        raise ValueError(f"{path}: component shape {rho.shape}")
    return mill, rho


IMAG_TOL = 1e-5      # relative imaginary part tolerated (QE's saved LSDA magnetization coefficients are Hermitian only to ~1e-7)


def to_grid(mill, coeffs, shape, info=None):
    """Real-space values (ncomp, n1, n2, n3) of sum_G c_G exp(i G.r) on QE's grid (the real part; the imaginary part of each
    component relative to the largest value of the FIRST component (the total density, resp. the up channel; a magnetization can
    vanish) goes to info['rel_imag'] and raises above IMAG_TOL); raises if a G vector does not fit."""
    n = np.array(shape)
    if np.any(np.abs(mill) > (n - 1) // 2):
        raise ValueError(f"Miller indices up to {np.abs(mill).max(0)} do not fit the grid {tuple(shape)}")
    idx = tuple((mill % n).T)
    out, scale = [], None
    for c in coeffs:
        F = np.zeros(tuple(shape), complex)
        F[idx] = c
        f = np.fft.ifftn(F) * F.size
        scale = float(np.max(np.abs(f.real))) if scale is None else scale
        rel = float(np.max(np.abs(f.imag)) / max(scale, 1e-300))
        if info is not None:
            info.setdefault("rel_imag", []).append(rel)
        if rel > IMAG_TOL:
            raise ValueError(f"the field is not real (relative imaginary part {rel:.2e})")
        out.append(f.real)
    return np.stack(out)


def dense_grid_shape(pw_out):
    """QE's dense FFT grid from pw.out ('Dense  grid: ... FFT dimensions: ( n1, n2, n3)')."""
    m = re.search(r"Dense\s+grid:.*?FFT dimensions:\s*\(\s*(\d+),\s*(\d+),\s*(\d+)\)", open(pw_out).read())
    if not m:
        raise ValueError(f"{pw_out}: no dense FFT grid line")
    return tuple(int(x) for x in m.groups())


def saved_spin_fields(run_dir, prefix="xcsb", info=None):
    """(rho, tau, shape): (2, n1, n2, n3) Hartree, per spin, from <run_dir>/out/<prefix>.save and the grid of <run_dir>/pw.out."""
    save = os.path.join(run_dir, "out", f"{prefix}.save")
    shape = dense_grid_shape(os.path.join(run_dir, "pw.out"))
    mr, cr = read_rhog(os.path.join(save, "charge-density.dat"))
    mk, ck = read_rhog(os.path.join(save, "ekin-density.dat"))
    rho = to_grid(mr, cr, shape, info)
    kin = to_grid(mk, ck, shape, info)
    if rho.shape[0] == 2:                                    # (total, magnetization) -> (up, down)
        rho = np.stack([0.5 * (rho[0] + rho[1]), 0.5 * (rho[0] - rho[1])])
        tau = kin / RY_PER_HA
    elif rho.shape[0] == 1:
        rho = np.stack([0.5 * rho[0], 0.5 * rho[0]])
        tau = np.stack([0.5 * kin[0], 0.5 * kin[0]]) / RY_PER_HA
    else:
        raise ValueError(f"nspin {rho.shape[0]}: collinear only")
    return rho, tau, shape
