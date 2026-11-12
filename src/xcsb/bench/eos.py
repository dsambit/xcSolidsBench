"""Third-order Birch-Murnaghan E(V) fit (DESIGN Sec. 6.3, one EOS form): E = c0 + c1 x + c2 x^2 + c3 x^3 with x = V^(-2/3) (linear
least squares; exactly the BM3 family), V0 from dE/dV = 0, B0 = V E''(V0), B0' from the third derivative. Raises if V0 lies
outside the inner two thirds of the scanned volumes or if the rms residual exceeds RMS_TOL (global)."""
from __future__ import annotations

import numpy as np

RMS_TOL_RY = 2e-5            # Ry per cell: the global residual threshold of the fit (SCF energies converged to ~1e-8 Ry)
RY_BOHR3_TO_GPA = 14710.507848260711


class EOSError(RuntimeError):
    pass


def bm3_fit(V, E):
    V, E = np.asarray(V, float), np.asarray(E, float)
    if len(V) < 5:
        raise EOSError("BM3 fit needs at least 5 points")
    x = V ** (-2.0 / 3.0)
    A = np.stack([np.ones_like(x), x, x ** 2, x ** 3], axis=1)
    c, *_ = np.linalg.lstsq(A, E, rcond=None)
    resid = E - A @ c
    rms = float(np.sqrt(np.mean(resid ** 2)))
    # dE/dx = c1 + 2 c2 x + 3 c3 x^2 = 0 inside the scanned range
    roots = np.roots([3 * c[3], 2 * c[2], c[1]])
    xs = [float(r.real) for r in roots if abs(r.imag) < 1e-12 and x.min() * 0.8 <= r.real <= x.max() * 1.2]
    if not xs:
        raise EOSError("no stationary point of the BM3 fit near the scanned volumes")
    d2 = lambda xx: 2 * c[2] + 6 * c[3] * xx                     # noqa: E731  d2E/dx2
    xs = [xx for xx in xs if d2(xx) > 0]
    if len(xs) != 1:
        raise EOSError(f"BM3 fit: {len(xs)} minima near the scanned volumes")
    x0 = xs[0]
    V0 = x0 ** -1.5
    lo, hi = V.min(), V.max()
    if not (lo + (hi - lo) / 6 <= V0 <= hi - (hi - lo) / 6):
        raise EOSError(f"V0 {V0:.4f} outside the inner two thirds of [{lo:.4f}, {hi:.4f}]: a rescan centred at V0 is the user's call")
    if rms > RMS_TOL_RY:
        raise EOSError(f"BM3 rms residual {rms:.2e} Ry > {RMS_TOL_RY:.0e}")
    # E(V) derivatives through x(V) = V^(-2/3): dx/dV = -2/3 V^(-5/3), d2x/dV2 = 10/9 V^(-8/3), d3x/dV3 = -80/27 V^(-11/3)
    dEx = lambda xx: c[1] + 2 * c[2] * xx + 3 * c[3] * xx ** 2      # noqa: E731
    d3 = 6 * c[3]
    x1, x2, x3 = -2 / 3 * V0 ** (-5 / 3), 10 / 9 * V0 ** (-8 / 3), -80 / 27 * V0 ** (-11 / 3)
    E2 = d2(x0) * x1 ** 2 + dEx(x0) * x2
    E3 = d3 * x1 ** 3 + 3 * d2(x0) * x1 * x2 + dEx(x0) * x3
    B0 = V0 * E2
    B0p = -1.0 - V0 * E3 / E2
    E0 = float(c @ np.array([1, x0, x0 ** 2, x0 ** 3]))
    return {"V0_bohr3": float(V0), "E0_ry": E0, "B0_GPa": float(B0 * RY_BOHR3_TO_GPA), "B0_prime": float(B0p),
            "rms_ry": rms, "coeffs": c.tolist(), "n_points": int(len(V))}
