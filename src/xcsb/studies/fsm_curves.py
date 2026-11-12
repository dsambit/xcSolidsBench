"""E(m) curve helpers of the fixed-spin-moment studies: the spline minimum, the even-fit curvature at m = 0, the spline curvature."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar


def minimum(ms, es):
    cs = CubicSpline(ms, es)
    i = int(np.argmin(es))
    lo, hi = ms[max(i - 1, 0)], ms[min(i + 1, len(ms) - 1)]
    if lo == hi:
        return float(ms[i]), float(es[i])
    r = minimize_scalar(cs, bounds=(lo, hi), method="bounded")
    return float(r.x), float(r.fun)


def curv0(ms, y, mmax=0.3):
    """E''(0) from the even fit y = c0 + c2 m^2 + c4 m^4 over m <= mmax (Ry per mu_B^2): 2 c2."""
    sel = ms <= mmax + 1e-12
    if np.count_nonzero(sel) < 3:
        return None
    A = np.stack([np.ones(np.count_nonzero(sel)), ms[sel] ** 2, ms[sel] ** 4], axis=1)
    c = np.linalg.lstsq(A, y[sel], rcond=None)[0]
    return float(2.0 * c[1])


def curv_at(ms, y, m0):
    return float(CubicSpline(ms, y)(m0, 2))
