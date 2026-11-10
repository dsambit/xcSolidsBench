"""libxc functionals through pyscf's bundled libxc binding (pyscf.dft.libxc.eval_xc, spin-polarized form).

pyscf array convention (pinned by tests/test_v1_adapters.py against finite differences): rho is (2, nvar, N) with rows
(den, grad_x, grad_y, grad_z) for GGAs and (den, grad_x, grad_y, grad_z, laplacian, tau) for meta-GGAs, tau with the 1/2;
exc is per particle; vxc = (vrho (N, 2), vsigma (N, 3: uu, ud, dd), vlapl, vtau (N, 2)).
"""
from __future__ import annotations

import numpy as np

from .contract import PointFunctional, PointInputs, PointOutputs

LIBXC_CODES = {
    "PBE": "GGA_X_PBE,GGA_C_PBE",
    "r2SCAN": "MGGA_X_R2SCAN,MGGA_C_R2SCAN",
    "SCAN": "MGGA_X_SCAN,MGGA_C_SCAN",
    "TPSS": "MGGA_X_TPSS,MGGA_C_TPSS",
}


class LibxcFunctional(PointFunctional):
    uses_hessian = False

    def __init__(self, name: str, code: str | None = None):
        from pyscf.dft import libxc
        self.name = name
        self.code = code or LIBXC_CODES[name]
        self._libxc = libxc
        kind = libxc.xc_type(self.code)
        if kind not in ("GGA", "MGGA"):
            raise ValueError(f"{name}: libxc type {kind} not supported (GGA or MGGA only)")
        if libxc.is_hybrid_xc(self.code):
            raise ValueError(f"{name}: hybrid functionals are not supported (no exact exchange in the host)")
        self.is_meta = kind == "MGGA"
        self.uses_laplacian = bool(getattr(libxc, "needs_laplacian", lambda c: False)(self.code))

    @property
    def provenance(self) -> dict:
        import pyscf
        return {"name": self.name, "kind": "libxc", "code": self.code, "pyscf": pyscf.__version__,
                "libxc": self._libxc.libxc_version() if hasattr(self._libxc, "libxc_version") else None}

    def evaluate(self, x: PointInputs) -> PointOutputs:
        n = x.n
        nvar = 6 if self.is_meta else 4
        r = np.zeros((2, nvar, n))
        r[:, 0] = x.rho
        r[:, 1:4] = x.grad
        if self.is_meta:
            r[:, 4] = x.lapl
            r[:, 5] = x.tau
        exc, vxc = self._libxc.eval_xc(self.code, r, spin=1, deriv=1)[:2]
        vrho, vsigma = np.asarray(vxc[0]), np.asarray(vxc[1])
        rho_t = x.rho[0] + x.rho[1]
        e = np.asarray(exc).reshape(n) * rho_t
        g_u, g_d = x.grad[0], x.grad[1]
        d_grad = np.empty((2, 3, n))
        d_grad[0] = 2.0 * vsigma[:, 0] * g_u + vsigma[:, 1] * g_d
        d_grad[1] = 2.0 * vsigma[:, 2] * g_d + vsigma[:, 1] * g_u
        d_tau = np.zeros((2, n))
        d_lapl = np.zeros((2, n))
        if self.is_meta:
            d_tau[:] = np.asarray(vxc[3]).T
            if vxc[2] is not None:
                d_lapl[:] = np.asarray(vxc[2]).T
        return PointOutputs(float(np.sum(e)), np.ascontiguousarray(vrho.T), d_grad, d_tau, d_lapl,
                            np.zeros((2, 6, n)), extra={"e": e})
