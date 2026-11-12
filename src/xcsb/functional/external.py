"""External per-point functionals from the pinned xchole tree (read-only), wired into the host like the libxc adapter: inputs
(rho, grad rho, tau) per spin, exact first derivatives supplied by the external code, no Laplacian, no Hessian.

First use (user 2026-11-11): the Perdew-group spin-balanced r2SCAN as implemented on strongxc-v4 (explore/spin_balance/
sb_r2scan.py), names 'r2SCAN-SB' (R0 half split, the primary) and 'r2SCAN-SB-kin'. xchole is pinned through the line
'v4-sb' (functional.xchole.XCHOLE_PINS); the module file's sha256 goes into the provenance.

The interface used (strongxc-v4 b2501727c sb_r2scan.py; EXTERNAL[name] = (path relative to the xchole root, function)):

    evaluate_arrays(rho, grad, tau, r0_form=NAMES[name], omega=OMEGA) -> dict(e=(N,), d_rho=(2, N), d_grad=(2, 3, N), d_tau=(2, N), parts)

Hartree atomic units; rho (2, N) per spin; grad (2, 3, N) the vectors grad rho_s; tau (2, N) = (1/2) sum_i |grad phi_i,s|^2;
e the XC energy per volume; d_* the exact per-point partial derivatives of e. Zero-density points (the padding of the last
chunk) must give e = 0 with finite derivatives. The adapter checks shapes and finiteness and raises otherwise."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import re

import numpy as np

from .contract import PointFunctional, PointInputs, PointOutputs

EXTERNAL = {"r2SCAN-SB": ("explore/spin_balance/sb_r2scan.py", "evaluate_arrays"),
            "r2SCAN-SB-kin": ("explore/spin_balance/sb_r2scan.py", "evaluate_arrays")}
LINE = {"r2SCAN-SB": "v4-sb", "r2SCAN-SB-kin": "v4-sb"}


def parse_external(name: str):
    """'r2SCAN-SB' -> ('r2SCAN-SB', None); 'r2SCAN-SB-kin(omega=10)' -> ('r2SCAN-SB-kin', 10.0): a gate-width DIAGNOSTIC (the
    published functional has omega 90; omega = 0 is the lam = 1 ceiling, diagnostic only)."""
    m = re.fullmatch(r"(r2SCAN-SB(?:-kin)?)(?:\(omega=([0-9.eE+-]+)\))?", name)
    if not m:
        raise KeyError(f"{name!r}: external functionals {sorted(EXTERNAL)} (optionally '(omega=<w>)')")
    w = None if m.group(2) is None else float(m.group(2))
    if w is not None and w < 0:
        raise ValueError(f"{name}: omega must be >= 0")
    return m.group(1), w


class ExternalFunctional(PointFunctional):
    uses_laplacian = False
    uses_hessian = False
    is_meta = True

    def __init__(self, name: str):
        from .xchole import XCHOLE_LINE, import_pinned_xchole
        base, w_override = parse_external(name)
        if XCHOLE_LINE != LINE[base]:
            raise RuntimeError(f"{name} needs XCSB_XCHOLE_LINE={LINE[base]} (now {XCHOLE_LINE})")
        _, commit = import_pinned_xchole()
        root = os.path.dirname(os.path.abspath(os.environ["XCSB_XCHOLE_SRC"]))
        rel, fn = EXTERNAL[base]
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path}: the external functional is not in the pinned tree {commit}")
        spec = importlib.util.spec_from_file_location(f"xcsb_external_{os.path.basename(path)[:-3]}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        r0_form = mod.NAMES[base]
        omega = float(mod.OMEGA) if w_override is None else w_override
        fun = getattr(mod, fn)
        self.name = name
        self._f = lambda nm, rho, grad, tau: fun(rho, grad, tau, r0_form=r0_form, omega=omega)
        self._prov = {"name": name, "kind": "external", "xchole_commit": commit, "module": rel, "function": fn,
                      "r0_form": r0_form, "omega": omega, "omega_is_module_default": w_override is None,
                      "module_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}

    @property
    def provenance(self) -> dict:
        return dict(self._prov)

    def evaluate(self, x: PointInputs) -> PointOutputs:
        n = x.rho.shape[1]
        o = self._f(self.name, np.asarray(x.rho), np.asarray(x.grad), np.asarray(x.tau))
        want = {"e": (n,), "d_rho": (2, n), "d_grad": (2, 3, n), "d_tau": (2, n)}
        for k, shp in want.items():
            if k not in o or np.shape(o[k]) != shp:
                raise ValueError(f"{self.name}: output {k} has shape {np.shape(o.get(k))}, expected {shp}")
            if not np.all(np.isfinite(o[k])):
                raise FloatingPointError(f"{self.name}: non-finite {k}")
        e = np.asarray(o["e"], float)
        return PointOutputs(float(e.sum()), np.asarray(o["d_rho"], float), np.asarray(o["d_grad"], float),
                            np.asarray(o["d_tau"], float), np.zeros((2, n)), np.zeros((2, 6, n)), extra={"e": e})


def sb_parts(name, grid, rho, tau, r3d_bohr=1.3):
    """Post diagnostic of an r2SCAN-SB run on its converged density: the term Delta = lam delta_m integrated (Ry) overall and in the
    3d region (within r3d_bohr of the first atom: one-atom M1/SB cells), the |delta_m|-weighted lam distribution and the electron
    fraction with lam < 0.5, overall and 3d (the module's evaluate_arrays parts)."""
    from ..host.evaluate import point_inputs
    f = ExternalFunctional(name)
    x, _ = point_inputs(grid, rho, tau, False, False)
    o = f._f(name, x.rho, x.grad, x.tau)
    pr = o["parts"]
    fr = np.stack(np.meshgrid(*[np.arange(n) / n for n in grid.shape], indexing="ij")).reshape(3, -1)
    best = np.full(fr.shape[1], np.inf)
    for sh in np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij")).reshape(3, -1).T:
        best = np.minimum(best, np.linalg.norm((fr - np.round(fr) + sh[:, None]).T @ grid.lattice, axis=1))
    n = x.rho.sum(0)
    edges = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0 + 1e-12)
    out = {"omega": f.provenance["omega"], "r0_form": f.provenance["r0_form"], "r3d_bohr": r3d_bohr}
    for reg, sel in (("all", np.ones_like(n, bool)), ("3d", best <= r3d_bohr)):
        w = grid.weight
        a = np.abs(pr["delta_m"][sel])
        h = np.histogram(pr["lam"][sel], bins=edges, weights=a)[0]
        out[reg] = {"delta_E_ry": 2 * w * float(pr["delta"][sel].sum()), "delta_m_ry": 2 * w * float(pr["delta_m"][sel].sum()),
                    "retained": float(pr["delta"][sel].sum() / pr["delta_m"][sel].sum()) if pr["delta_m"][sel].sum() else None,
                    "electron_fraction_lam_lt_0.5": float((n[sel] * (pr["lam"][sel] < 0.5)).sum() / n[sel].sum()),
                    "lam_edges": list(edges), "abs_delta_m_share": (h / h.sum()).tolist() if h.sum() else None,
                    "electrons": w * float(n[sel].sum())}
    return out
