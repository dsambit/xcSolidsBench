"""Numbers behind the laptop tiers V1-V2 (the tests assert them; this prints the worst values). Seconds on the laptop.
Usage: PYTHONPATH=src python scripts/report_v12.py [functional ...]"""
import json
import sys

import numpy as np

sys.path.insert(0, "tests")
import test_v1_spectral as t1  # noqa: E402
import test_v2_potential_fd as t2  # noqa: E402
import test_v2b_orbital_fd as t2b  # noqa: E402
from xcsb.functional import get  # noqa: E402
from xcsb.testing import BandLimitedDensity, cells, grid_for  # noqa: E402


def v1():
    out = {}
    for name, lat in cells().items():
        g = grid_for(lat, t1.ECUT)
        d = BandLimitedDensity(g, seed=3)
        grad, lapl, hess = g.derivatives(d.rho())
        rng = np.random.default_rng(7)
        f, h = rng.standard_normal(g.shape), rng.standard_normal(g.shape)
        gf, _, _ = g.derivatives(f)
        gh, _, _ = g.derivatives(h)
        adj = max(abs(np.sum(gf[i] * h) + np.sum(f * gh[i])) / np.sqrt(np.sum(gf ** 2) * np.sum(h ** 2)) for i in range(3))
        out[name] = {"grad": t1.rel(grad, d.grad()), "hess": t1.rel(hess, d.hess()), "lapl": t1.rel(lapl, d.lapl()),
                     "adjoint": float(adj), "npts": g.npts, "ngm": g.ngm}
    return out


def v2(names):
    out = {}
    for n in names:
        f = get(n)
        g, rho, tau = t2.setup("triclinic")
        rep = t2.fd_spin(f, g, rho, tau)
        out[n] = {f"{w}_spin{s}": min(e.values()) for (s, w), e in rep.items()}
        out[n]["V2b_worst"] = t2b.check(n)
    return out


if __name__ == "__main__":
    names = sys.argv[1:] or ["model:test", "PBE", "r2SCAN", "TPSS"]
    print(json.dumps({"V1": v1(), "V2": v2(names)}, indent=1, default=float))
