"""Diagnosis of the V2 mismatch for SCH-SI: is xchole's e_grad the derivative of its own energy, point by point?

On K points sampled from the V2a test density (same grid and density as tests/test_v2_potential_fd.py), each input of
each point is perturbed alone; five-point differences of E = sum_k e_k at several step sizes are compared with the
e_grad derivative. Writes a JSON table (max relative error per input and step). Node only (JAX compile).
Usage: python scripts/debug/xchole_point_fd.py <out.json> [SCH-SI-A:v3.1]
"""
import json
import sys

import numpy as np

sys.path.insert(0, "tests")
from test_v2_potential_fd import setup  # noqa: E402

from xcsb.functional import PointInputs, get  # noqa: E402
from xcsb.host.evaluate import point_inputs  # noqa: E402


def main(out, name="SCH-SI-A:v3.1", K=24, seed=0):
    f = get(name)
    g, rho, tau = setup("triclinic")
    x, _ = point_inputs(g, rho, tau, True, True)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(x.n, size=K, replace=False))
    p = PointInputs(x.rho[:, idx].copy(), x.grad[:, :, idx].copy(), x.tau[:, idx].copy(), x.lapl[:, idx].copy(),
                    x.hess[:, :, idx].copy())
    base = f.evaluate(p)
    fields = {"rho": (2,), "grad": (2, 3), "tau": (2,), "lapl": (2,), "hess": (2, 6)}
    table = {}
    for fname, shp in fields.items():
        for comp in np.ndindex(shp):
            key = f"{fname}{list(comp)}"
            table[key] = {}
            for rel_h in (1e-3, 1e-4, 1e-5, 1e-6):
                errs = []
                for k in range(K):
                    arr = getattr(p, fname)
                    h = rel_h * max(abs(arr[comp + (k,)]), 1e-3 * float(np.mean(np.abs(arr))) + 1e-12)
                    E = {}
                    for t in (-2, -1, 1, 2):
                        q = PointInputs(*[a.copy() for a in (p.rho, p.grad, p.tau, p.lapl, p.hess)])
                        getattr(q, fname)[comp + (k,)] += t * h
                        E[t] = f.evaluate(q).e_sum
                    fd = (8 * (E[1] - E[-1]) - (E[2] - E[-2])) / (12 * h)
                    an = getattr(base, "d_" + fname)[comp + (k,)]
                    errs.append(abs(fd - an) / (abs(an) + 1e-10))
                table[key][str(rel_h)] = {"max_rel": float(max(errs)), "median_rel": float(np.median(errs)),
                                          "worst_point": int(np.argmax(errs))}
            print(key, {h: f"{v['max_rel']:.2e}" for h, v in table[key].items()}, flush=True)
    worst = {k: min(v[h]["max_rel"] for h in v) for k, v in table.items()}
    # grid level: the V2a pairing error against the FD amplitude (truncation shrinks with eps, noise grows)
    from test_v2_potential_fd import d5, energy
    from xcsb.host.evaluate import evaluate_on_grid
    from xcsb.testing import sphere_limited_random
    res = evaluate_on_grid(f, g, rho, tau, chunk=4096)
    scan = {}
    for which in ("rho", "tau"):
        dlt = np.zeros((2,) + g.shape)
        dlt[0] = sphere_limited_random(g, seed=17 + (which == "tau"))
        basearr = rho if which == "rho" else tau
        pot = res.v[0] if which == "rho" else res.v_tau[0]
        an = g.integrate(pot * dlt[0])
        scale = g.integrate(np.abs(pot * dlt[0]))
        for eps in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5):
            amp = eps * float(np.mean(basearr[0]))
            E = (lambda r: energy(f, g, r, tau)) if which == "rho" else (lambda t: energy(f, g, rho, t))
            fd = d5(E, basearr, dlt, amp)
            scan[f"{which}/{eps}"] = abs(fd - an) / scale
            print("grid", which, eps, f"{scan[f'{which}/{eps}']:.3e}", flush=True)
    json.dump({"functional": name, "K": K, "table": table, "best_over_h": worst, "grid_eps_scan": scan,
               "points": {"rho": p.rho.tolist(), "tau": p.tau.tolist()}}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3] or []))
