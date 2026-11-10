"""V1: the functional adapters. libxc and the synthetic test functional against central finite differences of the
per-point energy density (laptop, seconds); xchole against the v3.1 handoff sanity points (node: the JAX compile)."""
import json
import os

import numpy as np
import pytest

from xcsb.functional import LIBXC_CODES, PointInputs, get

DATA = os.path.join(os.path.dirname(__file__), "data")


def random_points(n=64, seed=0, polarized=True):
    rng = np.random.default_rng(seed)
    rho = rng.uniform(0.05, 0.5, size=(2, n))
    if not polarized:
        rho[1] = rho[0]
    grad = 0.3 * rng.standard_normal((2, 3, n)) * rho[:, None, :]
    tw = np.sum(grad ** 2, axis=1) / (8 * rho)
    tau = 1.5 * tw + 0.2 * rho ** (5 / 3) * rng.uniform(0.5, 1.5, size=(2, n))
    hess = 0.4 * rng.standard_normal((2, 6, n)) * rho[:, None, :]
    lapl = hess[:, 0] + hess[:, 3] + hess[:, 5]
    return PointInputs(rho, grad, tau, lapl, hess)


def per_point_e(f, x):
    if hasattr(f, "energy_density"):
        return f.energy_density(x)
    out = f.evaluate(x)
    return out.extra["e"]


def fd_check(f, x, fields, h=1e-5, tol=2e-6):
    out = f.evaluate(x)
    for name in fields:
        arr = getattr(x, name)
        d = getattr(out, "d_" + name)
        for idx in np.ndindex(arr.shape[:-1]):
            step = h * np.maximum(np.abs(arr[idx]), 0.05)
            xp, xm = (PointInputs(*[a.copy() for a in (x.rho, x.grad, x.tau, x.lapl, x.hess)]) for _ in range(2))
            getattr(xp, name)[idx] += step
            getattr(xm, name)[idx] -= step
            fd = (per_point_e(f, xp) - per_point_e(f, xm)) / (2 * step)
            scale = np.max(np.abs(d[idx])) + np.max(np.abs(fd)) + 1e-12
            err = np.max(np.abs(fd - d[idx])) / scale
            assert err < tol, (f.name, name, idx, err)


@pytest.mark.parametrize("name", sorted(LIBXC_CODES))
@pytest.mark.parametrize("polarized", [True, False])
def test_libxc_fd(name, polarized):
    f = get(name)
    fields = ["rho", "grad"] + (["tau"] if f.is_meta else [])
    fd_check(f, random_points(seed=1, polarized=polarized), fields)


def test_libxc_e_sum_is_sum_of_e():
    f = get("r2SCAN")
    x = random_points()
    out = f.evaluate(x)
    assert abs(out.e_sum - np.sum(out.extra["e"])) < 1e-14 * abs(out.e_sum)


def test_model_fd():
    f = get("model:test")
    fd_check(f, random_points(seed=2), ["rho", "grad", "tau", "lapl", "hess"])


@pytest.mark.parametrize("name", sorted(LIBXC_CODES) + ["model:test"])
def test_zero_padding(name):
    f = get(name)
    out = f.evaluate(PointInputs.zeros(16)).validate(16)
    assert out.e_sum == 0.0


# ---- xchole: node only (the JAX compile of the SCH-SI model takes about a minute) ------------------------------------
@pytest.mark.node
@pytest.mark.parametrize("set_", ["A", "B"])
def test_xchole_sanity_points(set_):
    ref = json.load(open(os.path.join(DATA, f"xchole_sanity_v3.1_{set_}.json")))
    f = get(f"SCH-SI-{set_}:v3.1")
    pts = ref["points"]
    n = len(pts)
    x = PointInputs.zeros(n)
    for k, p in enumerate(pts):
        i = p["inputs"]
        x.rho[:, k] = i["rho_u"], i["rho_d"]
        x.grad[0, :, k], x.grad[1, :, k] = i["grad_u"], i["grad_d"]
        x.tau[:, k] = i["tau_u"], i["tau_d"]
        x.lapl[:, k] = i["lap_u"], i["lap_d"]
        x.hess[0, :, k], x.hess[1, :, k] = i["hess_u"], i["hess_d"]
    out = f.evaluate(x.padded(n + 3)).validate(n + 3)
    e_ref = sum(r["e_x"] + r["e_c"] for r in ref["values"])
    assert abs(out.e_sum - e_ref) <= 1e-10 * abs(e_ref)
    for k, r in enumerate(ref["values"]):
        de = r["de_xc"]
        for s, sfx in enumerate("ud"):
            pairs = [(out.d_rho[s, k], de["rho_" + sfx]), (out.d_tau[s, k], de["tau_" + sfx]),
                     (out.d_lapl[s, k], de["lap_" + sfx])]
            pairs += list(zip(out.d_grad[s, :, k], de["grad_" + sfx])) + list(zip(out.d_hess[s, :, k], de["hess_" + sfx]))
            for got, want in pairs:
                assert abs(got - want) <= 1e-10 * max(abs(want), 1e-8), (set_, k, sfx, got, want)
    z = f.evaluate(PointInputs.zeros(8)).validate(8)
    assert z.e_sum == 0.0
