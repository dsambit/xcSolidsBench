"""A closed-form synthetic test functional (verification only, DESIGN.md Sec. 3).

Deliberately non-physical, smooth, and dependent on every input (rho_sigma, grad rho_sigma, tau_sigma, lap rho_sigma and all
six Hessian components) with a spin coupling through rho_up rho_down and grad rho_up . grad rho_down. Exact derivatives by JAX
autodiff. Zero-density points give e = 0 with finite derivatives (the regularized denominators rho + RHO_REG only appear
multiplied by powers of rho that vanish faster).
"""
from __future__ import annotations

import numpy as np

from .contract import PointFunctional, PointInputs, PointOutputs

RHO_REG = 1e-10


def _energy_density(rho, grad, tau, lapl, hess):
    import jax.numpy as jnp
    e = 0.0
    for s in range(2):
        r = rho[s]
        rr = r + RHO_REG
        r53 = rr ** (5.0 / 3.0)
        x = jnp.sum(grad[s] ** 2, axis=0) / rr ** (8.0 / 3.0)
        t = tau[s] / r53
        lq = lapl[s] / r53
        h = hess[s]
        q = (h[1] + 0.5 * h[2] - 0.3 * h[4] + 0.2 * (h[0] - h[5]) + 0.1 * h[3]) / r53
        f = -1.0 + 0.2 * jnp.tanh(0.1 * x) + 0.1 * jnp.sin(0.3 * t) + 0.05 * jnp.tanh(0.2 * lq) + 0.03 * jnp.tanh(0.1 * q) \
            + 0.02 * jnp.tanh(0.05 * x * t)
        e = e + r ** (4.0 / 3.0) * f
    ru, rd = rho[0], rho[1]
    gud = jnp.sum(grad[0] * grad[1], axis=0)
    rt = ru + rd + RHO_REG
    e = e + 0.05 * ru * rd / rt ** (2.0 / 3.0) * (1.0 + 0.1 * jnp.tanh(gud / rt ** (8.0 / 3.0)))
    return e


class TestFunctional(PointFunctional):
    name = "model:test"
    uses_laplacian = True
    uses_hessian = True

    def __init__(self):
        import jax
        jax.config.update("jax_enable_x64", True)

        def total(rho, grad, tau, lapl, hess):
            return jax.numpy.sum(_energy_density(rho, grad, tau, lapl, hess))

        self._vg = jax.jit(jax.value_and_grad(total, argnums=(0, 1, 2, 3, 4)))
        self._e = jax.jit(_energy_density)

    @property
    def provenance(self) -> dict:
        return {"name": self.name, "kind": "synthetic test functional (xcsb.functional.model)"}

    def evaluate(self, x: PointInputs) -> PointOutputs:
        e_sum, (dr, dg, dt, dl, dh) = self._vg(x.rho, x.grad, x.tau, x.lapl, x.hess)
        f = lambda a: np.asarray(a, dtype=np.float64)
        return PointOutputs(float(e_sum), f(dr), f(dg), f(dt), f(dl), f(dh))

    def energy_density(self, x: PointInputs) -> np.ndarray:
        return np.asarray(self._e(x.rho, x.grad, x.tau, x.lapl, x.hess))
