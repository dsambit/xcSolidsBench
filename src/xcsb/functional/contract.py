"""The per-point functional contract (DESIGN.md Sec. 3).

Always spin-resolved (two channels, index 0 = up, 1 = down); a spin-unpolarized density is passed as two equal halves.
Hartree atomic units: rho_sigma in bohr^-3, tau_sigma = (1/2) sum_i f_i |grad phi_i|^2 (with the 1/2), energies in Ha.

Inputs per point: rho (2, N), grad (2, 3, N), tau (2, N), lapl (2, N), hess (2, 6, N) with the Hessian components in the
order xx, xy, xz, yy, yz, zz (each an independent input: d e / d H_xy is the derivative with respect to that single input).

Outputs: e_sum = sum_k e(r_k) (e = e_xc per volume, Ha bohr^-3) and the per-point first partial derivatives of e with
respect to every input (same shapes as the inputs).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np

HESS_PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))      # xx, xy, xz, yy, yz, zz


class ContractError(ValueError):
    """Inputs or outputs violate the contract (shape, dtype, finiteness)."""


class NegativeDensityError(ContractError):
    """A spin density is negative at some point (the host never clips; DESIGN.md Sec. 4.5)."""


def _check(name, a, shape):
    if not isinstance(a, np.ndarray) or a.dtype != np.float64:
        raise ContractError(f"{name}: float64 ndarray required, got {type(a).__name__} {getattr(a, 'dtype', None)}")
    if a.shape != shape:
        raise ContractError(f"{name}: shape {a.shape}, expected {shape}")
    if not np.all(np.isfinite(a)):
        raise ContractError(f"{name}: non-finite values ({np.count_nonzero(~np.isfinite(a))} of {a.size})")


@dataclass
class PointInputs:
    rho: np.ndarray
    grad: np.ndarray
    tau: np.ndarray
    lapl: np.ndarray
    hess: np.ndarray

    @property
    def n(self) -> int:
        return self.rho.shape[1]

    def validate(self) -> "PointInputs":
        n = self.rho.shape[-1]
        _check("rho", self.rho, (2, n))
        _check("grad", self.grad, (2, 3, n))
        _check("tau", self.tau, (2, n))
        _check("lapl", self.lapl, (2, n))
        _check("hess", self.hess, (2, 6, n))
        if np.any(self.rho < 0):
            neg = self.rho < 0
            raise NegativeDensityError(f"negative spin density at {np.count_nonzero(neg)} points (min {self.rho.min():.3e})")
        return self

    def slice(self, a: int, b: int) -> "PointInputs":
        return PointInputs(self.rho[:, a:b], self.grad[:, :, a:b], self.tau[:, a:b], self.lapl[:, a:b], self.hess[:, :, a:b])

    def padded(self, n: int) -> "PointInputs":
        """These points followed by zero-density points (all inputs 0) up to n points."""
        m = n - self.n
        if m < 0:
            raise ContractError(f"cannot pad {self.n} points to {n}")

        def pad(a):
            return np.concatenate([a, np.zeros(a.shape[:-1] + (m,))], axis=-1)

        return PointInputs(pad(self.rho), pad(self.grad), pad(self.tau), pad(self.lapl), pad(self.hess))

    @staticmethod
    def zeros(n: int) -> "PointInputs":
        z = np.zeros
        return PointInputs(z((2, n)), z((2, 3, n)), z((2, n)), z((2, n)), z((2, 6, n)))


@dataclass
class PointOutputs:
    e_sum: float
    d_rho: np.ndarray
    d_grad: np.ndarray
    d_tau: np.ndarray
    d_lapl: np.ndarray
    d_hess: np.ndarray
    extra: dict = field(default_factory=dict)

    def validate(self, n: int) -> "PointOutputs":
        if not np.isfinite(self.e_sum):
            raise ContractError(f"e_sum not finite: {self.e_sum}")
        _check("d_rho", self.d_rho, (2, n))
        _check("d_grad", self.d_grad, (2, 3, n))
        _check("d_tau", self.d_tau, (2, n))
        _check("d_lapl", self.d_lapl, (2, n))
        _check("d_hess", self.d_hess, (2, 6, n))
        return self

    def slice(self, a: int, b: int) -> "PointOutputs":
        """Derivatives of points a:b (e_sum is not sliceable: set to NaN, callers must not use it)."""
        return PointOutputs(float("nan"), self.d_rho[:, a:b], self.d_grad[:, :, a:b], self.d_tau[:, a:b],
                            self.d_lapl[:, a:b], self.d_hess[:, :, a:b])

    @staticmethod
    def concatenate(parts: list["PointOutputs"], e_sum: float) -> "PointOutputs":
        def cat(name):
            return np.concatenate([getattr(p, name) for p in parts], axis=-1)

        return PointOutputs(e_sum, cat("d_rho"), cat("d_grad"), cat("d_tau"), cat("d_lapl"), cat("d_hess"))


class PointFunctional(abc.ABC):
    """A functional of the per-spin point inputs. Implementations: libxc, xchole, the synthetic test functional."""

    name: str
    uses_laplacian: bool = False
    uses_hessian: bool = False

    @property
    def provenance(self) -> dict:
        return {"name": self.name}

    @abc.abstractmethod
    def evaluate(self, x: PointInputs) -> PointOutputs:
        """e_sum and every first partial derivative of e at each point. Must accept zero-density points (all inputs 0)
        and return finite values there with e = 0."""

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r})"
