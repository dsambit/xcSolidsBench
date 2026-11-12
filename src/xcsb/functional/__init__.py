"""Functional registry: names -> PointFunctional instances (DESIGN.md Sec. 3)."""
from __future__ import annotations

from .contract import (ContractError, NegativeDensityError, PointFunctional, PointInputs, PointOutputs, HESS_PAIRS)
from .libxc import LIBXC_CODES, LibxcFunctional

XCHOLE_NAMES = ("SCH-SI-A:v3.1", "SCH-SI-B:v3.1")


def get(name: str) -> PointFunctional:
    """PBE, r2SCAN, SCAN, TPSS (libxc); SCH-SI-A:v3.1, SCH-SI-B:v3.1 (xchole at the pin); model:test (synthetic)."""
    if name in LIBXC_CODES:
        return LibxcFunctional(name)
    if "+E1" in name and name.split("+E1")[0] in XCHOLE_NAMES:
        from .xchole import XcholeE1
        return XcholeE1(name)
    if name.endswith("#ueg-end") and name.split("#")[0] in XCHOLE_NAMES:
        from .xchole import XcholeUEGEnd
        return XcholeUEGEnd(name)
    if name.split("@")[0] in XCHOLE_NAMES:
        from .xchole import XcholeFunctional
        return XcholeFunctional(name)
    if name.startswith("r2SCAN-SB"):
        from .external import ExternalFunctional
        return ExternalFunctional(name)
    if name == "model:test":
        from .model import TestFunctional
        return TestFunctional()
    raise KeyError(f"unknown functional {name!r}: {sorted(LIBXC_CODES) + list(XCHOLE_NAMES) + ['model:test']}")


__all__ = ["get", "PointFunctional", "PointInputs", "PointOutputs", "ContractError", "NegativeDensityError", "HESS_PAIRS",
           "LibxcFunctional", "LIBXC_CODES", "XCHOLE_NAMES"]
