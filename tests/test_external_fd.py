"""V2a for the external per-point functionals (xcsb.functional.external; node, xchole line v4-sb): the assembled potentials
v_sigma, v_tau,sigma are the derivatives of the grid energy (five-point differences along random sphere-limited directions,
both spins, polarized and unpolarized densities), the same check as for PBE, r2SCAN and SCH-SI (test_v2_potential_fd)."""
import os

import pytest

from test_v2_potential_fd import fd_spin, setup

pytestmark = pytest.mark.node
NAMES = ("r2SCAN-SB", "r2SCAN-SB-kin")


@pytest.mark.skipif(os.environ.get("XCSB_XCHOLE_LINE") != "v4-sb", reason="needs XCSB_XCHOLE_LINE=v4-sb")
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("polarized", (True, False))
def test_external_potential_fd(name, polarized):
    from xcsb.functional import get
    g, rho, tau = setup(polarized=polarized)
    fd_spin(get(name), g, rho, tau)
