"""Energy-vs-potential FD check (V2a, tests/test_v2_potential_fd.fd_spin) of ONE functional in this process (polarized and
unpolarized band-limited densities); exits 0 on pass. One functional per process (the E1 install is process-wide).
Usage: python scripts/fd_check.py NAME"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
from test_v2_potential_fd import fd_spin, setup  # noqa: E402

from xcsb.functional import get  # noqa: E402


def main(name):
    f = get(name)
    for pol in (True, False):
        g, rho, tau = setup(polarized=pol)
        fd_spin(f, g, rho, tau)
        print(f"{name} polarized={pol}: FD pass", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
