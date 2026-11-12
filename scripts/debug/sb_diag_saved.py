"""r2SCAN-SB term diagnostics (xcsb.functional.external.sb_parts) on a converged run's SAVED fields (QE save, checked against the
record's electrons per spin). Light (numpy + libxc on one density). Needs XCSB_XCHOLE_LINE=v4-sb and its XCSB_XCHOLE_SRC.
Usage: python scripts/debug/sb_diag_saved.py <run dir> [<functional name>]"""
import json
import os
import sys

import numpy as np

from xcsb.functional.external import sb_parts
from xcsb.grid.spectral import SpectralGrid
from xcsb.host.qe_files import saved_spin_fields


def main(d, name=None):
    r = json.load(open(os.path.join(d, "record.json")))
    name = name or r["functional"]
    rho, tau, shape = saved_spin_fields(d)
    g = SpectralGrid(np.asarray(r["case"]["structure"]["lattice_bohr"]), shape, r["case"]["ecutrho"])
    ne = [g.integrate(rho[s]) for s in range(2)]
    ref = r["density_stats"]["n_electrons"]
    if max(abs(a - b) for a, b in zip(ne, ref)) > 1e-8:
        raise RuntimeError(f"saved fields: electrons {ne} vs record {ref}")
    out = dict(sb_parts(name, g, rho, tau), run=os.path.basename(d), m=r["magnetization"]["total"])
    print(json.dumps(out, default=float))


if __name__ == "__main__":
    main(*sys.argv[1:3])
