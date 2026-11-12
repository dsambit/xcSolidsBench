"""Check a run's saved QE densities against its record (login/compute node; prints numbers only, no grid data written):
electrons per spin, min rho, and E_xc of the record's functional on the saved fields vs the record's e_xc_final_ry.
Usage: python scripts/debug/check_saved_fields.py <run dir> [--no-exc]"""
import json
import os
import sys

import numpy as np

from xcsb.functional import get
from xcsb.grid.spectral import SpectralGrid
from xcsb.host.evaluate import evaluate_on_grid
from xcsb.host.qe_files import saved_spin_fields


def main(d, exc=True):
    r = json.load(open(os.path.join(d, "record.json")))
    rho, tau, shape = saved_spin_fields(d)
    lat = np.asarray(r["case"]["structure"]["lattice_bohr"])
    g = SpectralGrid(lat, shape, r["case"]["ecutrho"])
    out = {"dir": os.path.basename(d), "shape": shape, "n_el_saved": [g.integrate(rho[s]) for s in range(2)],
           "n_el_record": r.get("density_stats", {}).get("n_electrons"), "min_rho_saved": [float(rho[s].min()) for s in range(2)],
           "min_rho_record": r.get("density_stats", {}).get("min_rho"), "m_saved": g.integrate(rho[0] - rho[1]),
           "m_record": r["magnetization"]["total"]}
    if exc:
        f = get(r["functional"])
        res = evaluate_on_grid(f, g, rho, tau)
        out.update(exc_saved_ry=2 * res.e_xc, exc_record_ry=r.get("e_xc_final_ry"))
        out["d_exc_ry"] = out["exc_saved_ry"] - out["exc_record_ry"]
    print(json.dumps(out, default=float))


if __name__ == "__main__":
    main(sys.argv[1], "--no-exc" not in sys.argv)
