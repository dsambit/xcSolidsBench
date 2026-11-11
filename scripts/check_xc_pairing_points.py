"""Cross-check of the Fe FSM pairing test against strongxc-v3 explore/xc_pairing/results/testpoints_xp.json (A:v3.1 base):
(1) reference.e_xc_points reproduces the stored P1/P2 outputs at omega 0, 0.5, 1; (2) xcsb's own composition (fe_magnetism._kernel
and pairing) agrees with P1: the model e_xc = P1 at omega 0, w_c := wbar_x = P1 at omega 1. Laptop, seconds.
Usage: XCSB_XCHOLE_SRC=<pinned src> python scripts/check_xc_pairing_points.py"""
import json
import os

import numpy as np

from xcsb.functional import get
from xcsb.functional.xchole import xc_pairing_reference
from xcsb.studies.fe_magnetism import _kernel, pairing


def main():
    ref = xc_pairing_reference()
    root = os.path.dirname(os.path.abspath(os.environ["XCSB_XCHOLE_SRC"]))
    d = json.load(open(os.path.join(root, "explore", "xc_pairing", "results", "testpoints_xp.json")))
    P = d["points"]
    a = lambda k: np.array([p[k] for p in P], dtype=float)                       # noqa: E731
    at = lambda k: np.array([p[k] for p in P], dtype=float).T                    # noqa: E731
    args = (a("rho_up"), a("rho_dn"), at("grad_up"), at("grad_dn"), a("tau_up"), a("tau_dn"), a("lap_up"), a("lap_dn"),
            at("hess_up"), at("hess_dn"))
    out = {"n_points": len(P)}
    worst = 0.0
    for key, stored in d["outputs"].items():
        form, om = key.split("_omega")
        o = ref.e_xc_points(*args, form=form, omega=float(om), base="SCH-SI-A:v3.1")
        dev = {k: float(np.max(np.abs(o[k] - np.asarray(v)) / np.maximum(np.abs(np.asarray(v)), 1e-300)))
               for k, v in stored.items() if k in o}
        out[f"stored_vs_reference_{key}"] = max(dev.values())
        worst = max(worst, max(dev.values()))
    fm = {k: np.asarray(v) for k, v in _kernel(get("SCH-SI-A:v3.1"))(*args).items()}
    pr = pairing(fm, args[0], args[1])
    p1_0 = ref.e_xc_points(*args, form="P1", omega=0.0, base="SCH-SI-A:v3.1")["e_xc"]
    p1_1 = ref.e_xc_points(*args, form="P1", omega=1.0, base="SCH-SI-A:v3.1")["e_xc"]
    rel = lambda u, v: float(np.max(np.abs(u - v) / np.maximum(np.abs(v), 1e-300)))   # noqa: E731
    out["xcsb_model_vs_P1_omega0"] = rel(pr["model"][0], p1_0)
    out["xcsb_model_vs_installed_total"] = rel(pr["model"][0], fm["model_total"])
    out["xcsb_wc_eq_wbarx_vs_P1_omega1"] = rel(pr["wc_eq_wx"][0], p1_1)
    out["max_rel_deviation"] = max(worst, out["xcsb_model_vs_P1_omega0"], out["xcsb_wc_eq_wbarx_vs_P1_omega1"])
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
