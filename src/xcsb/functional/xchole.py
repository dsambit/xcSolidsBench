"""Adapter to the user's xchole package (SCH-SI model sets), read-only use (DESIGN.md Sec. 3).

xchole is the source tree of xcHoleModelMeta, branch strongxc-v3, pinned to the v3.1 install commit XCHOLE_PIN. Its `src`
directory is given by the environment variable XCSB_XCHOLE_SRC and must be a git checkout at the pin (Perlmutter) or an
export with a file XCHOLE_COMMIT next to `src` that names the pin (laptop: `git archive`, never a worktree of that repo).
The adapter puts that directory first on sys.path before the first `import xchole`, and raises if xchole was already
imported from anywhere else or if the commit or a record hash differs from the pinned values.

Public xchole API used: xchole.benchmarks.functionals.get(name) -> SCHSIFunctional; .build_model() -> ModelXC_SCHSI;
model.e_grad(w, ru, rd, gu, gd, tu, td, lu, ld, Hu, Hd) -> (sum_k w_k e_k, per-point w_k de/dx_k) and
model.e_parts(...) -> (E_x, E_c). With w = 1 the derivatives are those of e itself.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys

import numpy as np

from .contract import PointFunctional, PointInputs, PointOutputs

XCHOLE_PIN = "84af4629ae790df3cd9fe05e83a837e3e284d309"     # strongxc-v3: "v3.1 install (user GO 2026-11-09)"
RECORD_SHA256 = {                                            # installed records at the pin (src/xchole/sch_si/parameter_sets)
    "SCH-SI-A:v3.1": ("SCH-SI-A-v3.1.json", "cbd4b9c58bda0b15a6857bd463197c5e3c82da6ec2263a4232a09a9e8974280f"),
    "SCH-SI-B:v3.1": ("SCH-SI-B-v3.1.json", "5b9d3a24162b78f8314319b13085d46f2da84e5a983096bfe6c1fd217bbda277"),
}


class XcholePinError(RuntimeError):
    pass


def _source_commit(src: str) -> str:
    root = os.path.dirname(os.path.abspath(src))
    marker = os.path.join(root, "XCHOLE_COMMIT")
    if os.path.exists(marker):
        return open(marker).read().split()[0]
    try:
        out = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", root, "status", "--porcelain", "--", "src"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise XcholePinError(f"cannot determine the xchole commit of {root}: {exc}") from None
    if dirty:
        raise XcholePinError(f"xchole tree {root} has local changes under src/:\n{dirty}")
    return out


def import_pinned_xchole():
    """Import xchole from $XCSB_XCHOLE_SRC at the pin; return the module and its source commit."""
    src = os.environ.get("XCSB_XCHOLE_SRC")
    if not src:
        raise XcholePinError("XCSB_XCHOLE_SRC is not set (path to the pinned xchole src/ directory)")
    src = os.path.abspath(src)
    commit = _source_commit(src)
    if commit != XCHOLE_PIN:
        raise XcholePinError(f"xchole at {src} is commit {commit}, pinned {XCHOLE_PIN}")
    if "xchole" in sys.modules:
        f = os.path.abspath(sys.modules["xchole"].__file__)
        if not f.startswith(src + os.sep):
            raise XcholePinError(f"xchole already imported from {f}, not from the pinned tree {src}")
    else:
        sys.path.insert(0, src)
    import xchole
    if not os.path.abspath(xchole.__file__).startswith(src + os.sep):
        raise XcholePinError(f"imported xchole from {xchole.__file__}, not from {src}")
    return xchole, commit


def parse_variant(name: str):
    """'SCH-SI-B:v3.1@ups_p=0.5,rs0_pb=0' -> ('SCH-SI-B:v3.1', {'ups_p': 0.5, 'rs0_pb': 0.0}); no '@': no edits."""
    base, _, spec = name.partition("@")
    edits = {}
    for item in filter(None, spec.split(",")):
        k, _, v = item.partition("=")
        edits[k.strip()] = float(v)
    return base, edits


class XcholeFunctional(PointFunctional):
    """An installed v3.1 set, or a DIAGNOSTIC variant of it with edited theta fields (name 'SET@key=value,...').
    Variants may edit only theta fields that the cached norms do not depend on (xchole registry.HASH_EXCLUDE, e.g. the
    payback's ups_p, rs0_pb), so the record's cached norms stay valid; the model is built with xchole's public
    params.candidate_params / params.of_record / jax_si.ModelXC_SCHSI (exactly as SCHSIFunctional.build_model), which
    validate the record (ranges, per-set P0, d_hash, m_hash) and raise. A variant is never a published parameter set."""
    uses_laplacian = True
    uses_hessian = True

    def __init__(self, name: str):
        base, edits = parse_variant(name)
        if base not in RECORD_SHA256:
            raise KeyError(f"{base!r}: only the pinned v3.1 sets {sorted(RECORD_SHA256)}")
        xchole, commit = import_pinned_xchole()
        import copy
        import json
        from xchole.sch_si import params, registry
        from xchole.sch_si.jax_si import ModelXC_SCHSI
        fname, sha = RECORD_SHA256[base]
        path = os.path.join(params.PARAMS_DIR, fname)
        raw = open(path, "rb").read()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != sha:
            raise XcholePinError(f"{path}: sha256 {digest} != pinned {sha}")
        rec = json.loads(raw)
        bad = sorted(set(edits) - set(registry.HASH_EXCLUDE))
        if bad:
            raise ValueError(f"{name}: {bad} enter the cached norms (not in registry.HASH_EXCLUDE): a variant would need a norm "
                             f"re-solve; not supported here")
        missing = sorted(set(edits) - set(rec["best"]["theta"]))
        if missing:
            raise KeyError(f"{name}: {missing} are not theta fields of {fname}")
        rec = copy.deepcopy(rec)
        rec["best"]["theta"].update(edits)
        r2, si = params.of_record(rec)
        from xchole.sch_si.jax_sc import sc_dict
        xp, cp, scp = params.candidate_params(r2)
        self.name = name
        self._model = ModelXC_SCHSI(xp, cp, scp, si)
        self._S = sc_dict(scp)                     # as ModelXC_SCHSI passes them to make_energy_si (diagnostics only)
        self._si = dict(r_c=si.r_c, s_w=si.s_w, b_g=si.b_g, t_G=si.t_G, r_q_T=si.r_q_T, kappa_T=si.kappa_T,
                        kappa_G=si.kappa_G)
        self._prov = {"name": name, "kind": "xchole" if not edits else "xchole diagnostic variant", "xchole_commit": commit,
                      "record": fname, "record_sha256": digest, "theta_edits": edits,
                      "norms": "cached (edited fields are norm-independent: registry.HASH_EXCLUDE)",
                      "published_set": not edits}
        self.record = rec

    @property
    def provenance(self) -> dict:
        return dict(self._prov)

    @staticmethod
    def _args(x: PointInputs):
        return (x.rho[0], x.rho[1], x.grad[0], x.grad[1], x.tau[0], x.tau[1], x.lapl[0], x.lapl[1], x.hess[0], x.hess[1])

    def evaluate(self, x: PointInputs) -> PointOutputs:
        w = np.ones(x.n)
        exc, g = self._model.e_grad(w, *self._args(x))
        g = [np.asarray(a, dtype=np.float64) for a in g]
        return PointOutputs(float(exc), np.stack([g[0], g[1]]), np.stack([g[2], g[3]]), np.stack([g[4], g[5]]),
                            np.stack([g[6], g[7]]), np.stack([g[8], g[9]]))

    def energy_parts(self, x: PointInputs) -> tuple[float, float]:
        """(sum e_x, sum e_c) over the points (reporting only)."""
        ex, ec = self._model.e_parts(np.ones(x.n), *self._args(x))
        return float(ex), float(ec)
