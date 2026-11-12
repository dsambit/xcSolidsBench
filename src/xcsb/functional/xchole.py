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

XCHOLE_PINS = {                                              # line -> pinned xchole commit (XCSB_XCHOLE_LINE selects; default v3.1)
    # strongxc-v3, 2026-11-11 (M1 pin): the v3.1 sets of 84af4629; against the previous pin f9a275025 (FSM study)
    # src/xchole/sch_si, core, ueg_gauge, benchmarks and explore/xc_pairing differ only in two docstrings (a note moved to
    # manuscript/), the records are byte-identical (sha256 below); the new commits add xchole.ellipticity (unused here)
    "v3.1": "0e9e8ee28338d77f4876b98d2ee082031a4ab5fa",
    # strongxc-v4 dd8b5a517 (user 2026-11-11, the v4 Fe items): the same sets (src/xchole/sch_si, core, ueg_gauge, benchmarks and
    # explore/xc_pairing identical to 0e9e8ee28, records byte-identical) plus explore/spin_balance (the E1 reference function)
    "v4": "dd8b5a517b45740cbbd409fb92a187eb1cd68e43",
}
XCHOLE_LINE = os.environ.get("XCSB_XCHOLE_LINE", "v3.1")
if XCHOLE_LINE not in XCHOLE_PINS:
    raise KeyError(f"XCSB_XCHOLE_LINE {XCHOLE_LINE!r}: one of {sorted(XCHOLE_PINS)}")
XCHOLE_PIN = XCHOLE_PINS[XCHOLE_LINE]
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


def variant_cache_path(name: str, record_sha: str) -> str:
    root = os.environ.get("XCSB_NORM_CACHE") or os.path.join(os.environ.get("W", "/tmp"), "cache", "xchole_norms")
    key = hashlib.sha256(f"{XCHOLE_PIN}|{record_sha}|{name}".encode()).hexdigest()[:16]
    return os.path.join(root, f"{name.split('@')[0].replace(':', '_')}_{key}.json")


def prepare_variant(name: str) -> str:
    """Re-solve the norms of a variant with norm-dependent edits through xchole's public external.load (records without
    c1, r_q, r_q_T, kappa_T, d_hash are re-solved; m_hash removed: a modified fitted record is not the fitted model) and cache
    the record. Raises on any violation (ranges, parameter-set fields). Returns the cache path."""
    import copy
    import json
    base, edits = parse_variant(name)
    import_pinned_xchole()
    from xchole.sch_si import params
    from xchole.sch_si.external import load
    fname, sha = RECORD_SHA256[base]
    raw = open(os.path.join(params.PARAMS_DIR, fname), "rb").read()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise XcholePinError(f"{fname}: sha256 mismatch")
    rec = copy.deepcopy(json.loads(raw))
    rec["best"]["theta"].update(edits)
    for k in ("c1", "r_q", "r_q_T", "kappa_T", "d_hash", "m_hash"):
        rec["best"].pop(k, None)
    solved, source = load(rec)
    path = variant_cache_path(name, sha)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    solved = dict(solved, xcsb_variant={"name": name, "edits": edits, "norms": source.get("norms")})
    tmp = path + ".tmp"
    json.dump(solved, open(tmp, "w"), indent=1)
    os.replace(tmp, path)
    return path


class XcholeUEGEnd(PointFunctional):
    """The UEG-gauge end ALONE (diagnostic; name 'SCH-SI-B:v3.1#ueg-end'): both outer switches pinned at the UEG-gauge end
    everywhere, e = sum_s e_x^UEG(rho_s, grad rho_s, lap rho_s) + e_c^UEG(rho_up, rho_dn, grad, lap), evaluated with xchole's
    public UEG-end functions at the record's parameters exactly as the full model uses them (xchole.sch_si.ueg_end.ex_spin with
    params_P of the exchange params, ueg_end.ec with params_P of the correlation params). No Lambda' (strong-coupling) correction,
    no tau. Exact derivatives by JAX autodiff."""
    uses_laplacian = True
    uses_hessian = False

    def __init__(self, name: str):
        import jax
        base = name.split("#")[0]
        self.name = name
        full = XcholeFunctional(base)
        import_pinned_xchole()
        from xchole.core import jax_model as jm
        from xchole.sch_si import ueg_end
        Px = full._model.P
        px, pc = ueg_end.params_P(Px), ueg_end.params_P(jm._cparams(Px))

        def total(ru, rd, gu, gd, lu, ld):
            e = ueg_end.ex_spin(ru, gu, lu, px) + ueg_end.ex_spin(rd, gd, ld, px) + ueg_end.ec(ru, rd, gu, gd, lu, ld, pc)
            return jax.numpy.sum(e)

        self._vg = jax.jit(jax.value_and_grad(total, argnums=(0, 1, 2, 3, 4, 5)))
        self._prov = dict(full.provenance, name=name, kind="xchole UEG-gauge end alone (diagnostic)", published_set=False)

    @property
    def provenance(self):
        return dict(self._prov)

    def evaluate(self, x: PointInputs) -> PointOutputs:
        e, (dru, drd, dgu, dgd, dlu, dld) = self._vg(x.rho[0], x.rho[1], x.grad[0], x.grad[1], x.lapl[0], x.lapl[1])
        f = lambda a: np.asarray(a, dtype=np.float64)      # noqa: E731
        n = x.n
        return PointOutputs(float(e), np.stack([f(dru), f(drd)]), np.stack([f(dgu), f(dgd)]), np.zeros((2, n)),
                            np.stack([f(dlu), f(dld)]), np.zeros((2, 6, n)))


def parse_variant(name: str):
    """'SCH-SI-B:v3.1@ups_p=0.5,rs0_pb=0' -> ('SCH-SI-B:v3.1', {'ups_p': 0.5, 'rs0_pb': 0.0}); no '@': no edits."""
    base, _, spec = name.partition("@")
    edits = {}
    for item in filter(None, spec.split(",")):
        k, _, v = item.partition("=")
        edits[k.strip()] = float(v)
    return base, edits


def xc_pairing_reference():
    """explore/xc_pairing/reference.py of the pinned tree (the P1/P2 frozen pairing diagnostics; read-only xchole use)."""
    import_pinned_xchole()
    d = os.path.join(os.path.dirname(os.path.abspath(os.environ["XCSB_XCHOLE_SRC"])), "explore", "xc_pairing")
    if not os.path.exists(os.path.join(d, "reference.py")):
        raise XcholePinError(f"{d}/reference.py missing in the pinned tree")
    if d not in sys.path:
        sys.path.insert(0, d)
    import reference
    if not os.path.abspath(reference.__file__).startswith(d + os.sep):
        raise XcholePinError(f"imported reference from {reference.__file__}, not from {d}")
    return reference


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
        missing = sorted(set(edits) - set(rec["best"]["theta"]))
        if missing:
            raise KeyError(f"{name}: {missing} are not theta fields of {fname}")
        norm_dep = sorted(set(edits) - set(registry.HASH_EXCLUDE))
        if norm_dep:
            # edited fields that enter the cached norms: the norms are re-solved ONCE by xchole's public external.load
            # (prepare_variant, a separate serial step) and read here from the cache; never solved per rank
            path_c = variant_cache_path(name, digest)
            if not os.path.exists(path_c):
                raise FileNotFoundError(f"{name}: norm-dependent edits {norm_dep}; run 'xcsb prep-variant {name}' first "
                                        f"({path_c} missing)")
            rec = json.load(open(path_c))
            if {k: rec["best"]["theta"][k] for k in edits} != edits:
                raise ValueError(f"{path_c}: cached record does not carry the edits {edits}")
        else:
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
                      "norms": (f"re-solved by xchole external.load for {norm_dep} (cache {variant_cache_path(name, digest)})"
                                if norm_dep else "cached (edited fields are norm-independent: registry.HASH_EXCLUDE)"),
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
