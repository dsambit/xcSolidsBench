"""Pseudopotential checks (DESIGN.md Sec. 4.5, 6.1): norm-conserving, no nonlinear core correction, sha256 = manifest."""
from __future__ import annotations

import hashlib
import json
import os
import re

MANIFEST = os.path.join(os.path.dirname(__file__), "..", "..", "pseudo", "MANIFEST.json")


class PseudoError(RuntimeError):
    pass


def manifest(path: str | None = None) -> dict:
    p = path or os.environ.get("XCSB_PSEUDO_MANIFEST") or MANIFEST
    return json.load(open(p))


def upf_header(path: str) -> dict:
    txt = open(path, errors="replace").read(200000)
    def attr(name):
        m = re.search(name + r'\s*=\s*"\s*([^"]*?)\s*"', txt)
        return m.group(1) if m else None
    return {"pseudo_type": attr("pseudo_type"), "core_correction": attr("core_correction"),
            "z_valence": float(attr("z_valence")), "functional": attr("functional"), "relativistic": attr("relativistic")}


def check_element(element: str, pseudo_dir: str, man: dict | None = None) -> dict:
    """The manifest entry of `element` after checking its file (sha256, NC, no NLCC). Raises PseudoError."""
    man = man or manifest()
    if element not in man["files"]:
        raise PseudoError(f"{element}: not in the pseudopotential manifest ({man['family']} {man['version']})")
    entry = man["files"][element]
    path = os.path.join(pseudo_dir, entry["file"])
    if not os.path.exists(path):
        raise PseudoError(f"{path}: missing (fetch with scripts/fetch_pseudos.py)")
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if digest != entry["sha256"]:
        raise PseudoError(f"{path}: sha256 {digest} != manifest {entry['sha256']}")
    h = upf_header(path)
    if h["pseudo_type"] != "NC":
        raise PseudoError(f"{path}: pseudo_type {h['pseudo_type']}: only norm-conserving pseudopotentials are supported")
    if h["core_correction"] not in ("F", "f", "false", ".false.", "FALSE"):
        raise PseudoError(f"{path}: nonlinear core correction ({h['core_correction']}) is not supported (no core tau)")
    return dict(entry, path=path, header=h)
