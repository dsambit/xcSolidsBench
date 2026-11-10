"""Download the SG15 UPF files listed in pseudo/MANIFEST.json into a directory and verify their sha256 (raises on mismatch).

Usage: python fetch_pseudos.py <target_dir> <MANIFEST.json>      (login node; downloads only)
"""
import hashlib
import json
import os
import sys
import urllib.request


def main(target, manifest):
    m = json.load(open(manifest))
    os.makedirs(target, exist_ok=True)
    for el, entry in sorted(m["files"].items()):
        path = os.path.join(target, entry["file"])
        if not os.path.exists(path):
            urllib.request.urlretrieve(m["url"] + entry["file"], path)
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if digest != entry["sha256"]:
            raise RuntimeError(f"{entry['file']}: sha256 {digest} != manifest {entry['sha256']}")
        print(f"{el:3s} {entry['file']} ok")


if __name__ == "__main__":
    main(*sys.argv[1:3])
