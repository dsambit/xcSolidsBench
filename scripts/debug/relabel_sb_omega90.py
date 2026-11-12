"""One-off relabel (user decision 2026-11-12: 'r2SCAN-SB' now means omega 10): the r2SCAN-SB / -kin run directories of the
r2SCAN-SB run made with the default omega 90 (jobs 61167271, 61170583) are renamed to the explicit names 'r2SCAN-SB(omega=90)' /
'r2SCAN-SB-kin(omega=90)' (run_dir spelling '(omega90)'); their claims move to claims_relabelled_omega90/. Every move is
checked against the record's functional provenance (omega 90) and logged in RELABEL_omega90.json. Records are not edited.
Usage: python scripts/debug/relabel_sb_omega90.py <run dir> [--dry]"""
import json
import os
import re
import shutil
import sys


def main(run, dry=False):
    moves = []
    for d in sorted(os.listdir(run)):
        m = re.fullmatch(r"(SB-[A-Za-z0-9.-]+__hooked__)(r2SCAN-SB(?:-kin)?)((?:__cb[0-9.]+)?)", d)
        if not m:
            continue
        p = os.path.join(run, d, "record.json")
        r = json.load(open(p)) if os.path.exists(p) else {}
        w = (r.get("functional_provenance") or {}).get("omega")
        if r and w is not None and float(w) != 90.0:
            raise RuntimeError(f"{d}: provenance omega {w}, not 90: not relabelled")
        new = f"{m.group(1)}{m.group(2)}(omega90){m.group(3)}"
        if os.path.exists(os.path.join(run, new)):
            raise RuntimeError(f"{new} exists")
        moves.append({"from": d, "to": new, "status": r.get("status"), "provenance_omega": w})
    cl = os.path.join(run, "claims")
    cmoves = [c for c in sorted(os.listdir(cl)) if re.fullmatch(r"SB-[A-Za-z0-9.-]+__r2SCAN-SB(?:-kin)?(?:__cb[0-9.]+)?", c)] \
        if os.path.isdir(cl) else []
    log = {"decision": "user 2026-11-12: r2SCAN-SB / -kin without an explicit omega mean omega 10 (external.DEFAULT_OMEGA)",
           "run_dirs": moves, "claims_moved": cmoves}
    print(json.dumps(log, indent=1))
    if dry:
        return 0
    for mv in moves:
        os.rename(os.path.join(run, mv["from"]), os.path.join(run, mv["to"]))
    if cmoves:
        os.makedirs(os.path.join(run, "claims_relabelled_omega90"), exist_ok=True)
        for c in cmoves:
            shutil.move(os.path.join(cl, c), os.path.join(run, "claims_relabelled_omega90", c))
    json.dump(log, open(os.path.join(run, "RELABEL_omega90.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], "--dry" in sys.argv))
