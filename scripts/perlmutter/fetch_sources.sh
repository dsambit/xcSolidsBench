#!/bin/bash
# Login node, downloads only (git clones and the SG15 tarball): run once before build_env.sh.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
source "$REPO/scripts/perlmutter/env_xcsb.sh"
mkdir -p "$XCSB_BUILD" "$W/src" "$XCSB_PSEUDO_DIR"
[ -d "$XCSB_QE" ] || git clone -q --depth 1 --branch qe-7.6 https://gitlab.com/QEF/q-e.git "$XCSB_QE"
if [ ! -d "$XCSB_QEPY_SRC" ]; then
  git clone -q https://github.com/Quantum-MultiScale/QEpy.git "$XCSB_QEPY_SRC"
  git -C "$XCSB_QEPY_SRC" checkout -q eb03de776a0f898bdcf408be62caaa52d6893c59
fi
XROOT=$(dirname "$XCSB_XCHOLE_SRC")
if [ ! -d "$XROOT" ]; then
  git clone -q "$W/xcHoleModelMeta.git" "$XROOT"
  git -C "$XROOT" checkout -q 84af4629ae790df3cd9fe05e83a837e3e284d309
fi
python3 "$REPO/scripts/fetch_pseudos.py" "$XCSB_PSEUDO_DIR" "$REPO/pseudo/MANIFEST.json"
