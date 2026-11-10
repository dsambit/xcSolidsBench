#!/bin/bash
# xcSolidsBench build on Perlmutter: conda env + QE 7.6 + QEpy (qe76, eb03de77) with patches/qepy-qe76-tau-hook.patch.
# Run on a COMPUTE node (1-node interactive allocation), from a clone of $W/xcSolidsBench.git:
#     bash scripts/perlmutter/build_env.sh <phase...>     phases: env qe qepy check   (default: all four, in order)
# The sources (QE, QEpy, xchole pin, SG15) are fetched beforehand on the login node by fetch_sources.sh.
# Each phase writes a stamp and a log under $XCSB_BUILD/logs; a finished phase is skipped. Failures stop the script.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
source "$REPO/scripts/perlmutter/env_xcsb.sh"
LOGS=$XCSB_BUILD/logs; mkdir -p "$LOGS"
NP=${NP:-32}
PHASES=${*:-env qe qepy check}

stamp() { echo "$(date -Is) $(cd "$REPO" && git rev-parse HEAD)" > "$XCSB_BUILD/.done_$1"; }
done_() { [ -f "$XCSB_BUILD/.done_$1" ]; }

phase_env() {
  done_ env && { echo "env: done"; return; }
  conda create -y -p "$XCSB_ENV" python=3.12 pip
  set +u; conda activate "$XCSB_ENV"; set -u
  python -m pip install "numpy>=2.0,<2.5" scipy "meson>=1.4" ninja "f90wrap==0.2.16" setuptools wheel pytest \
      "jax==0.11.2" "pyscf==2.14.0" pandas matplotlib
  MPICC="cc -shared" python -m pip install --no-binary=mpi4py mpi4py
  python -m pip install --no-deps -e "$REPO"
  stamp env
}

phase_qe() {
  done_ qe && { echo "qe: done"; return; }
  set +u; conda activate "$XCSB_ENV"; set -u
  cd "$XCSB_QE"
  [ "$(git rev-parse HEAD)" = 9f93ddec427d2b9a45bb72d828c6d324f62fcabd ] || { echo "QE tree is not qe-7.6 (9f93ddec)"; exit 1; }
  export CFLAGS="-O2 -fPIC" FFLAGS="-O2 -fPIC" try_foxflags="-fPIC"
  export BLAS_LIBS="-L${CRAY_LIBSCI_PREFIX_DIR}/lib -lsci_gnu_mp" LAPACK_LIBS="-L${CRAY_LIBSCI_PREFIX_DIR}/lib -lsci_gnu_mp"
  export FFT_LIBS="-L${FFTW_DIR} -lfftw3_omp -lfftw3"
  ./configure MPIF90=ftn F90=ftn CC=cc --enable-openmp --with-scalapack=no
  # QEpy's own recipe drops ' -pg' lines from PIOUD's Makefile
  [ -f PIOUD/src/Makefile ] && sed -i '/ -pg/d' PIOUD/src/Makefile
  make -j"$NP" all || make -j"$NP" all
  stamp qe
}

phase_qepy() {
  done_ qepy && { echo "qepy: done"; return; }
  set +u; conda activate "$XCSB_ENV"; set -u
  cd "$XCSB_QEPY_SRC"
  [ "$(git rev-parse HEAD)" = eb03de776a0f898bdcf408be62caaa52d6893c59 ] || { echo "QEpy tree is not eb03de77"; exit 1; }
  if ! git diff --quiet; then
    git diff | cmp -s - "$REPO/patches/qepy-qe76-tau-hook.patch" || { echo "QEpy tree has foreign changes"; exit 1; }
  else
    git apply "$REPO/patches/qepy-qe76-tau-hook.patch"
  fi
  export qedir="$XCSB_QE" CFLAGS="-O2 -fPIC" FFLAGS="-O2 -fPIC" MAKEFLAGS="-j8"
  python -m pip install --no-build-isolation -v . 2>&1 | tail -200
  stamp qepy
}

phase_check() {
  set +u; conda activate "$XCSB_ENV"; set -u
  cd "$REPO"
  python - <<'PY'
import json, os, subprocess, sys, hashlib
import numpy, scipy, jax, pyscf, mpi4py, qepy
from qepy.driver import Driver
from pyscf.dft import libxc
info = {
  "python": sys.version.split()[0], "numpy": numpy.__version__, "scipy": scipy.__version__, "jax": jax.__version__,
  "pyscf": pyscf.__version__, "libxc": libxc.libxc_version() if hasattr(libxc, "libxc_version") else None,
  "mpi4py": mpi4py.__version__, "mpi": mpi4py.MPI.Get_library_version().split("\n")[0],
  "qepy_file": qepy.__file__, "qe_commit": "9f93ddec427d2b9a45bb72d828c6d324f62fcabd",
  "qepy_commit": "eb03de776a0f898bdcf408be62caaa52d6893c59",
  "patch_sha256": hashlib.sha256(open("patches/qepy-qe76-tau-hook.patch", "rb").read()).hexdigest(),
  "xcsb_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
}
from qepy import qepy_pw
assert hasattr(qepy_pw.qepy_mod, "qepy_set_extkin"), "patched QEpy: qepy_set_extkin missing"
assert hasattr(qepy_pw.qepy_mod, "qepy_set_tau"), "patched QEpy: qepy_set_tau missing"
out = os.path.join(os.environ["XCSB_BUILD"], "BUILD_INFO.json")
json.dump(info, open(out, "w"), indent=1); print(json.dumps(info, indent=1))
PY
}

for p in $PHASES; do echo "=== phase $p $(date -Is)"; ( phase_$p ) 2>&1 | tee "$LOGS/$p.log"; done
