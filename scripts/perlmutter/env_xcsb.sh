# source this on Perlmutter (login or compute node) before any xcSolidsBench command.
# Keeps every write inside $W; loads the Cray toolchain QE/QEpy were built with; activates the conda env.
export W=/pscratch/sd/d/dsambit/xc_functional_dev
export HOME=$W/.home XDG_CACHE_HOME=$W/.cache PIP_CACHE_DIR=$W/.cache/pip TMPDIR=$W/tmp MPLCONFIGDIR=$W/.cache/mpl
export CONDA_PKGS_DIRS=$W/.cache/conda/pkgs CONDARC=$W/.home/.condarc
export XCSB_ENV=$W/envs/xcsb
export XCSB_BUILD=$W/build/xcsb
export XCSB_QE=$XCSB_BUILD/q-e-7.6
export XCSB_QEPY_SRC=$XCSB_BUILD/qepy-eb03de77
export XCSB_XCHOLE_SRC=$W/src/xchole-84af4629/src          # strongxc-v3 at the v3.1 install commit (read-only use)
export XCSB_PSEUDO_DIR=$W/pseudo/sg15-1.2
export JAX_COMPILATION_CACHE_DIR=$W/cache/jax
export JAX_ENABLE_X64=1
mkdir -p $HOME $TMPDIR $XDG_CACHE_HOME $CONDA_PKGS_DIRS $JAX_COMPILATION_CACHE_DIR
# module and conda scripts are not 'set -eu' clean: relax the caller's shell options while loading
_xcsb_opts=$(set +o); set +eu
module load cpu >/dev/null 2>&1                 # CPU target (login nodes default to the GPU stack)
module load PrgEnv-gnu >/dev/null 2>&1
module load cray-fftw cray-libsci >/dev/null 2>&1
module load conda >/dev/null 2>&1
eval "$(conda shell.bash hook)"
if [ -d "$XCSB_ENV" ]; then conda activate "$XCSB_ENV"; fi
eval "$_xcsb_opts"; unset _xcsb_opts
