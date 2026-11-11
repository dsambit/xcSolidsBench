# xcSolidsBench

A modular solid-state test bench for exchange-correlation functionals of (ρ_σ, ∇ρ_σ, τ_σ, ∇²ρ_σ) and optionally the density
Hessian, run self-consistently (generalized Kohn-Sham, collinear spin) in Quantum ESPRESSO 7.6 through QEpy, with
norm-conserving SG15 ONCV pseudopotentials.

Status: M0 (infrastructure + verification tiers) implemented; M1 (magnetism of bcc Fe and V, starting with the cutoff
ladder) waits for the user's go. Design: [DESIGN.md](DESIGN.md).

## Layout
- `src/xcsb/functional/`: the per-point contract; libxc (PBE, r2SCAN, SCAN, TPSS) and xchole (SCH-SI-A:v3.1, SCH-SI-B:v3.1)
  adapters; a synthetic test functional
- `src/xcsb/grid/spectral.py`: FFT derivatives with the ecutrho sphere; adjoint potential assembly
- `src/xcsb/parallel/mpi.py`: point-parallel evaluation (fixed chunks, rank-count independent)
- `src/xcsb/host/`: QE layouts and units (`layout.py`), the QEpy session (`qe.py`), the grid-level host step (`evaluate.py`)
- `src/xcsb/scf/`: the SCF driver and the global settings; `src/xcsb/tiers.py`, `src/xcsb/cli.py`
- `patches/qepy-qe76-tau-hook.patch`: the τ-potential hook (extkin) and `qepy_set_tau` for QEpy `qe76` @ eb03de77
- `scripts/perlmutter/`: environment, source fetch, build (QE 7.6 + patched QEpy in a conda env in $W), tier job
- `tests/`: V1-V2 (laptop, seconds; `pytest`), node tests (`pytest -m node`)

## Use (Perlmutter, inside an allocation)
    source scripts/perlmutter/env_xcsb.sh
    srun -n 8 -c 32 python -m xcsb.cli scf --case Fe --mode hooked --name SCH-SI-A:v3.1 --root $W/runs/xcsb/test
