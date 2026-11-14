# xcSolidsBench: design (phase 0)

Status: DESIGN ONLY (2026-11-10), for the user's review. Nothing here is implemented yet; implementation (M0 infrastructure and
verification tiers, then M1 starting with the cutoff ladder) waits for the user's go.

A modular solid-state test bench for exchange-correlation functionals of the per-spin inputs
(ρ_σ, ∇ρ_σ, τ_σ, ∇²ρ_σ) and optionally the density Hessian ∇∇ρ_σ, run self-consistently (generalized Kohn-Sham, GKS) in
Quantum ESPRESSO 7.6 through QEpy, with plane waves and norm-conserving SG15 ONCV pseudopotentials (no nonlinear core
correction). Collinear spin polarization (QE nspin = 2) is first-class throughout.

**Clean room.** Everything below is derived from the physics and from the public sources of QE 7.6 (gitlab.com/QEF/q-e, tag
`qe-7.6`, commit `9f93ddec`) and QEpy (github.com/Quantum-MultiScale/QEpy, branch `qe76`, commit `eb03de77`, GPL-2). The source
facts the design relies on are cited by file in Sec. 4.1. No code from any other fork or any "scanfam" repository is read or used.

---

## 1. Roadmap

| # | milestone | content |
|---|---|---|
| M0 | infrastructure and verification | contract, adapters, spectral host, τ hook (QEpy patch), SCF driver, tiers V1-V6 (Sec. 7), Perlmutter build |
| **M1** | **magnetism of bcc Fe and bcc V** (first benchmark, user 2026-11-10) | step 1: the cutoff ladder with drift diagnosis (Sec. 6.2); step 2: k-mesh/smearing; step 3: moments. nspin = 2, experimental lattice constants, fixed lattice; spin moment per atom, E(FM) − E(NM); SCH-SI-A:v3.1, SCH-SI-B:v3.1 vs PBE, r2SCAN (libxc through the hooks) |
| M2 | lattice-constant scans | Fe and V first (E(V) with spin, at the M1 settings), then LC20 lattice constants and bulk moduli (family F1) |
| M3 | ferromagnetic metals (F2) | bcc Fe, hcp Co, fcc Ni: moments and lattice constants with spin |
| M4 | paramagnetic metals (F3) | bcc V, fcc Pd: spurious magnetism (FM vs NM starts and energies; the moment should collapse to 0) |
| M5 | antiferromagnetic monoxides (F4) | FeO, CoO, NiO, rocksalt AFM-II: local moments, E(AFM) − E(FM), GKS gaps |
| M6 | band gaps of the LC20 semiconductors and insulators | GKS gaps |
| M7 | "Sol62" and its subsets (F5) | the 64-solid set of Zhang, Reilly, Tkatchenko, Scheffler (NJP 2018); planned in detail once its supplement is fetched and read and Q11 is answered |
| M8 | cohesive energies | isolated atoms in boxes; design deferred (Q6) |

All magnetic moments and all oxide properties are validation / hold-out data (A vs B tests whether B's payback overmagnetizes):
never in an objective, never used to choose parameters. The same holds for every number this bench produces unless the user
decides otherwise.

**Out of scope:** non-collinear magnetism (nspin = 4) and spin-orbit coupling: the functionals are collinear; the host raises if
`noncolin` or `lspinorb` is set. Ultrasoft/PAW pseudopotentials and nonlinear core corrections (τ, ∇²ρ and the Hessian would need
augmentation and core terms) raise as well. Forces and stress are not provided (every structure through M7 has no free internal
coordinate at the fixed or scanned lattice; hcp Co at fixed c/a); the host raises if they are requested.

---

## 2. Architecture

```
            CLI (xcsb run / mag / eos / report)       TOML case files -> run dirs + JSON records
                         |
              bench/  (families F1-F5)                 reads refdata/ (archive), writes results
                         |
              scf/driver.py   SCF loop: hook -> QE diagonalize -> QE mix -> ...
                 |                         |
       host/qe.py (QEpy session,      host/hook.py:  layouts + units  <->  grid/spectral.py
       layouts, units, patch API)       |                                   (FFT derivatives, adjoint
                 |                      |                                    assembly, sphere projection)
       QEpy (+ tau-hook patch)       parallel/mpi.py (point partition, gather)
                 |                      |
       QE 7.6 pw library            functional/  contract | libxc | xchole | model (test functional)
```

Package `src/xcsb/`:

| module | responsibility | imports QEpy? |
|---|---|---|
| `functional/contract.py` | the per-point contract (Sec. 3): containers, shape and unit checks | no |
| `functional/libxc.py` | libxc adapter (PBE, r2SCAN, SCAN, TPSS) | no |
| `functional/xchole.py` | adapter to the user's `xchole` package (SCH-SI sets) | no |
| `functional/model.py` | closed-form synthetic test functional using every input (verification only) | no |
| `grid/spectral.py` | reciprocal lattice, ecutrho sphere mask, spectral derivatives, adjoint assembly, projection | no |
| `parallel/mpi.py` | contiguous point partition, Scatterv/Gatherv, fixed-size padded chunks | no |
| `host/qe.py` | QEpy session: input generation, nspin 1/2 layouts, Ry↔Ha, pseudopotential checks, patch API | yes |
| `host/hook.py` | one SCF-step evaluation: ρ, τ → contract → v, v_τ → extpot, extkin, extene | yes |
| `scf/driver.py`, `scf/settings.py` | SCF loop, convergence, records; the global numerical settings | yes |
| `bench/structures.py` | cells: bcc, fcc, hcp, diamond, zincblende, rocksalt, CsCl, rocksalt AFM-II; species labels for magnetic sublattices | no |
| `bench/magnetism.py`, `eos.py`, `oxides.py`, `gaps.py` | family protocols | via driver |
| `refdata.py` | read-only access to the archive worktree `refdata/` | no |
| `cli.py` | entry points | via bench |

`functional/`, `grid/`, `parallel/` and `bench/structures.py` are pure numpy (+ JAX/libxc inside adapters) and are tested on the
laptop in seconds without QE. Only `host/` imports QEpy. Repository extras: `patches/qepy-qe76-tau-hook.patch` (Sec. 4.4),
`scripts/perlmutter/` (build and job scripts), `tests/`, `cases/` (TOML case files), `pseudo/MANIFEST.json` (SG15 names and
sha256; the files themselves are never committed).

---

## 3. (a) The functional contract

One per-point interface, always spin-resolved (two channels); a spin-unpolarized density is passed as two equal halves. Hartree
atomic units: ρ_σ in bohr⁻³, τ_σ = ½ Σ_i f_iσ |∇φ_iσ|² (with the ½), energies in Ha.

```python
class PointFunctional(Protocol):
    name: str                      # e.g. "PBE", "r2SCAN", "SCH-SI-A:v3.1"
    uses_laplacian: bool           # the host computes ∇²ρ_σ only if True
    uses_hessian: bool             # the host computes the 6 Hessian components only if True
    provenance: dict               # library versions, commits, record hashes
    def evaluate(self, x: PointInputs) -> PointOutputs: ...

PointInputs:   rho  (2, N)      ρ_σ ≥ 0
               grad (2, 3, N)   ∇ρ_σ (vectors, not |∇ρ|²)
               tau  (2, N)      τ_σ
               lapl (2, N)      ∇²ρ_σ                        (zeros if unused)
               hess (2, 6, N)   ∂i∂jρ_σ, order xx, xy, xz, yy, yz, zz   (zeros if unused)
PointOutputs:  e    (N,)        e_xc per volume (Ha bohr⁻³)
               d_rho, d_grad, d_tau, d_lapl, d_hess   ∂e/∂x per point, same shapes as the inputs
```

- The six Hessian components are independent inputs: ∂e/∂H_xy is the derivative with respect to the single input H_xy.
  ∇²ρ_σ and tr H_σ are both passed (equal to rounding).
- The contract is a smooth map: no density cutoff and no clipping in the host; a negative ρ_σ raises (Sec. 4.5).
- Chunks: fixed size (`CHUNK`, a global constant: one compiled shape per process); the last chunk is padded with zero-density
  points (all inputs 0), for which an adapter must return e = 0 with finite derivatives (checked in V1).

**libxc adapter.** libxc through pyscf's bundled binding (`pyscf.dft.libxc`; pyscf is needed by `xchole` anyway; the libxc version
is recorded). Strings: PBE `GGA_X_PBE,GGA_C_PBE`; r2SCAN `MGGA_X_R2SCAN,MGGA_C_R2SCAN`; SCAN `MGGA_X_SCAN,MGGA_C_SCAN`; TPSS
`MGGA_X_TPSS,MGGA_C_TPSS`. Map: e = ρ ε_xc (libxc returns per particle); ∂e/∂∇ρ_↑ = 2 v_σ↑↑ ∇ρ_↑ + v_σ↑↓ ∇ρ_↓ (↓ likewise);
d_tau = v_τ; d_lapl = v_lapl (zero for these four); d_hess = 0. The array convention of the pinned pyscf version is pinned by a
unit test against finite differences.

**xchole adapter** (read-only use of `/Users/dsambit/Workspace/xcHoleModelMeta-strongxc-v3`, branch `strongxc-v3`, sets
SCH-SI-A:v3.1 and SCH-SI-B:v3.1; that repository is never modified). Public API used:

```python
import xchole.benchmarks.functionals as F
f = F.get("SCH-SI-A:v3.1")                          # SCHSIFunctional; the record is validated on load (raises on mismatch)
m = f.build_model()                                 # xchole.sch_si.jax_si.ModelXC_SCHSI
exc, g = m.e_grad(w, ru, rd, gu, gd, tu, td, lu, ld, Hu, Hd)   # Σ_k w_k e_k and w_k ∂e/∂x_k
# g = (v_rho_u, v_rho_d, v_grad_u (3,N), v_grad_d, v_tau_u, v_tau_d, v_lap_u, v_lap_d, v_H_u (6,N), v_H_d)
ex, ec = m.e_parts(w, ru, rd, gu, gd, tu, td, lu, ld, Hu, Hd)  # E_x, E_c split (reporting)
```

- The adapter calls `e_grad` with w = 1, so g are the per-point derivatives of e itself (the v3.1 handoff, Sec. 8); the host needs
  only the energy sum, and the per-point e (a diagnostic) is not part of the hot path.
- float64 (`jax_enable_x64` is set by xchole on import). Uses the Laplacian and the Hessian.
- Provenance per run: xchole commit, record file sha256, `d_hash`, `m_hash`.
- V1 reproduces the handoff sanity points (`handoff/SCH-SI-v3.1/sanity_values_{A,B}.json`) to 1e−10 relative.
- `xchole` imports pyscf and JAX at module level: both go into the bench environment.

**Test functional** (`functional/model.py`): a closed-form, smooth, deliberately non-physical e(ρ_σ, ∇ρ_σ, τ_σ, ∇²ρ_σ, H_σ) that
depends on every input and couples the spins, in JAX (autodiff) and checked against hand derivatives. Verification only (V1, V2).

---

## 4. (b) The QE host

### 4.1 QE / QEpy conventions (verified from source)

| fact | where (QE 7.6 / QEpy qe76) |
|---|---|
| Rydberg units: energies and potentials in Ry, e2 = 2, kinetic operator −∇² | QE throughout |
| nspin = 2: `rho%of_r`, `rho%of_g` hold (total, magnetization) | `PW/src/sum_band.f90` (`rhoz_or_updw(...,'->rhoz')`), `PW/src/scf_mod.f90` |
| nspin = 2: `rho%kin_r`, `rho%kin_g` stay per spin (up, down); only of_r/of_g are converted | `sum_band.f90`; comment in `PW/src/v_of_rho.f90` (v_xc_meta) |
| `rho%kin_r(:,σ) = Σ_k,i w_ki ∣∇ψ_ki∣²` (no ½; occupations in w, 2 for nspin = 1), i.e. kin_r = 2 τ_Ha | `sum_band.f90` (grad_psic, `get_rho_k`, `w1 = wg/omega`) |
| QE's own meta call uses τ = kin_r/e2 (Hartree, ½ convention), per spin | `v_of_rho.f90` v_xc_meta: `tau(k,is) = rho%kin_r(k,is)/e2` |
| `v%of_r(:,σ)` per spin (up, down) for nspin = 2; the Hartree potential is added to both | `v_of_rho.f90` (v_xc_meta, v_h) |
| `v%kin_r(:,σ) = ∂e_Ha/∂τ_σ,Ha` (dimensionless): `kedtaur = 0.5·e2·v3` | `v_of_rho.f90` v_xc_meta |
| H_meta ψ = −∇·(kedtau ∇ψ) (Ry), with kedtau(:, current_spin) | `PW/src/h_psi_meta.f90` |
| kedtau ← v%kin_r by Fourier interpolation (dense → smooth grid); vrs(:,σ) = vltot + v%of_r(:,σ) | `PW/src/set_vrs.f90` |
| deband = −∫ρ v − ∫kin_r v%kin_r (meta); descf likewise with ρ_in − ρ_out | `PW/src/electrons.f90` (delta_e, delta_escf); QEpy `qepy_delta_e.f90`, `qepy_electrons.f90` |
| τ is mixed with ρ (kin_g in mix_rho) and symmetrized (`sym_rho` on kin_g) | `PW/src/mix_rho.f90`, `sum_band.f90` |
| τ in sum_band, h_psi_meta, τ mixing and the τ energy terms are active only if `xclib_dft_is('meta')` | same files |
| starting τ: Thomas-Fermi from the starting ρ, per spin | `PW/src/potinit.f90` |
| dense grid: nr_i = 2·max∣n_i∣ + 1, rounded up to a good FFT order ⇒ the Nyquist index lies outside the ecutrho sphere | `FFTXlib/src/fft_types.f90` (`realspace_grid_init`, `grid_set`) |
| nspin = 2 requires an explicitly set starting_magnetization (an explicit 0.0 counts as set) | `PW/src/input.f90` (iosys) |
| QEpy `exttype` bit 4 skips QE's XC (v%of_r := 0, etxc := 0); Hartree still computed | QEpy `src/pw/qepy_v_of_rho.f90` |
| QEpy adds `embed%extpot` to v%of_r before set_vrs; nothing comparable exists for v%kin_r | QEpy `src/pw/qepy_v_of_rho_all.f90` |
| QEpy energy: etot += extene (if extene = 0 it uses ∫extpot ρ) | QEpy `src/pw/qepy_electrons.f90` |
| `qepy_get_rho` returns (up, down) for nspin = 2; `qepy_get_tau` returns raw kin_r | QEpy `src/pw/qepy_mod.f90` |
| iterative mode: each `diagonalize()` first rebuilds v from the current ρ (with extpot) and calls set_vrs, then c_bands and sum_band; `mix()` mixes (ρ, τ) and returns | QEpy `qepy_electrons.f90`, `qepy/driver.py` |

These facts are re-verified by tests (V3, V4), not trusted: e.g. the host's E_xc of a native QE PBE or TPSS ground state must equal
QE's etxc, in both spin cases.

### 4.2 Units and spin layout at the host boundary

From QE (each SCF step, gathered on the world root):

| | nspin = 1 | nspin = 2 |
|---|---|---|
| ρ_σ | ρ/2 for both | `qepy_get_rho` (up, down) |
| τ_σ (Ha) | kin_r/4 for both | kin_r(:,σ)/2 |

Back to QE (v_σ and v_τ,σ in Ha from Sec. 4.3):

| | nspin = 1 | nspin = 2 |
|---|---|---|
| extpot (Ry) | 2·½(v_↑ + v_↓) = v_↑ + v_↓ | 2 v_σ |
| extkin (dimensionless) | ½(v_τ↑ + v_τ↓) | v_τ,σ |
| extene (Ry) | 2 E_xc | 2 E_xc |

For nspin = 1 the averages are the exact chain rule for ρ_σ = ρ/2 (no assumption that the two channel derivatives agree; they do by
symmetry, and V4 checks nspin = 1 against nspin = 2 at zero magnetization). Apart from this layout table there is one code path for
both spin cases: the spectral operators, the contract call and the assembly act per spin channel.

### 4.3 Spectral derivatives and the potential

Grid: QE's dense FFT grid, N = nr1·nr2·nr3 points, cell volume Ω, weight w = Ω/N, gathered in QE's order (x fastest; the gather
order and padding are pinned by a test). G = 2π Bᵀn with signed integer n (numpy `fftfreq` order), B from QE's `alat·at`. Sphere
mask M(G) = [|G|² ≤ ecutrho] (|G|² in bohr⁻² equals the energy in Ry); the host checks Σ M = QE's ngm_g.

```
D_i = F⁻¹ M (iG_i) F;   ∇ρ_σ = Dρ_σ;   ∇²ρ_σ = F⁻¹ M (−|G|²) F ρ_σ;   H_σ,ij = F⁻¹ M (−G_iG_j) F ρ_σ;   P = F⁻¹ M F
```

The grid energy is E_grid = w Σ_k e(inputs at r_k). D_i is real and M is even in G with the Nyquist index excluded, so
D_iᵀ = −D_i and the second-order operators are symmetric. The exact derivative of E_grid with respect to the grid values, divided
by w, is, per spin channel,

```
v_σ   = P [ d_rho,σ − Σ_i D_i d_grad,σ,i + ∇²_M d_lapl,σ + Σ_{i≤j} (D_iD_j) d_hess,σ,ij ]
v_τ,σ = P [ d_tau,σ ]
```

assembled in G space (one forward FFT per derivative field, one inverse FFT per spin: 11 + 1 per spin for the full input set;
unused fields are skipped). The off-diagonal Hessian components enter once each, as inputs. QE's density lies in the sphere
(Sec. 4.5), so components of v outside it are annihilated in every matrix element on an alias-free grid; P makes v exactly the
gradient with respect to the sphere coefficients (the actual degrees of freedom) and removes aliasing noise at no loss. τ
variations lie in the same sphere, hence the same P for v_τ.

FFTs run on the world root in numpy (milliseconds to a second for these grids). The pointwise evaluation, the expensive part for
SCH-SI, is distributed (Sec. 4.6).

### 4.4 The τ-potential hook: minimal QEpy patch

QEpy can replace QE's XC energy and multiplicative potential (exttype 4, extpot, extene) but cannot set `v%kin_r`. The patch
(`patches/qepy-qe76-tau-hook.patch`, against QEpy `qe76` at `eb03de77`) adds one array mirroring extpot:

1. `src/pw/qepy_common.f90`: `real(dp), allocatable :: extkin(:,:)` in `embed_base`; `allocate_extkin` (nnr × nspin, as
   `allocate_extpot`); deallocation in `free_embed`.
2. `src/pw/qepy_v_of_rho.f90`: in the XC-off branch (exttype bit 4) also `IF (xclib_dft_is('meta')) v%kin_r(:,:) = 0` (today
   v%kin_r keeps stale values there, e.g. from potinit's native evaluation).
3. `src/pw/qepy_v_of_rho_all.f90`: after the extpot line, `IF (ALLOCATED(embed%extkin)) v%kin_r = v%kin_r + embed%extkin`, with
   `errore` if extkin is allocated in a non-meta run.
4. `src/pw/qepy_mod.f90`: `qepy_set_extkin(vin, gather)`, a copy of `qepy_set_extpot` (scatter from the root, nspin columns: per
   spin for nspin = 2).
5. No change to `qepy/driver.py`: the host calls `qepy_mod.qepy_set_extkin` directly, after the same group broadcast that
   `Driver.set_external_potential` performs (implementation 2026-11-10: the patch is Fortran-only).
6. For the ripple-seeded restart of M1 step 1 (Sec. 6.2): `qepy_set_tau(tau, gather)` in `qepy_mod.f90`, a mirror of the existing
   `qepy_set_rho` for `rho%kin_r` (per spin, then `rho_r2g` into `rho%kin_g`).

Everything downstream exists in QE: set_vrs interpolates v%kin_r into kedtau, h_psi_meta applies −∇·(kedtau∇ψ) per spin, delta_e and
delta_escf contain the −∫kin_r v%kin_r terms, mix_rho mixes τ. The patch is CPU-only (OpenACC clauses untouched; GPU builds are
out of scope for now).

**The meta carrier.** QE computes τ, applies h_psi_meta and mixes τ only for a meta input_dft. Every hooked run therefore sets
`input_dft = 'TPSS'` (QE-internal, no libxc build needed) with exttype = 4, so QE's own XC is never used in the SCF. QE's TPSS is
evaluated once, in potinit, for the starting potential that builds the starting orbitals (atomic + random); this start is the same
for every functional and is replaced by the hooked potential before the first diagonalization. A τ-free functional (PBE) runs
through the same path with v_τ = 0.

### 4.5 Densities: positivity and the grid

- **ecutrho = d·ecutwfc, d ≥ 4.** |ψ|² and Σ|∇ψ|² (support within 2√ecutwfc) then lie in the ecutrho sphere, the FFT grid is
  alias-free for them, and ρ_out,σ(r_k) ≥ 0 and τ_out,σ ≥ τ_W,σ hold exactly at the grid points (symmetrization averages
  nonnegative band-limited functions; convexity of |∇ρ|²/ρ keeps τ ≥ τ_W). d is the integration-grid knob: d = 4 is the protocol;
  d = 8 (the check of M1 step 1) refines only the quadrature of the nonlinear e_xc (QE's doublegrid Fourier-interpolates ρ, τ to the dense grid and v,
  v_τ back): the aliasing check that the v3.1 handoff asks for.
- **Mixed densities** (Broyden) are not guaranteed nonnegative. The host raises on any ρ_σ < 0 (count, minimum, integrated negative
  charge, iteration): no clipping, no floor. Not expected in bulk solids; if it occurs the remedy is the user's decision (Q6).
  τ_σ < τ_W,σ in mixed densities is allowed (libxc and SCH-SI are defined there) and reported.
- **No NLCC**: the host reads each UPF header and raises if `core_correction` is true; so rho_core = tau_core = 0.

### 4.6 MPI work sharing

- QE runs with its own parallelization (k-point pools `-nk`, R&G inside pools). After sum_band every pool holds the full density;
  the host gathers ρ and τ on the world root, runs the FFTs there, scatters the per-point inputs over all world ranks in contiguous
  blocks (mpi4py Scatterv), each rank evaluates its block in fixed-size chunks, the root gathers the outputs (Gatherv), sums E in
  grid order, assembles v and v_τ and hands them to QEpy (root → pools; QEpy scatters within each pool).
- Determinism: per-point results do not depend on the partition (fixed chunk shape, no cross-point reductions); the energy sum is
  formed once on the root in grid order. The host step is bitwise independent of the rank count (V5).
- Threads: QE ranks × OpenMP threads fill the node; XLA intra-op threads are limited to the rank's OpenMP count; a persistent JAX
  compilation cache on $W avoids the ~1 min compile per process and job.
- Room for later: e and the derivative fields have the crystal symmetry, so only the irreducible grid points need evaluation (up
  to 48× in cubic cells). Not before M3.

---

## 5. (c) SCF driver and CLI

One SCF per (structure, magnetic order, functional, settings); QEpy iterative mode; cold start:

```
start: the carrier (QE's native TPSS, exttype 0) is converged first in the same QEpy session and loop, from QE's cold
       start (atomic ρ, starting magnetization per species); the hook then continues from that density; the same for
       every functional. Found in M0 (2026-11-10): the atomic superposition is not positive (SG15 Fe: 0.85 e of negative
       charge); the bare starting orbitals' density is far from self-consistency (Fe: dr2 ~ 7e2 Ry, then negative mixed
       densities); a save/restart through QE's files changed the Fe density (0.054 e negative after reading back).
loop i = 1..MAXITER:
    hook:        ρ_σ, τ_σ <- QE;  e, d <- functional;  v_σ, v_τ,σ <- spectral;  set extpot, extkin, extene (exttype 4)
    diagonalize: QE rebuilds v (Hartree + extpot; v%kin_r <- extkin), c_bands, sum_band -> ρ_out, τ_out
    mix:         QE Broyden on (ρ, τ) -> ρ_in
    converged if QE's conv_elec (estimated scf accuracy < conv_thr); |ΔE_tot| is recorded, not a criterion (QEpy's
    iterative energy lags the density by one step: found in M0)
end: final hook on the converged density; record
failure (MAXITER, negative ρ_σ, NaN, any QE error): raise; the record is written with status "failed" and the history
```

Second start, `Case.start = "pbe-scaffold"` (2026-11-13; `settings.STARTS`; nspin 1 only; used by `xcsb.studies.ionic_lc`, every
earlier study keeps `"tpss-carrier"`, recorded in `case.start` and `carrier.kind`): QE's own XC is never applied.
(0) one diagonalization with XC off (extpot = extkin = 0, exttype 4: kinetic, ionic and Hartree terms of QE's starting
density), mixed with coefficient 1, so the density handed on is that step's output density (Σ f|ψ|² ≥ 0, τ ≥ τ_W; the atomic
superposition itself is not positive, M0); (1) `settings.SCAFFOLD_FUNCTIONAL` (libxc PBE, v_τ = 0) through the hook loop to
QE's conv_elec at the case's β; (2) the hook loop of the functional. Reason: Sec. 6.12 (the native carrier stalled for LiF).
Not yet run on a node when written: the diagnostic job of Sec. 6.12 is its first test.

Mixer reset at the phase change (2026-11-13, after the ionic scan 61248332): QE's Broyden buffer (last residual and input density, `MIXING_NDIM` secant pairs) and its iteration counter survive the start phase's convergence. The first step of the functional then forms a secant pair with Δ(residual) = the whole new residual and Δ(input) ≈ 0, and Broyden's update with it cancels the step: in every hooked run so far (either start) the density stayed frozen for `MIXING_NDIM` + 1 = 9 iterations, the pairs stored meanwhile were normalized noise, and the step after them was uncontrolled (7 of 68 A/B ionic points ended with a negative mixed density). With `"pbe-scaffold"` the driver calls `driver.end_scf()` between the scaffold and the functional (mix buffer deleted, iteration counter 0; density and orbitals stay in memory; `carrier.mixer_reset` in the record). `"pbe-scaffold-stale"` keeps the old behaviour, for the before/after test only (`job_ionic_lc_mix.sh`, `xcsb lc mix-summary`). OPEN: `"tpss-carrier"` (all Fe / M1 / gap studies) is unchanged and has the same nine frozen iterations; the ripple-seeded restarts also keep the history.

- SCF threshold (user 2026-11-10), two named constants in `scf/settings.py`, the class recorded in every result:
  CONV_THR_BENCH = 1e-8 Ry per cell for every benchmark SCF (all systems and functionals, M1 onward) and
  CONV_THR_VERIFY = 1e-10 Ry per cell for the verification tiers V3-V6 (labelled "verification").
- Global settings, the same for every system and functional (`scf/settings.py`, recorded in every result):
  MAXITER, mixing (Broyden 'plain'; beta and ndim fixed by the M1 convergence study), d = 4, Gaussian smearing for every system
  (one rule: for gapped systems the smearing is inert once degauss ≪ gap, and it handles the cases where a functional closes a
  gap, e.g. Ge or the monoxides). Occupations must lie in [0, 1] so that ρ_σ = Σ f|ψ|² ≥ 0: cold (Marzari-Vanderbilt) and
  Methfessel-Paxton smearing have negative occupations and gave ρ_σ < 0 in SG15 Fe (found in M0, 2026-11-10); the free
  energy's O(σ²) error is handled by the degauss convergence (and QE's σ → 0 estimate is recorded). Per-system physical settings (ecutwfc, k-mesh, degauss) come from the convergence
  protocols (Sec. 6.0) and live in the case files; once fixed they are the same for all functionals.
- Energy: QE's etot with extene = 2E_xc (Ry) in place of etxc; V3 proves this bookkeeping (incl. descf and the τ terms) by
  reproducing QE-native energies, at nspin = 1 and 2.
- Record (JSON, one per SCF): inputs (structure, species labels, starting magnetizations, pseudo sha256, cutoffs, k-mesh, smearing,
  nspin, symmetry use), functional and code provenance (xcSolidsBench commit, QEpy commit + patch sha256, QE version, libxc, JAX,
  xchole), results (E_tot and components, E_xc and the E_x/E_c split where available, Fermi level, total and absolute magnetization,
  site moments, band edges), history (E, dr², min ρ_σ, fraction of points with τ_σ < τ_W,σ), timings. No grid-level data.
- Robustness checks use cold starts (workspace rule); warm starts only as a labelled efficiency option later.
- CLI: `xcsb run case.toml`; `xcsb mag case.toml` (F2-F4 protocols); `xcsb eos case.toml` (F1, M2); `xcsb report <run dir>`.
  Run dirs: `$W/runs/xcsb/<date>_<topic>`; only JSON/CSV come back to the laptop.

---

**Warm starts (user 2026-11-11, per point).** A production point that fails from the cold start may, with the user's go for
that point, be rerun from a converged neighbour: that run's QE outdir (wavefunctions, ρ, τ) is copied and read by QE
(startingwfc and startingpot 'file') and the hook starts directly from it (no carrier phase). The record carries the label
(e.g. "warm start from m = 2.6"), the source run, and the earlier failed attempts (`record_attempt<n>.json`, summarized with
their dr2 history); every table shows the label. conv_thr is never relaxed. Robustness tests stay cold. First use: Fe FSM
m = 2.8, SCH-SI-B:v3.1 (cold: 200 hook iterations, last dr2 2.08·10⁻⁸ Ry against 10⁻⁸; job 61150345).

## 6. (d) Benchmark layer

### 6.0 Convergence protocol (all families)

**Why the cutoff needs care (the user's own documented result).** SCH-SI has no energy minimizer in a complete basis:
xcHoleModelMeta `strongxc-v3`, `productionModelMathematicalFormulation.tex`, paragraph "Basis regularization (user 2026-10-13)"
and the exact-conditions row "variational stability" (status: no); proof in the archive note `refdata/notes/brLaplacianInstability`
(Theorem 1: a semilocal energy density that depends on ∇²ρ, or on the Hessian, is lower semicontinuous along ripple sequences only
if it is convex in it, and a convex function bounded above is constant; so no semilocal Laplacian-dependent hole model is ripple
stable). Mechanism (the note's Lemma 1): multiplying every occupied orbital of a spin channel by a common factor
g = 1 + (A/k²) h(k n·r) gives ρ' = g²ρ and τ' − τ_W' = g²(τ − τ_W) exactly; as k → ∞, δρ = O(A/k²), δ∇ρ = O(A/k), δτ = O(A/k),
while ∇²ρ moves by an O(A) amount (the Hessian by A W ρ ψ nnᵀ), at a kinetic cost O(A/k², A²/k²). Since τ − τ_W is invariant,
ellipticity (1 + ∂e/∂τ_σ > 0, which v3 makes hold on real densities) does not stabilize it. Where e is concave in ∇²ρ (BR: reduced
curvature z < −0.861), averaging over the ripple lowers the energy. Gaussian bases cannot represent the ripple and regularize the
model (stable in cc-pV6Z; tight even-tempered sets drift); plane waves approach a complete basis as ecutwfc grows, so the
plane-wave limit may drift. Hence M1 starts with a cutoff ladder (Sec. 6.2, step 1), and every family repeats it.

Per system, with all functionals of the lane:

- ecutwfc ladder up to well beyond the PBE-converged value (×2-3), looking for a drift that does not level off (Sec. 6.2 step 1);
- d = ecutrho/ecutwfc: 4 (protocol) and a check at 8 (quadrature / aliasing);
- k-mesh (Monkhorst-Pack, Γ-centred) and degauss converged together at the chosen cutoff; reported quantities must change by less
  than the family's criterion between the last two rungs;
- if SCH-SI drifts with ecutwfc, the bench reports it, stops the SCH-SI lanes and brings it to the user; it never picks a cutoff to
  hide it.

| family | criteria between the last two rungs |
|---|---|
| F1 lattice constants | a₀ 0.001 Å, B₀ 0.5 GPa, E(V) curvature 1 % |
| F2/F3 metals (moments) | moment 0.005 μB/atom, E(FM) − E(NM) 1 meV/atom |
| F4 monoxides | site moment 0.01 μB, E(AFM) − E(FM) 1 meV/f.u., GKS gap 0.02 eV |

### 6.1 SG15 availability (checked 2026-11-10)

All needed elements exist in SG15 version 1.2 (`<El>_ONCV_PBE-1.2.upf`, upload 2020-02-06 set; scalar-relativistic, generated
with PBE, `core_correction="F"` read from each header): valence charges

| element | Z_val | | element | Z_val | | element | Z_val |
|---|---|---|---|---|---|---|---|
| Li | 3 | | Al | 11 | | V | 13 |
| Na | 9 | | Ga | 13 | | Fe | 16 |
| K | 9 | | C | 4 | | Co | 17 |
| Mg | 10 | | Si | 4 | | Ni | 18 |
| Ca | 10 | | Ge | 14 | | Cu | 19 |
| Sr | 10 | | As | 5 | | Rh | 17 |
| Ba | 10 | | O | 6 | | Pd | 18 |
| F | 7 | | Cl | 7 | | Ag | 19 |

(Ca, Sr, Ba listed in case the LC20 definition includes them.) The transition metals carry ns np semicore states (high ecutwfc
expected). Every functional uses these PBE-generated pseudopotentials (no consistent pseudopotentials exist for SCH-SI); this is
stated with every table. The files go to `$W/pseudo/sg15-1.2/` with sha256 in `pseudo/MANIFEST.json` (Q3).

### 6.2 M1: magnetism of bcc Fe and bcc V (first benchmark)

- bcc, one atom per primitive cell, experimental lattice constants (sourced, Sec. 8; Q10), fixed lattice.
- Functionals: SCH-SI-A:v3.1, SCH-SI-B:v3.1, PBE, r2SCAN (libxc through the hooks).

**Step 1: the cutoff ladder (user decision 2026-11-10; comes first).**

- Systems: bcc Fe (FM start) and bcc V (NM start: starting_magnetization = 0.0; FM start), nspin = 2, fixed experimental lattice.
  k-mesh and Gaussian smearing held at a generous fixed setting during the ladder (e.g. 20³, degauss 0.01 Ry; converged in step 2).
- Functionals: PBE, r2SCAN, SCH-SI-A:v3.1, SCH-SI-B:v3.1; every SCF from a cold start.
- Ladder: first PBE alone finds its conventional converged cutoff E_c^PBE (the F2/F3 criteria of the table above); then all four
  functionals on ecutwfc/E_c^PBE ∈ {0.6, 0.8, 1, 1.25, 1.5, 2, 2.5, 3}, at d = 4, plus d = 8 at ecutwfc/E_c^PBE = 1, 2 and 3.
- Reported per rung and functional: E_tot per atom, the spin moment per atom (and absolute magnetization), for V E(FM start) −
  E(NM start), and the SCF iteration count (plus the history of Sec. 5).
- Drift diagnosis:
  - (a) Shape: successive differences ΔE_n = E(rung n) − E(rung n−1). PBE and r2SCAN set the expected pattern (monotone,
    shrinking differences, plateau). For SCH-SI: is the decrease monotone, do the differences shrink or grow with the cutoff, and
    does the excess over the PBE pattern scale with ecutwfc (e.g. with the ripple wavenumber the basis can represent, √ecutwfc)?
    Tables only, linear-axis plots.
  - (b) BR-concave population: per rung, the fraction of electrons with reduced curvature z_σ < z* = −0.861,
    z_σ = 3Q_σ/(2π^{2/3}ρ_σ^{5/3}), Q_σ = ⅙[∇²ρ_σ − 4λ(τ_σ − τ_W,σ)] (the SCH-SI curvature, λ from the record; formulation
    Q_σ of the exchange iso end), i.e. Σ_σ ∫ρ_σ [z_σ < z*] / N on the grid. A diagnostic of the converged density, not part of
    the model or the potential. Reported for every functional's density (the PBE value is the reference).
  - (c) Ripple-seeded restart at the highest rung: take the converged state, apply the common-factor ripple of the note's Lemma 1
    at the density level, ρ_σ' = g²ρ_σ and τ_σ' = g²τ_σ + ½g∇g·∇ρ_σ + ½ρ_σ|∇g|² (exact for a common orbital factor), with
    g = 1 + (A/k²) Σ_{k ∈ star} cos(k·r): a symmetric star of reciprocal-lattice vectors, so QE's symmetrization keeps it, |k| near
    the largest wavenumber the orbitals represent, both fields projected onto the ecutrho sphere (P); several amplitudes A. Then
    re-converge with the hook (the converged orbitals as the starting wavefunctions). Reported: E before and after, the amplitude of
    the seeded Fourier components of ρ_σ and of ∇²ρ_σ before and after (decay = locally stable; persistence or growth with a lower
    energy = the SCF finds a rippled state). Needs one more QEpy setter, `qepy_set_tau` (Sec. 4.4, item 6).
- Outcome rule:
  - plateau (SCH-SI follows the PBE pattern; the last two rungs within the criteria; (c) relaxes back to the unrippled state with
    no lower energy): converged cutoffs as usual, the M1 steps 2-3 follow;
  - drift (no plateau, or (c) finds a lower rippled state): report it, stop the SCH-SI lanes (PBE and r2SCAN continue), and bring
    it to the user. Options for the user: define the model at a stated per-element cutoff, or prioritize the nonlocal ∇²ln ρ
    smoothing.

**Step 1 as staged (2026-11-11; `xcsb.studies.m1_ladder`, `scripts/perlmutter/job_m1_ladder.sh`, not yet run).**

- Lattice (Q10 decided, user 2026-11-11: static lattice, zero-point corrected, for every solids calculation): a_static =
  a_measured − Δ_ZP = 2.858 Å (Fe), 3.023 Å (V); archive entry `solids/lattice_constants_static_2026-11-11`. a_measured = 2.861,
  3.028 Å (entry `solids/lattice_constants_exp_2026-11-11`: Zhang et al. NJP 2018, Supplementary Material Table II, 'Exp.' = with
  zero-point; the measured values there cite Haas, Tran, Blaha PRB 79, 085104 (2009), not read). Δ_ZP = a_PBE(Corr.) −
  a_PBE(Uncorr.) = 0.003 / 0.005 Å, the same table: per its caption 'Corr.' includes the zero-point energy (phonons at a set of
  volumes) and 'Uncorr.' does not, so Δ_ZP > 0 is the zero-point expansion and the static lattice is smaller.
- Cases `M1-<El>-<state>-<ecutwfc>-d<dual>`: state FM (starting_magnetization 0.5), NM (0.0, explicit), NM1 (nspin 1, the
  E(FM) − E(NM) reference of Fe in the PBE scan); 20³ Γ-centred k (256 irreducible points), Gaussian 0.01 Ry, conv_thr 1e−8.
- E_c^PBE per element: PBE on ecutwfc ∈ {40, 50, ..., 120, 140} Ry (Fe FM, Fe NM1, V NM, V FM); E_c^PBE = the smallest scan
  cutoff from which EVERY successive pair is within the F2/F3 criteria (moments of both states 0.005 μB/atom, E(FM) − E(NM)
  1 meV/atom). Written once to `EC.json`; a complete scan that never meets them raises (the ladder is then not built).
- Ladder: r ∈ {0.6, 0.8, 1, 1.25, 1.5, 2, 2.5, 3} × E_c^PBE at d = 4 and r ∈ {1, 2, 3} at d = 8, PBE, r2SCAN, SCH-SI-A:v3.1,
  SCH-SI-B:v3.1, states Fe FM, V NM, V FM; every SCF from the standard cold start (carrier TPSS, Sec. 5).
- (b) on every converged density: λ = 0.35 (theta.lam of both v3.1 records, read from the pinned, sha256-checked records).
- (c) at r = 3, d = 4, Fe FM and V NM, all four functionals. Seed g = 1 + c h, h = Σ_star cos(G·r), every seed with the same
  PEAK density modulation max |g − 1| = c N_star = ε ∈ {10⁻³, 10⁻², 5·10⁻²} (user 2026-11-11). At one star the curvature form
  c = A₂/k² and the gradient form c = A₁/k then give the SAME g, so a second seed at the same star would be redundant. What
  separates the channels is the wavenumber: by Lemma 1 the common-factor seed changes ∇ρ_σ by O(εk) and ∇²ρ_σ by O(εk²), so the
  second-order energy of a seed of peak ε is ε²(a_c k⁴ + a_g k² + O(k⁰)) (the k³ cross term averages out: ∇h and ∇²h are in
  quadrature), a_c from the curvature channel and a_g from the gradient channel (the von Weizsäcker term included). The second
  seed is therefore the same construction at a SMALLER wavenumber band: two bands, star = the largest full shell with |G| ≤
  κ√ecutwfc, κ = 0.8 ('high') and 0.4 ('low'); e.g. Fe at 240 Ry: k = 12.20 bohr⁻¹ (144 vectors) and 5.93 bohr⁻¹ (72).
  The ladder then sees ripples at two scales; the low band weights the gradient channel four times more relative to the
  curvature channel, and the two bands at equal ε separate a_c and a_g. Shells are checked invariant under the cubic group.
  ρ_σ' = g²ρ_σ, τ_σ' = g²τ_σ + ½g∇g·∇ρ_σ + ½ρ_σ|∇g|²
  (exact for a common orbital factor; then τ_σ' − τ_W,σ' = g²(τ_σ − τ_W,σ)), both × N_σ/N_σ' (electron count per spin kept)
  and projected onto the ecutrho sphere. QE's ρ and τ are overwritten (`set_density`, `qepy_set_tau`, with a read-back check);
  QE copies ρ into the mixer's input at the next diagonalization, so the Broyden history pairs the seeded input with its own
  output; orbitals: QE's converged ones in memory. Seeds run in order (ε ascending), each from the converged base fields.
  Reported per seed: E before/after, ΔE_xc of the seeded fields, moments, iterations and history, and for ρ_σ, ∇²ρ_σ, |∇ρ_σ|
  the star-mean Fourier amplitudes (base, seeded, final) and ratio = Re⟨δ_final, δ_seed⟩/⟨δ_seed, δ_seed⟩ (δ = change
  against the base at the star; 0 decayed, 1 persisted). The rung's own SCF carries the high-band seeds; a second SCF of the
  same case (run-directory tag `riplow`) the low-band seeds (each step: base SCF + 3 seeds, so that it fits 4 h).
- Parallel layout: QE k-point pools (`-nk`, set through QEpy's `qepy_sys.set_command_line`), one rank per pool, 4 OpenMP
  threads; a quarter node (8 ranks) per SCF, a full node (32 ranks) for the r = 3 rungs. Pools are new in M1: phase 0 of the
  first allocation checks npool 8 against npool 1 (PBE and SCH-SI-B, Fe 40 Ry 8³ k; |ΔE| ≤ 10⁻⁶ Ry, |Δ|m|| ≤ 10⁻³ μB: |m| because a run may land in the mirror ferromagnet, and 10⁻³ = 1/5 of the moment criterion since m is first order in the SCF residual; revised 2026-11-11 after job 61150346, whose phase 0 failed only on these two points: m = −m, and SCH-SI-B d|m| 1.75·10⁻⁴ μB with ΔE 3·10⁻⁸ Ry) and that
  the seeded restart runs (PBE, all six seeds, both bands); a failure stops the job.
- Stages (user 2026-11-11): the job and the launcher take a stage selector, `P` or `L`, and the runner never starts a task of
  the other stage. The FIRST allocation runs `P` only: phase 0, the PBE scan (→ `EC.json`) and a full-node probe (the 80 Ry
  Fe FM scan case again on 32 ranks, tag `w4`: the measured full-node speedup); then it stops and writes
  `stage_p_report.{json,md}`: EC.json, the measured per-SCF times with pools, the cost model refitted from them (QE s per
  iteration, PBE iteration counts, full-node speedup, the SCH-SI hook time from phase 0) and the revised node-hours of stage L
  with the seeds. The user then decides batch or interactive for stage L (which needs phase 0 PASS and EC.json).
- Campaign runner (`xcsb.campaign`): tasks in order (stage P; then, once E_c is known, the d = 4 rungs ascending, the r = 3
  rungs with the seeds, the d = 8 points), as srun steps in quarter-node slots; a task starts only if 1.25 × its estimate fits
  the remaining allocation time; every step ends at the deadline; claims (mkdir) let two allocations share one run directory;
  a killed step is rerun later, a failed one is not. Tables: `xcsb m1 table --run DIR` (`ladder_table.json`, text tables).
- Cost (estimate, before calibration; scheduling only): QE 3.7 s per SCF iteration at 80 Ry, 8³ k, 8 ranks, npool 1 (Fe80
  records) scaled to 20³ k (× 256/29) and (ecutwfc/80)^1.5; SCH-SI hook 5.3 s per round of 16384-point chunks; 20 carrier,
  15 (PBE, r2SCAN) or 25 (SCH-SI) hook iterations, 20 per seed; full node 3.5 × a quarter. Per SCF (quarter node, E_c = 80 Ry,
  SCH-SI): 0.2 h at r = 0.6, 0.45 h at 1, 1.2 h at 2, 1.65 h at 2.5; r = 3 on a full node 0.64 h, + 3 seeds ≈ 1.65 h. Stage P
  ≈ 3 node-h; the whole step ≈ 36 / 48 / 66 node-h for E_c = 60 / 80 / 100 Ry (top-rung seeds ≈ 40 %), i.e. 10-18 allocations of
  4 h. The runner refits the QE constant from the stage-P records (with pools) before stage L.

**Step 2:** k-mesh and degauss converged together at the chosen cutoff (Sec. 6.0 criteria), with PBE and confirmed with SCH-SI.

**Step 3: the moments** at the chosen cutoff and k-mesh:

- Runs per functional and metal: (1) NM: nspin = 1 (exactly nonmagnetic); (2) FM start: nspin = 2 with a large starting moment,
  converged freely; (3) V: also an NM start in nspin = 2 (starting_magnetization = 0.0 set explicitly; the random part of the
  starting orbitals breaks the up/down symmetry slightly, so a spurious instability can still develop).
- Reported: spin moment per atom (the cell's total magnetization, no sphere radius), absolute magnetization, E(FM) − E(NM) per
  atom (meV), and for V the energies of every start relative to nspin = 1. The V test: does a functional spuriously magnetize it?
- Experiment: measured saturation moments are total (spin + orbital); the comparison is to the spin moment derived with the
  measured g′ factor, the total reported alongside (Q4).
- Next (M2 start): E(V) scans of Fe (FM) and V at the M1 settings, with the moment along the scan.

### 6.3 F1 (M2): LC20 lattice constants and bulk moduli

- The 20 solids of the LC20 definition (source, Sec. 8), primitive cells, nonmagnetic (nspin = 1), fixed ecutwfc across volumes.
- E(V): 11 volumes V/V_ref ∈ [0.88, 1.12] (about ±4 % in a), V_ref from the zero-point-corrected experimental a₀; third-order
  Birch-Murnaghan fit (one EOS form) → a₀, B₀, B₀′, E₀. The fit raises if V₀ lies outside the inner two thirds of the scanned range
  or if the rms residual exceeds a global threshold; a rescan centred at V₀ is then an explicit, recorded step decided by the user.
- Dispersion: raw and + PBE-D4 (dftd4 4.3.0, PBE parameters, ATM s9 = 1, as in the v3.1 handoff; geometry-only periodic call),
  both reported (Q8).
- Reference data: the LC20 definition; zero-point-corrected experimental a₀ and B₀ (Sec. 8).
- Reported: a₀, B₀, B₀′ per solid and functional; ME, MAE, MARE over LC20, raw and + D4.

### 6.4 F2 (M3): ferromagnetic metals bcc Fe, hcp Co, fcc Ni

- bcc Fe (1 atom), hcp Co (2 atoms, experimental c/a fixed), fcc Ni (1 atom); nspin = 2, FM start.
- (i) moments at the experimental lattice constants (the M1 protocol); (ii) lattice constants with spin: E(V) scans as F1 with the
  moment free at each volume (and B₀); the moment at each functional's own a₀ is reported too.
- Reference data: saturation moments and g′ factors (spin/orbital split); low-temperature lattice constants (+ c/a for Co) and bulk
  moduli, zero-point-corrected where available.

### 6.5 F3 (M4): paramagnetic metals bcc V, fcc Pd (spurious magnetism)

- nspin = 1 (NM) and nspin = 2 from an FM start and from a zero start; at the experimental lattice constant and along the F1-type
  E(V) scan (an expanded lattice favours spurious moments, so the moment is tracked along the scan).
- The physical outcome is a collapse to m = 0 with E(FM-start) = E(NM). Reported: the converged moment, E(start) − E(NM) per atom.
  A functional that keeps a moment at the experimental volume fails this test.
- Reference data: experimental evidence that V and Pd are not magnetically ordered (susceptibility / neutron data), low-temperature
  lattice constants.

### 6.6 F4 (M5): antiferromagnetic monoxides FeO, CoO, NiO

- **Cell.** Rocksalt with AFM-II order (ferromagnetic (111) planes stacked antiferromagnetically along [111]). With a the cubic
  rocksalt lattice constant, the rhombohedral cell
  a₁ = a(1, ½, ½), a₂ = a(½, 1, ½), a₃ = a(½, ½, 1)   (volume a³/2, two formula units),
  TM1 at 0, TM2 at ½(a₁+a₂+a₃) = a(1, 1, 1), O at ¼(a₁+a₂+a₃) and ¾(a₁+a₂+a₃). TM1 and TM2 lie on alternating (111) planes
  (x+y+z = 0, 2a vs a, 3a). TM1 and TM2 are separate species labels with the same pseudopotential and opposite
  starting_magnetization (±0.5 of the maximum); O neutral. FM reference: the same cell with equal starting magnetizations. A unit
  test checks the cell (neighbour shells, the AFM-II plane condition). Fixed cubic lattice at the experimental a (the small
  rhombohedral distortion below T_N is not modelled).
- **k-mesh.** Γ-centred Monkhorst-Pack on the rhombohedral reciprocal lattice, starting at 6×6×6 and converged by the F4
  criteria; Gaussian smearing as everywhere (FeO and CoO may come out metallic with semilocal functionals).
- **Symmetry-breaking start.** Two runs per functional and oxide: (a) AFM-II start with QE's symmetry (D3d of the magnetic cell);
  (b) the same start with `nosym = .true.` and the random component of the starting orbitals, which lets the minority-spin t2g
  occupation of Fe²⁺ (d⁶) and Co²⁺ (d⁷) break the trigonal symmetry (orbital order) if that lowers the energy. Both states and
  energies are reported; the lower one is the result, the other a labelled diagnostic. Plus the FM run (symmetric).
- **Reported:** site moments (QE's sphere-integrated magnetization at a fixed global radius per element, recorded), the cell's
  absolute magnetization, E(AFM) − E(FM) per formula unit, the GKS gap (band edges from the SCF mesh plus a non-SCF k-path with the
  converged ρ, τ and hooked v, v_τ held fixed). Semilocal functionals are known to be weak here (small or vanishing gaps,
  reduced moments); that is reported as is.
- **Reference data:** neutron local moments (spin + orbital; CoO and FeO have large orbital contributions, so the comparison is
  stated with that caveat), optical / photoemission gaps, Néel temperatures or exchange constants for the sign and size of
  E(AFM) − E(FM) (an indirect comparison, stated as such), lattice constants.

### 6.7 F5 (M7): the Zhang-Reilly-Tkatchenko-Scheffler set ("Sol62") and its subsets

Source (read 2026-11-10, main text only; archive `refdata/literature/ZhangReillyTkatchenkoScheffler2018_NJP20_063020.pdf`):
G.-X. Zhang, A. M. Reilly, A. Tkatchenko, M. Scheffler, New J. Phys. 20, 063020 (2018). What the main text fixes:

- 64 cubic solids (not 62; which 62 the user means is Q11): 24 metals and 40 semiconductors and insulators; structures A1 fcc (13),
  A2 bcc (11), A4 diamond (4), B1 rocksalt (16), B3 zincblende (20); classes MM 9 (groups IA, IIA, Al, Pb), TM 15 (periods 4-7),
  SC 24 (group IVA, IIIA-VA, IIB-VIA), IC 6 (LiCl, LiF, NaCl, NaF, MgO, MgS), TMCN 10 (carbides and nitrides of Ti, Zr, Hf, V, Nb).
  The full list is only in the supplementary material (Tables SII-SVII), which is not yet in the archive.
- Lattice constants, cohesive energies and bulk moduli from third-order Birch-Murnaghan fits to 7-11 points in V₀ ± 20 %.
- Zero-point effects: quasi-harmonic ab initio phonons (PBE level, 2×2×2 supercells), added to the DFT energies of every functional
  before comparing with low-temperature or 0 K-extrapolated experimental values (i.e. the correction is applied on the theory side).
  Average ZPE contributions: metals 0.2 % (a₀), 0.7 % (E₀), 1.4 % (B₀); non-metals 0.3 %, 1.1 %, 2.2 %; largest for LiF (about 0.04 Å
  and 5.6 GPa).
- Spin-restricted crystals except the ferromagnets Fe and Ni; all-electron FHI-aims, scalar relativistic, no spin-orbit coupling.

Plan once the supplement is read: the same F1 protocol (our 11-point ±12 % window, Q13), the per-solid experimental values and
the per-solid PBE ZPE contributions from the supplement (the bench compares static-lattice results with experiment corrected by
those contributions, equivalent to adding them on the theory side), SG15 availability checked for every element (Ti, Zr, Hf, Nb,
Pb, and whatever else the list contains; SG15 1.2 has all of these), cohesive energies only once M8 is designed. The overlap with
LC20, F2 and F3 (e.g. Fe, Ni, V, Pd if present) is reused, not recomputed.

### 6.8 M6, M8: LC20 band gaps; cohesive energies

- M6: GKS gaps of the LC20 semiconductors and insulators, as in F4 (SCF + non-SCF k-path, potentials held fixed); compared with
  experiment with the GKS-eigenvalue caveat.
- M8: needs spin-polarized, symmetry-broken atoms in large boxes: low-density tails where mixed ρ_σ may turn negative (raises,
  Sec. 4.5), and SCH-SI vacuum behaviour in plane waves is untested. Deferred (Q6).

---

### 6.9 MgO band gap (user 2026-11-11; staged, `xcsb.studies.mgo_gap`, `scripts/perlmutter/job_mgo_gap.sh`)

- LATTICE (user decision 2026-11-11, supersedes the static-lattice value below for the gaps): the gap grid runs at OUR r2SCAN a0
  (no sourced r2SCAN MgO value exists: Zhang et al. predates r2SCAN; the TMO paper has no MgO). Stage E: r2SCAN E(V) on 9
  volumes, a = 4.189 Å × (1 + δ), δ ∈ [−0.03, 0.03], at 2 E_c^PBE and 8³ k; third-order Birch-Murnaghan fit (`xcsb.bench.eos`:
  raises if V0 is outside the inner two thirds of the scanned volumes or the rms residual exceeds 2e-5 Ry per cell; EOS_STOP.txt
  then stops stage G) → a0, B0, B0′, banked once (`xcsb mgo bank-eos`, a reference-functional result). Order: PBE cutoff scan at
  4.189 Å (sets E_c) → r2SCAN EOS → gap grid at a0 for all four functionals. Estimate ≈ 0.7-1.3 node-h: one 4 h allocation.

- Rock salt, primitive 2-atom cell, a_static = 4.189 Å (Q10 rule: measured 4.207 − PBE zero-point expansion 0.018; Zhang et al.
  NJP 2018 SM Table III; archive `solids/lattice_constants_static_mgo_2026-11-11`), SG15 Mg, O; nspin 1, fixed occupations
  (16 electrons, 8 occupied bands), nbnd 12 with `diago_full_acc` (empty bands converged like the occupied ones); PBE, r2SCAN
  (reference functionals: banked once at this protocol, `xcsb mgo bank`), SCH-SI-A:v3.1, SCH-SI-B:v3.1, through the hooks,
  cold starts; one QE pool per SCF (the k list, hence the eigenvalues, is then complete).
- E_c^PBE: PBE scan 40-140 Ry at 8³ k; the smallest cutoff from which every successive pair has |ΔE_tot| ≤ 1 meV/atom (the
  F2/F3 energy criterion carried over to an insulator, which has no moment) and |Δgap_Γ| ≤ 0.02 eV (the F4 gap criterion).
- Grid: ecutwfc/E_c^PBE ∈ {1, 1.5, 2} × Γ-centred unshifted k ∈ {6³, 8³, 10³} (Γ, X, L lie on every even mesh), d = 4, all
  four functionals (36 SCFs), the gap at every point (a cutoff drift of SCH-SI is visible). Selected: the smallest r with
  |gap_Γ(r) − gap_Γ(r = 2)| ≤ 0.02 eV at 10³ k for all four, and the smallest k with |gap_Γ(k) − gap_Γ(10³)| ≤ 0.02 eV at r = 2;
  if only the largest value qualifies, the axis is reported as not shown converged.
- The gap: the GKS eigenvalue gap ε_CBM − ε_VBM of the converged SCF (the τ potential is non-multiplicative). Reported: the
  direct gaps at Γ, X, L, the minimum gap over the SCF mesh (with its k points), per functional with E_c^PBE, the settings, the
  iterations and the differences to PBE. Interpretation (Perdew-group research note 2026-11-11, archive literature; private,
  not for citation): for delocalized carriers in an extended solid and a continuous approximate GKS operator, an approximation's
  fundamental gap equals its own GKS orbital gap; this does not make it an accurate experimental gap.
- Experiment: no sourced experimental MgO gap in the archive literature (open item for the user: the fundamental gap, with
  zero-point renormalization and the exciton caveat).
- Bands along Γ-X-L (not built): the host has no non-SCF band calculation. It would take (i) the converged ρ_σ, τ_σ saved by
  the host on the node (npz in the run directory; not QE's files, which changed the density on read-back in M0), (ii) a second
  QEpy session with an explicit k list along the path and fixed occupations, whose potentials are set once from the saved
  fields through the existing hooks (extpot, extkin, extene) and then diagonalized without mixing (QEpy's nscf diagonalization,
  or one diagonalize call per k batch with the density update suppressed), with QE's own v_of_rho kept from overwriting the
  fixed potential (exttype 4 already switches QE's XC off; the Hartree part must come from the saved density), and (iii) a
  check that the same procedure on the SCF k list reproduces the SCF eigenvalues to 1e−5 Ry.
- Cost (Fe 80 Ry timing scaled by cell volume, nspin, irreducible k and (ecutwfc/80)^1.5): ≤ 0.3 h per SCF on a quarter node,
  all 46 SCFs ≈ 0.6-1.0 node-h for E_c 70-110 Ry: one 4 h allocation.

### 6.10 v4 Fe items (user 2026-11-11; strongxc-v4 line: run dirs v4_*, records line = v4, xchole pin strongxc-v4 dd8b5a517)

The v4 line pins xchole to strongxc-v4 dd8b5a517 (`XCSB_XCHOLE_LINE=v4`, `scripts/perlmutter/env_xcsb_v4.sh`): the v3.1 sets with
model sources (src/xchole/sch_si, core, ueg_gauge, benchmarks, explore/xc_pairing) identical to 0e9e8ee28 and byte-identical
records (sha256 checked), plus explore/spin_balance (the E1 reference function).

- Item 1, dense FSM scan + hysteresis (`xcsb.studies.fe_fsm_dense`, `job_fe_fsm_dense.sh`, run dir
  `$W/runs/xcsb/v4_2026-11-12_fe_fsm_dense`): m = 0.6-0.9 cold for A, B, A#ueg-end, PBE; for A and B also warm starts from the
  converged m = 0.5 and m = 1.0 wavefunctions of the FSM run (Sec. 5, mode 'wavefunctions'). Two branches at m: two converged
  starts differing by > 1e-4 Ry. Reported per point and start: E, iterations, m-weighted w_x per spin and w_c, piece energies;
  no orbital (Löwdin) projections: the host has none (QE's projwfc is not in the QEpy session).
- Item 2, frozen spin balance (`xcsb.studies.fe_sb_frozen`, `job_fe_sb.sh`, run dir `$W/runs/xcsb/v4_2026-11-12_fe_sb`): the
  converged fields of every FSM grid point are read from the point's QE save (`xcsb.host.qe_files`; checked on PBE/r2SCAN points:
  E_xc on the saved fields = the record's to 7e-12 Ry, electrons and min ρ identical; the saved LSDA magnetization coefficients
  are Hermitian to ~1e-7, the real part is used) and checked again per point (electrons per spin 1e-8; Σ w (e_x + e_c^SI) = the
  record's E_xc to 1e-7 Ry). Variants: E1 as defined (UEG-gauge reference, ω 90, base placement, R0 kin), PBE reference, R0 half,
  and ω = 0 (λ = 1: the ceiling, a diagnostic only). Per point ΔE; per curve the frozen corrected E(m), implied moment, E″ and E_xc″
  at m = 0 and at the implied moment; the gate diagnostics (v4's GateAccumulator) overall and in the 3d region (r ≤ 1.3 bohr of
  the Fe site; diagnostic only). Texture (frozen, no model term): density-weighted distributions (overall, 3d region, per density
  decade) of ζ, |∇ζ|/(2k_F) and |∇ζ|/(2k_s) (k_s = √(4k_F/π)), c_θ = ∇ζ·∇n/(|∇ζ||∇n|), w_c′, with the He-pair and flat pieces
  along m. Cross-check first: the reference function on v4's testpoints_sb.json (every stored key; whether it was regenerated
  with R0 kin for both sets is recorded).
  ω scan (user 2026-11-11): E1 as defined at ω ∈ {10, 30, 90, 300, 1000} plus λ = 1: implied moment, E_xc″ at m = 0 and at
  the implied moment, retained δ_m fraction in the 3d region. A SENSITIVITY diagnostic only: Fe/Co/Ni moments are a hold-out and
  never choose ω (ω is a run-time argument of the reference function: no extra compile).
- Item 3, self-consistent spin-balanced r2SCAN (`xcsb.studies.r2scan_sb`, `xcsb.functional.external`, `job_r2scan_sb.sh`, run dir
  `$W/runs/xcsb/v4_2026-11-12_r2scan_sb`): the Perdew-group functional as implemented on strongxc-v4
  (explore/spin_balance/sb_r2scan.py; names r2SCAN-SB (R0 half, primary), r2SCAN-SB-kin; ω = 90) through an external per-point
  adapter (rho, grad rho, tau per spin; exact first derivatives from the external code; no Laplacian; interface in the adapter's
  docstring); xchole line v4-sb, pin set when the functional lands. Phase 0: the energy-vs-potential FD check
  (tests/test_external_fd.py). PBE, r2SCAN, r2SCAN-SB, r2SCAN-SB-kin at identical settings (80/320 Ry, 8³ k, Gaussian 0.02,
  conv_thr 1e-8, β 0.3, FM start) on bcc Fe (static 2.858 Å) and fcc Ni (static 3.507 Å; archive
  solids/lattice_constants_static_ni_2026-11-11); Fe also at m = 2.2, 2.4, 2.6. hcp Co: not in the Zhang et al. tables (open:
  a sourced static Co lattice and c/a). The 80 Ry cutoff is the Fe setting; the ladder has not covered Ni. PBE and r2SCAN are
  computed once at this protocol and banked (the earlier Fe80 runs used a = 2.866 Å).
  Landed 2026-11-11 at strongxc-v4 b2501727c (pin line v4-sb; model sources and records identical to v3.1): the adapter calls
  sb_r2scan.evaluate_arrays(rho, grad, tau, r0_form, omega = 90); FD-checked through the adapter on the laptop (both names,
  polarized and unpolarized) and again in phase 0 on the node.
- Item 4, the E1 ω fit (user decision 2026-11-11, strongxc-v4 rules exception 92b1cfb56; `xcsb.studies.fe_omega_fit`,
  `job_fe_omega_fit.sh`, run dir `$W/runs/xcsb/v4_2026-11-12_fe_omega_fit`): per set, SCH-SI-X:v3.1 + E1 (UEG reference, base
  placement, R0 kin; sb_model.install, exact JAX derivatives, FD-checked in phase 0) self-consistently on bcc Fe at the SB-Fe
  settings (static 2.858 Å, 80/320 Ry, 8³ k, Gaussian 0.02, FM start, cold carrier start); target = our PBE moment at THESE
  settings (the r2SCAN-SB job's PBE Fe record; the FSM free PBE 2.381 μB was at 2.866 Å). Bracket from the frozen ω scan; 3 SC
  points (bracket ends and geometric mean); ω* by interpolation in log ω; the H-atom check of the term at ω* (|ΔE| ≤ 1e-6 Ha);
  a confirming SC run at ω* with the E1 parts. Functional names 'SCH-SI-A:v3.1+E1(omega=<w>)', 'SCH-SI-B:v3.1+E1(omega=<w>)' (ω in
  the provenance); xchole line v4-e1 = strongxc-v4 76cfa7c20 (the E1 SCF path verified there; code identical to b2501727c); the
  target (PBE at 2.858 Å from the r2SCAN-SB job) confirmed by the user 2026-11-11. STOP rules: no bracket, a bracket below ω 10, m(ω) not monotone or not reaching
  the target, the H check failing. Co and Ni stay hold-outs.

- Results so far (2026-11-12). Dense scan (job 61167245): TWO FSM branches below m = 1. Cold starts at m = 0.6-0.9 (A; B 0.7-0.8)
  land 0.036-0.043 Ry below the branch continued from m = 0.5 (w_x,↓ ~0.30 vs ~0.43): the m ≤ 0.5 cold points of the FSM run are
  on the upper branch, so E″(0) and the Stoner analysis of the earlier curves describe that branch. Warm starts from m = 1.0 all
  failed at once (the source has 13 bands, the targets 12: QE's default nbnd depends on tot_magnetization; fixed: a wavefunction
  warm start now takes the source's band count, recorded as a deviation); B 0.9 from 0.5 did not converge in 200 iterations.
  Follow-up (job_v4_combo.sh, `xcsb.studies.fe_fsm_lowm`): cold starts at m = 0.2-0.5 (A also 0.1) with starting_magnetization
  0.2 / 0.5 / 0.8 (the only change between starts), a = 2.866 Å as the FSM run, and the lower-branch E(m)
  (`scripts/fe_fsm_lower_branch.py`). r2SCAN-SB (job 61167271): Fe moments PBE 2.359, r2SCAN 2.778, r2SCAN-SB 2.678, -kin 2.692 μB
  (not PBE-like). Frozen on the converged ω = 90 densities the gate retains 71 % (half) and 93 % (kin) of δ_m, but that does NOT
  predict the self-consistent response: stage R (job 61170583, SC, same settings) gives r2SCAN-SB (half) |m| = 2.283 / 2.325 /
  2.429 / 2.678 μB at ω = 0 / 10 / 30 / 90 (PBE 2.359), i.e. strongly gate-limited; r2SCAN-SB-kin 2.678 / 2.679 / 2.682 / 2.692 (ω
  insensitive: its δ_m is about half of the half-split one, 365 vs 745 meV at ω = 0). Ni (static 3.507 Å): every first run aborted in
  the TPSS carrier start (random wavefunctions: the SG15 files carry no atomic orbitals; divergence at iteration 3 at β 0.3); with
  the carrier phase at β 0.1 (a recorded deviation of the start only) all converged: PBE 0.691, r2SCAN 0.696, r2SCAN-SB 0.683,
  r2SCAN-SB-kin 0.694 μB.
  DECISION (user 2026-11-12): r2SCAN-SB means ω = 10 from now on (the half-split form; Fe 2.325 vs our PBE 2.359 μB; a linear
  interpolation between ω 10 and 30 would put the exact match near ω ≈ 16; 10 is the user's choice): `external.DEFAULT_OMEGA`, with
  the provenance "ω = 10, chosen by the user to match OUR PBE Fe moment in this setup (SG15, 80/320 Ry, a 2.858 Å); not the
  Perdew-group value (90 in their note)". r2SCAN-SB-kin takes ω = 10 as well, for consistency (it is insensitive to ω). The
  earlier default runs (ω = 90) are relabelled 'r2SCAN-SB(omega=90)' / 'r2SCAN-SB-kin(omega=90)' (run directories renamed,
  RELABEL_omega90.json; records unchanged, their provenance carries ω = 90). The NiO/MnO job runs r2SCAN-SB at ω = 10 plus
  r2SCAN-SB(omega=90) as a labelled comparison.
  DECISION (user 2026-11-12): the HALF split R0 is ADOPTED for OUR E1 term: '<set>+E1(omega=<w>)' installs sb_model with
  r0_form 'half' (the model form; the ω fit uses it too); '<set>+E1kin(omega=<w>)' is the labelled comparison. (The earlier frozen
  item 2 ran with kin as its default and half as a labelled row, before this decision.) SC E1 job (job_e1_sc.sh,
  `xcsb.studies.fe_e1_sc`, run dir $W/runs/xcsb/v4_2026-11-12_fe_e1_sc): installed A/B at 2.858 Å and A/B + E1 / E1kin at ω 10,
  30 (FD phase 0 at ω 10), with r2SCAN-SB / -kin at ω 10 for Ni (β 0.1 start) and the Fe points m = 2.2, 2.4, 2.6.
  SC ω scan (user go 2026-11-12; job_e1_scan.sh, stage S of `fe_e1_sc`, same run dir): A/B + E1 (half) at ω 60, 90, 150, 300,
  combined with ω 10, 30 into |m|(ω); ω* by log-ω interpolation to the TARGET 2.22 μB (user decision 2026-11-12: the experimental
  TOTAL moment of bcc Fe, user-provided, no archive source read by us; our moments are spin moments, collinear without SOC, so the
  fit compares a spin moment with a total moment by the user's choice, and it absorbs the SG15 no-NLCC offset of our absolute
  moments), with our PBE 2.359 μB as a reference column. This supersedes the earlier fit target (our PBE at 2.858 Å).

### 6.11 v4: AFM NiO and MnO gaps (user 2026-11-11; staged: `xcsb.studies.tmo_gap`, `job_tmo_gap.sh`)

Reproduction of the Perdew group's statement (private communication) that spin-balanced r2SCAN improves the NiO and MnO gaps:
the term and its potential vanish at spin-symmetric states, so only magnetic insulators can show it.

- Geometry (user decision 2026-11-11: r2SCAN lattices for the band-gap tests, replacing the static-lattice rule Q10 for these
  tests only; the SAME geometry for every functional, later A/B + E1 too): the r2SCAN-relaxed AFM cells of Swathilakshmi, Devi,
  Sai Gautam (JCTC 2023; VASP PAW, no U), from github.com/sai-mat-group/r2SCAN-U-benchmarking 839f4a00 (CONTCAR and OUTCAR read
  as data; archive `solids/tmo_structures_r2scan_relaxed_2026-11-11`). NiO: their 4-atom cell (2.940/5.874/2.940 Å, ~60°);
  MnO: their 16-atom cell reduced along its magnetic order to 4 atoms (A1 = a1/2, A2 = a2/2, A3 = a3; 3.152/3.152/6.186 Å,
  59.37/59.37/60.00°, MnO's relaxed rhombohedral distortion kept); both agree with their SI Table S2 r2SCAN rows. Magnetic order
  from their MAGMOM and final moments (TM1 +, TM2 −).
- nspin 2 with ±0.5 starting moments on TM1/TM2, fixed occupations with tot_magnetization 0, no U, nbnd = n_occ + 8 with
  diago_full_acc, SG15 Ni/Mn/O (Mn added to the manifest 2026-11-11: z 15, no NLCC), plain Broyden β 0.3.
- PBE cutoff scan 60-140 Ry at k level 4, then a short PBE check (cutoff ×1, 1.5, 2 at level 4; levels 4, 6, 8 at ×1), then
  PBE, r2SCAN, r2SCAN-SB, r2SCAN-SB-kin at the selected settings (xchole line v4-sb). k level n: the mesh n × (1, ½, 1) for
  NiO and n × (1, 1, ½) for MnO (the cells' vector lengths 1:2:1 and 1:1:2).
- Reported: the minimum GKS gap over the mesh with the band-edge k points and spins, the TM site moments (∫(ρ↑ − ρ↓) within
  2.0 bohr) and the absolute magnetization, iterations.
- References (archive `solids/tmo_expt_r2scan_swathilakshmi2023_tableS2`, SI Table S2 read, original references not read):
  experimental MnO 3.60-3.80 eV, 4.580 μB; NiO 4.30 eV, ±1.64-1.90 μB (gap type not stated in the SI); their r2SCAN (external,
  different PP/code) MnO 1.45 eV, 4.446 μB; NiO 2.11 eV, 1.577 μB. No zero-point information.
- Their protocol (main text Sec. 2.1, read; archive `solids/tmo_protocol_swathilakshmi2023_2026-11-12`): VASP 6.2.1,
  frozen-core PBE PAW (datasets of their refs 38, 39, names not given), 520 eV; Γ-centred MP at 48 k per Å (48/|a_i| rounded
  up); relaxations without symmetry; Gaussian 0.05 eV; GAPS from the DOS of a single SCF with zero-weighted k-points at 96 per Å
  (−20 to 20 eV, 0.005 eV steps); on-site moments with an unstated sphere. Comparability: their gap is a smeared DOS gap on a
  denser mesh, ours the GKS eigenvalue gap on our SCF mesh with fixed occupations; their moment sphere is unknown, so ours are
  reported at 1.6, 2.0 and 2.4 bohr. Gap type (optical vs fundamental) of the experimental values: stated in neither the main text
  nor the SI (open).
- Cost (scaled from Fe80, uncertain): about 50 min per SCF at 100 Ry, level 4, quarter node (AFM ~100 iterations); level 8 on a
  full node ~1-2 h; stages P+G ≈ 2 allocations, stage F ≈ 1-2 allocations of 4 h (batch would fit better).

### 6.12 Relaxed lattice constants of ionic solids (user 2026-11-13; staged: `xcsb.studies.ionic_lc`, `job_ionic_lc.sh`)

User request: "test the relaxed lattice constants of ionic solids using the approach my collaborator has used". This is the
collaborator's LC20 protocol applied to the five rock-salt members of LC20, not the F1 protocol of Sec. 6.3 (11 volumes, ±4 %,
d = 4, own convergence ladder), which stays the plan for the full family. Nothing was read from the collaborator's code: the
protocol and the numbers were given by the user.

- Solids and reference: LiF 3.972, LiCl 5.070, NaF 4.582, NaCl 5.569, MgO 4.189 Å (conventional; zero-point-corrected
  experiment, LC20 set). Values, per-element cutoff floors, the collaborator's cross-check values and all source strings:
  `src/xcsb/studies/ionic_lc_reference.json` (NOT yet an archive entry: banking is outside this repo; open item).
- Functionals: PBE, r2SCAN (reference functionals: computed once in this protocol, then banked, `xcsb lc bank`),
  SCH-SI-A:v3.2, SCH-SI-B:v3.2. xchole line `v3.2` = strongxc-v3 019c8acd6 (the v3.2 sets replace v3.1 there; the adapter
  selects the installed records by line, `RECORDS_BY_VERSION`; a v3.1 name on the v3.2 line, or the reverse, raises).
- Cell and SCF: rock salt, primitive, 2 atoms, nspin 1, fixed occupations, nbnd = n_occ + 4 (n_occ from the SG15 valence
  charges: LiF, LiCl 5; NaF, NaCl, MgO 8); ecutwfc = max(80 Ry, element floors Li 82, F 96, Na 96, Cl 66, Mg 96, O 96) = LiF 96,
  LiCl 82, NaF 96, NaCl 96, MgO 96 Ry; ecutrho = 8 ecutwfc; conv_thr 1e-8 Ry, Broyden β 0.3; cold starts with the start
  `pbe-scaffold` (Sec. 5: XC-free step, converged PBE through the hooks, then the functional; the collaborator: iteration 0 =
  libxc PBE through the hooks).
- STALL of the first attempt (jobs 61245307 ref, 61245548 model, 2026-11-13, cancelled after 50 min; start `tpss-carrier`,
  8 ranks × 4 threads, 8 pools): phase 0 passed (MgO 40 Ry, 6³ k: PBE 18 + 12 iterations in 104 s, SCH-SI-A:v3.2 18 + 17 in
  274 s), but all eight LiF SCFs (624 k, 96 / 768 Ry, 9 bands) were still in carrier iteration 1 after 48 min, pw.out showing
  18-31 lines "c_bands: 1 (or 5) eigenvalues not converged". Hypotheses, ranked: (1) QE's TPSS on the starting state (potinit:
  the atomic superposition, which is not positive, with a Thomas-Fermi guess for τ, so τ < τ_W wherever gradients are large:
  hard Li 1s / F 2p cores at 96 Ry) gives a first Hamiltonian whose τ-dependent kinetic term Davidson cannot converge; each k
  point then runs to Davidson's iteration limit (~20-30 × a normal solve), times 78 k per pool. The carrier is the common
  factor of PBE and set A, the warnings are QE's own, and the collaborator reports the same for QE's TPSS start. (2) Layout:
  78 k points per pool in sequence, with threads that buy little on 36³-48³ grids, so even a healthy iteration is slow (the
  first scheduling estimate assumed the Fe timing, 5-10 × too low). (3) Random starting orbitals (SG15 UPFs are believed to
  carry no pseudo-wavefunctions, so 'atomic+random' is random; to confirm in pw.out) with only 4 buffer bands: slower first
  solves, but not non-convergence by itself. MgO in `mgo_gap` used the same carrier, diagonalization and starting orbitals
  (differences: dual 4, 12 bands with diago_full_acc, 16-47 k points, no Li / F): no setting protected it; either its starting
  Hamiltonian is benign or the slow first iteration went unnoticed at 29 k points (its records are not on the laptop).
- DIAGNOSTIC (`launch_ionic_lc.sh diag`, `job_ionic_lc_diag.sh`, 1 node, 30 min; `xcsb lc diag-summary`): LiF at a_ref, PBE;
  phase A, 12³ k (72 irreducible): old start | new start | new start with SCH-SI-A:v3.2 (hook cost) | new start on 8 ranks ×
  4 threads; phase B, 28³ k: old start | new start (the production slot) | new start on 64 pools. Output: per-iteration wall
  times, QE's cpu times and c_bands warnings per iteration, seconds per k point and iteration, energy differences between
  starts and layouts, `diag_summary.md`, and `COST.json` (measured c_k, hook time, iteration counts) for the estimates.
- k mesh: N_i = max(1, ceil(|b_i| / 0.1 Å⁻¹)), |b_i| with 2π (VASP k-spacing convention), from the reference cell, fixed across
  the scan: LiF 28³ (624 irreducible), LiCl 22³ (328), NaF 24³ (413), NaCl 20³ (256), MgO 26³ (511); Γ-centred, unshifted (the
  host's convention, Sec. 6.0). Layout: all parallelism in k-point pools: a quarter-node slot = 32 ranks × 1 thread (one
  physical core each), npool 32 (8-20 k points per pool); the campaign runner takes the layout from the study
  (`srun_layout`), other studies keep 8 ranks × 4 threads.
- Scan and fit: a = a_ref (1 + s), s = −3, …, +3 % (7 points; stages R: reference functionals, M: model sets); third-order
  Birch-Murnaghan (`xcsb.bench.eos.bm3_fit`). EXTENSION stage (RX, MX; scan design, not a relaxed check): when the unguarded
  fit of a complete scan has V0 outside the inner two thirds of the scanned volumes, points are added at 1 % spacing on the
  side of the minimum (as many as put that V0 inside, at least one; |s| ≤ 8 %) and the test repeats; the final, guarded fit
  uses all points. A scan not bracketed at |s| = 8 %, or a failed fit, gives no value (reported). `bm3_fit(V, E, guard=False)`
  returns the two guard conditions as flags; the default (guarded) behaviour is unchanged for every other study. Expected from
  the collaborator's PBE values: LiF, NaF, NaCl (a0/a_ref − 1 = 2.3-2.6 %, beyond the inner bound 2.05 %) get one PBE extension
  point (+4 %); r2SCAN none.
- Table (`xcsb lc table`): per solid and functional a_ref, a0, a0 − a_ref, B0, B0′, rms, the scan range (extended scans
  marked), the plain ±3 % fit (unguarded, labelled; the collaborator's protocol like for like) next to the collaborator's a0
  and B0 (cross-check, their own host); ME, MAE, MARE over the five solids for both fits.
- Cost (scheduling): seconds = overhead + (1 + it_scaffold + it_hook) × ⌈n_k / 32⌉ × c_k + it_hook × hook, c_k = c_k(LiF) ×
  (V / V_LiF)(ecutwfc / 96)^1.5 (nbnd / 9). Before the diagnostic only a PRIOR exists (c_k(LiF) 6 s per k point and iteration
  on one rank, an upper bound from phase 0 of the stalled job; the collaborator's 49 s per LC20 point on 128 ranks would mean a
  few tenths of a second): 140 base points ≈ 42 node-h at 6 s, ≈ 10 at 1 s, ≈ 6 at 0.4 s (then the SCH-SI hook dominates).
  The diagnostic's `COST.json` replaces the prior (copied into the run directory by the job); within a run the estimates also
  rescale from the finished points (median measured / estimated, from 4 points on). Stages resume (finished points skipped;
  claims let two allocations share the run directory).
- Job: `launch_ionic_lc.sh diag|ref|model|all [pin]` (1 node, m2360; diag 30 min, the others ≤ 4 h; own clone at main; xchole
  clone `$W/src/xchole-v3.2` at the pin, passed to the job explicitly; refuses at ≥ 2 interactive jobs). Run directories:
  `2026-11-13_ionic_lc_diag`, `2026-11-13_ionic_lc_scaffold` (the stalled attempt's `2026-11-13_ionic_lc` is not reused).
  Phase 0 of the production job: the four functionals load at the pin; PBE and SCH-SI-A:v3.2 converge `LC-phase0` (MgO, 40 Ry,
  6³ k, new start; plumbing only); a failure stops the job.

## 7. Verification tiers

| tier | what | where | pass |
|---|---|---|---|
| V1 | spectral derivatives vs closed forms on band-limited periodic test densities (finite Fourier series inside the sphere; triclinic, bcc, fcc, hcp and AFM-II cells), both spin channels; ∇²ρ = tr H; adjointness ⟨D_i f, g⟩ = −⟨f, D_i g⟩ and symmetry of the second-order operators; Σ M = ngm_g; adapters: libxc vs finite differences, xchole vs the handoff sanity points, zero-padding points | laptop, seconds | 1e−12 relative (derivatives, adjoints); 1e−10 (sanity points) |
| V2a | potential vs finite differences of the grid energy: E_grid(ρ ± εδρ_σ) and E_grid(τ ± εδτ_σ) along random sphere-limited directions, **each spin channel separately** (δρ_↑ only, δρ_↓ only, δτ_↑, δτ_↓) and in the nspin = 1 and nspin = 2 layouts, on polarized test densities (ζ up to ±0.9); test functional, PBE, r2SCAN, SCH-SI-A, -B | laptop, seconds | central differences at the O(ε²) level, ≤ 1e−7 relative |
| V2b | orbital-level GKS check: a numpy plane-wave toy (a few orbitals per spin, several k) with ρ_σ, τ_σ from orbitals; FD of E_xc along occupied-virtual rotations within each spin channel vs 2 Re⟨δψ∣Ĥ_xc,σ∣ψ⟩, Ĥ_xc,σ ψ = v_σψ − ½∇·(v_τ,σ∇ψ) (Ha); validates the ½ in τ and the τ operator per spin | laptop, seconds | ≤ 1e−7 relative |
| V3 | (finding, M0: QE's native meta-GGA drivers set e_xc = 0 at grid points with \|∇ρ\|² ≤ 1e-24, i.e. at centres of symmetry on the grid; its TPSS therefore differs from libxc TPSS by mRy on symmetric cells. TPSS is compared on cells without such points, the host E_xc check removes exactly those points, and the symmetric TPSS cases are diagnostics.) the hooks reproduce QE native: libxc PBE vs `input_dft='PBE'`, libxc TPSS vs `'TPSS'`; (i) bulk Si, nspin = 1, fixed occupations; (ii) **bcc Fe, nspin = 2, Gaussian smearing, FM**; and the host's E_xc of the native ground states vs QE's etxc in both cases | node | ΔE_tot ≤ 1e−5 Ry/atom, eigenvalues ≤ 1e−5 Ry, moment ≤ 1e−4 μB (limited by QE's internal density thresholds and QE vs libxc implementation details; the measured differences are recorded) |
| V4 | spin consistency: nspin = 2 at zero magnetization (Si with tot_magnetization = 0) equals nspin = 1 | node | ΔE ≤ 1e−8 Ry |
| V5 | MPI: the host step (inputs → potentials) bitwise identical for 1, 4, 32 ranks on the same gathered input, nspin = 1 and 2; a full SCF at 1 vs 32 ranks (QE itself is not bitwise reproducible across rank counts) | node | bitwise / ΔE ≤ 1e−8 Ry |
| V6 | smoke SCFs: bulk Si, then FM bcc Fe, with PBE, r2SCAN, SCH-SI-A, -B through the hooks, cold starts | node | converge; timings recorded (they replace the estimates of Sec. 9) |

V1-V2 are in the laptop test suite (seconds). V3-V6 are node tests (one interactive allocation). Every potential the bench uses
passes V2 before any benchmark lane runs. Before each lane a derivative pre-check (first partial derivatives finite, no spikes) runs
on the densities of that lane's systems, as in the workspace smoothness pre-check rule.

---

## 8. Reference data plan

Storage follows the user's archive pattern: an orphan branch `archive` with worktree `refdata/` (gitignored on main), write-once
JSON/CSV with provenance (source, table/page, retrieval date, checked by whom), never recomputed, read via `xcsb.refdata.path`.
Reference-functional results (PBE, r2SCAN; SCAN, TPSS if wanted, Q9) are computed once per protocol, banked there and never
recomputed; only the models under test rerun. Creating the `archive` branch needs the user's permission (Q2).

**Lattice constants (Q10, user 2026-11-11).** Every solids calculation at a fixed experimental lattice (M1, F2-F4) and every
comparison of computed equilibrium lattice constants with experiment (F1 LC20, F5 Sol62) uses STATIC-lattice values:
a_static = a_measured − Δ_ZP, with the zero-point expansion Δ_ZP of the source (a written-once archive entry per set, the
measured values and Δ_ZP recorded beside the result; first entry `solids/lattice_constants_static_2026-11-11`, Fe and V).

**Only the Zhang et al. main text (F5) has been read.** Every other citation below is a candidate from memory, to be confirmed when
fetched and read; no value is banked from a source before it has been read and checked. The LC20 source paper is not in the
archive yet (to fetch).

| family | needed | candidate primary sources (to fetch) |
|---|---|---|
| F1 | LC20 definition | J. Sun, M. Marsman, G. I. Csonka, A. Ruzsinszky, P. Hao, Y.-S. Kim, G. Kresse, J. P. Perdew, Phys. Rev. B 84, 035117 (2011) |
| F1 | zero-point-corrected experimental a₀ and B₀ | P. Hao, Y. Fang, J. Sun, G. I. Csonka, P. H. T. Philipsen, J. P. Perdew, Phys. Rev. B 85, 014111 (2012); L. Schimka, J. Harl, G. Kresse, J. Chem. Phys. 134, 024116 (2011); the zero-point anharmonic-expansion estimate of A. B. Alchagirov, J. P. Perdew, J. C. Boettger, R. C. Albers, C. Fiolhais, Phys. Rev. B 63, 224115 (2001); cross-check against K. Lejaeghere, V. Van Speybroeck, G. Van Oost, S. Cottenier, Crit. Rev. Solid State Mater. Sci. 39, 1 (2014); and the low-temperature measurements these cite |
| M1, F2 | saturation moments, g′ factors (spin/orbital split) of Fe, Co, Ni | H. Danan, A. Herr, A. J. P. Meyer, J. Appl. Phys. 39, 669 (1968); R. A. Reck, D. L. Fry, Phys. Rev. 184, 492 (1969); J. Crangle, G. M. Goodman, Proc. R. Soc. Lond. A 321, 477 (1971) |
| M1, F2, F3 | low-temperature lattice constants of V, Fe, Co (+ c/a), Ni, Pd; bulk moduli | from the F1 compilations if they contain them, otherwise the primary diffraction / ultrasonic papers they cite (to identify) |
| F3 | V and Pd not magnetically ordered (susceptibility, neutron) | primary sources to identify |
| F4 | neutron local moments of FeO, CoO, NiO | W. L. Roth, Phys. Rev. 110, 1333 (1958); B. E. F. Fender, A. J. Jacobson, F. A. Wedgwood, J. Chem. Phys. 48, 990 (1968); A. K. Cheetham, D. A. O. Hope, Phys. Rev. B 27, 6964 (1983) |
| F4 | gaps | NiO: G. A. Sawatzky, J. W. Allen, Phys. Rev. Lett. 53, 2339 (1984); CoO: J. van Elp et al., Phys. Rev. B 44, 6090 (1991); FeO: H. K. Bowen, D. Adler, B. H. Auker, J. Solid State Chem. 12, 355 (1975) |
| F4 | exchange constants / Néel temperatures (sign and size of E(AFM) − E(FM)), lattice constants | NiO magnons: M. T. Hutchings, E. J. Samuelsen, Phys. Rev. B 6, 3447 (1972); others to identify |
| F5 | Sol62 definition | G.-X. Zhang, A. M. Reilly, A. Tkatchenko, M. Scheffler, New J. Phys. 20, 063020 (2018): main text READ (in the archive); its supplementary material (Tables SII-SVII: the solid list, experimental a₀, E₀, B₀ with their literature, ZPE contributions, k-grids) TO FETCH |
| methods | | SG15: M. Schlipf, F. Gygi, Comput. Phys. Commun. 196, 36 (2015); ONCV: D. R. Hamann, Phys. Rev. B 88, 085117 (2013); QE: P. Giannozzi et al., J. Phys.: Condens. Matter 21, 395502 (2009) and 29, 465901 (2017); QEpy: its paper (citation from the QEpy documentation) |

Literature PDFs go to the workspace literature archive (`refdata/literature/` of the xcHoleModelMeta archive branch, its
LITERATURE.md updated whenever a paper is added or used), unless the user prefers a literature folder in this repo's archive (Q2).

---

## 9. Build plan on Perlmutter and cost

All writes inside `$W=/pscratch/sd/d/dsambit/xc_functional_dev`; built on a compute node (1-node interactive ≤ 4 h on m2360, after
checking with squeue that the user has fewer than 2 interactive jobs across all workstreams); code arrives only via git (bare repo
`$W/xcSolidsBench.git`); one script `scripts/perlmutter/build_env.sh`, logging every version.

1. Modules: `PrgEnv-gnu`, `cray-mpich`, `cray-fftw`, `cray-libsci` (gfortran, needed by f90wrap), `conda`.
2. Conda env `$W/envs/xcsb` (Python 3.12): numpy (< 2.5, QEpy's bound), scipy, meson, ninja, `f90wrap<=0.2.16` (QEpy's bound),
   pytest; `mpi4py` built from source against cray-mpich (`MPICC="cc -shared"`); `jax` (CPU) and `pyscf` at the versions xchole was
   validated with (jax 0.11.2, pyscf 2.14.0); `dftd4` 4.3.0; `xchole` installed editable from a clone of `$W/xcHoleModelMeta.git` at
   the v3.1 pin (read-only use).
3. QE 7.6: clone tag `qe-7.6` into `$W/build/q-e-7.6`; `./configure MPIF90=ftn CC=cc --enable-openmp` with `-fPIC` in FFLAGS and
   CFLAGS (QEpy links QE's static libraries into a Python extension); `make pw` (~15-30 min on 32 cores).
4. QEpy: clone `Quantum-MultiScale/QEpy`, check out `eb03de77` (branch qe76), `git apply patches/qepy-qe76-tau-hook.patch`,
   `qedir=$W/build/q-e-7.6 python -m pip install --no-build-isolation .` (~20-40 min); verify `import qepy` and
   `Driver.set_external_kinetic_potential`.
5. Pseudopotentials (login node, download only): SG15 1.2 into `$W/pseudo/sg15-1.2/`, sha256 checked against the manifest.
6. Caches: `JAX_COMPILATION_CACHE_DIR=$W/cache/jax`.

**Rough cost** (to be replaced by the measured V6 timings before any campaign). SCH-SI evaluation ≈ 2e−3 core-s per grid point
(handoff: a few seconds for 2·10⁴ points on 12 threads), libxc negligible; QE cost from plane waves × bands × k.

| family | cells, grid (d = 4) | per SCF (one 128-core node unless stated) | campaign |
|---|---|---|---|
| M1 step 1, cutoff ladder | 1 atom; E_c^PBE 60-100 Ry; up to ×3; d = 4: 21³-56³, d = 8 up to ~80³; 20³ k (256 irreducible) | from the Fe80 timings (npool 1): 0.15-1.7 h per SCF on a quarter node, the r = 3 rungs 0.5-1 h on a full node, + 3 seeds ≈ 1.1-2.3 h (Sec. 6.2, as staged) | stage P (PBE scan) ≈ 3 node-h; 132 ladder SCFs + 8 seed SCFs ≈ 36-66 node-h (E_c 60-100 Ry): 10-18 interactive allocations, or batch (Q7) |
| M1 steps 2-3, Fe, V | 1 atom; ecutwfc from step 1; 20³-24³ k (≈ 250-450 irreducible) | 2-5 s/step, 40-80 steps: 2-7 min at ~100 Ry | k/smearing study (mostly PBE) + 4 functionals × 2 metals × 3 runs: ≈ 2 node-h |
| F1 LC20 | 1-2 atoms; 32³-48³, alkali metals up to ~60³ | 2-15 min (SCH-SI dominates on the large grids) | 11 volumes × 20 solids × 4 functionals ≈ 900 SCFs, packed 4 lanes per node: ≈ 10-60 node-h, plus convergence |
| F2 Fe, Co, Ni | 1-2 atoms | 2-10 min | moments + E(V) scans × 4 functionals: ≈ 5-10 node-h |
| F3 V, Pd | 1 atom | 2-7 min | 3 starts × E(V) scans × 4 functionals: ≈ 3-6 node-h |
| F4 FeO, CoO, NiO | 4 atoms (2 f.u.); ~120 Ry (O and TM semicore); ~72³ (~4·10⁵ points); ~5·10³ PW; 6³-8³ k | ~10 s/step, 80-150 steps: 15-25 min; the `nosym` run several times longer | 3 runs × 3 oxides × 4 functionals + convergence: ≈ 15-30 node-h |

M1 steps 2-3 fit one allocation; the step-1 ladder needs several (its top rungs dominate). F1, F4 and the full F2/F3 scans exceed what a few 4 h interactive allocations can do;
batch jobs would need the user's permission (Q7).

---

## 9b. M0 status (2026-11-10)

Implemented and verified (results/m0/, tier summaries; node runs at QE 9f93ddec, QEpy eb03de77 + patch fae7f62d):
V1, V2a, V2b pass for the test functional, PBE, r2SCAN, TPSS and SCH-SI-A/B:v3.1 (handoff sanity points to 1e-10);
V3 passes for PBE (Si, Fe nspin 2) and TPSS on Si-lowsym; the host E_xc check passes on every native density after removing
QE's meta threshold points; on Fe-offset TPSS (nspin 2) energy and host E_xc pass but the moment (3.3e-4 μB) and occupied
eigenvalues (4e-5 Ry) differ by more than the tolerance (open: QE's native spin-polarized TPSS potential vs libxc's;
our potential passes V2 per spin); V4, V5 (bitwise, 8 and 32 ranks), V6 (Si, Fe; PBE, r2SCAN, SCH-SI-A, -B) pass.
Findings fixed during M0: the SCF start (carrier converged in the same session), Gaussian smearing (occupations in [0, 1]),
convergence = conv_elec, one SCF per process (QE keeps functional state between QEpy Drivers), QE's meta-GGA thresholds.

## 10. Open questions for the user

- Q1. License of this repository (QEpy is GPL-2; we ship only a patch to it).
- Q2. Archive: may I create an orphan `archive` branch in this repo (worktree `refdata/`) for reference data and banked
  reference-functional results? Literature in the existing workspace literature archive, or in this repo's archive?
- Q3. SG15 version: 1.2 (the latest; all needed elements present), fixed for the whole bench?
- Q4. Moments: compare the spin moment (via g′) with the computed spin moment, the total moment reported alongside?
- Q5. Cutoffs (if the ladder plateaus): one global ecutwfc for all systems, or per-system converged values fixed across
  functionals (proposed)?
- Q6. Negative mixed densities: the host raises. If they appear (atoms in boxes, M8), which structural remedy do you want?
- Q7. Batch permission for F1 (and F4), once the M1 timings are known.
- Q8. PBE-D4 for solids: report raw and + D4 for lattice constants and bulk moduli (the v3.1 handoff asks the collaborator for both)?
- Q9. Should SCAN and TPSS join PBE and r2SCAN as banked reference functionals?
- Q10 DECIDED (user 2026-11-11): static-lattice, zero-point-corrected experimental values (a_measured − Δ_ZP) for every solids calculation (M1 ladder, and the same rule later for LC20 and Sol62; Sec. 8). Was: Fixed lattices (M1, F2-F4 moments): low-temperature experimental values with or without zero-point correction? Proposed:
  zero-point-corrected where the source provides it, the uncorrected value reported alongside.
- Q11. "Sol62": Zhang et al. (NJP 2018) has 64 solids. Which 62 do you mean (which two are dropped, and why)?
- Q12. F4: is QE's sphere-integrated site moment (fixed radius per element) acceptable, or do you prefer a Löwdin projection?
- Q13. EOS window: our ±12 % in V (11 points) or Zhang et al.'s ±20 % (7-11 points)? ±12 % keeps the fit near the minimum; ±20 %
  matches their protocol for direct comparison with their tables.
- Q14. Ladder plateau criterion: plateau = the last two rungs within the F2/F3 criteria (moment 0.005 μB, 1 meV/atom) and
  successive differences shrinking as for PBE; drift = anything else, or a lower rippled state in (c). Acceptable, or a different
  threshold?
- Q15 (staged 2026-11-11, density level; equal peak modulation ε for every seed, two wavenumber bands 0.8 and 0.4 √ecutwfc instead of a separate gradient-form seed: Sec. 6.2). Ripple-seeded restart: amplitudes A (proposed: ripple factors with max |g − 1| = 10⁻³, 10⁻², 5·10⁻²) and wavenumber (the
  largest symmetric star with |k| ≤ 0.8 √ecutwfc, in bohr⁻¹ with ecutwfc in Ry); seed at the density level (proposed, needs only `qepy_set_tau`) or at the
  orbital level (needs write access to QE's wavefunctions)?
- Q16. If the ladder drifts: per-element cutoff definition of the model, or the nonlocal ∇²ln ρ smoothing first (your decision
  after the ladder results)?
