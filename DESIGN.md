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
start: QE potinit (atomic ρ, Thomas-Fermi τ; starting_magnetization per species) + wfcinit
loop i = 1..MAXITER:
    hook:        ρ_σ, τ_σ <- QE;  e, d <- functional;  v_σ, v_τ,σ <- spectral;  set extpot, extkin, extene (exttype 4)
    diagonalize: QE rebuilds v (Hartree + extpot; v%kin_r <- extkin), c_bands, sum_band -> ρ_out, τ_out
    mix:         QE Broyden on (ρ, τ) -> ρ_in
    converged if QE's conv_elec (estimated scf accuracy < conv_thr) and |ΔE_tot| < E_TOL over the last 3 steps
end: final hook on the converged density; record
failure (MAXITER, negative ρ_σ, NaN, any QE error): raise; the record is written with status "failed" and the history
```

- Global settings, the same for every system and functional (`scf/settings.py`, recorded in every result): conv_thr, E_TOL,
  MAXITER, mixing (Broyden 'plain'; beta and ndim fixed by the M1 convergence study), d = 4, Marzari-Vanderbilt cold smearing for
  every system (one rule: for gapped systems the smearing is inert once degauss ≪ gap, and it handles the cases where a functional
  closes a gap, e.g. Ge or the monoxides). Per-system physical settings (ecutwfc, k-mesh, degauss) come from the convergence
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
  k-mesh and cold smearing held at a generous fixed setting during the ladder (e.g. 20³, degauss 0.01 Ry; converged in step 2).
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
  criteria; cold smearing as everywhere (FeO and CoO may come out metallic with semilocal functionals).
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

## 7. Verification tiers

| tier | what | where | pass |
|---|---|---|---|
| V1 | spectral derivatives vs closed forms on band-limited periodic test densities (finite Fourier series inside the sphere; triclinic, bcc, fcc, hcp and AFM-II cells), both spin channels; ∇²ρ = tr H; adjointness ⟨D_i f, g⟩ = −⟨f, D_i g⟩ and symmetry of the second-order operators; Σ M = ngm_g; adapters: libxc vs finite differences, xchole vs the handoff sanity points, zero-padding points | laptop, seconds | 1e−12 relative (derivatives, adjoints); 1e−10 (sanity points) |
| V2a | potential vs finite differences of the grid energy: E_grid(ρ ± εδρ_σ) and E_grid(τ ± εδτ_σ) along random sphere-limited directions, **each spin channel separately** (δρ_↑ only, δρ_↓ only, δτ_↑, δτ_↓) and in the nspin = 1 and nspin = 2 layouts, on polarized test densities (ζ up to ±0.9); test functional, PBE, r2SCAN, SCH-SI-A, -B | laptop, seconds | central differences at the O(ε²) level, ≤ 1e−7 relative |
| V2b | orbital-level GKS check: a numpy plane-wave toy (a few orbitals per spin, several k) with ρ_σ, τ_σ from orbitals; FD of E_xc along occupied-virtual rotations within each spin channel vs 2 Re⟨δψ∣Ĥ_xc,σ∣ψ⟩, Ĥ_xc,σ ψ = v_σψ − ½∇·(v_τ,σ∇ψ) (Ha); validates the ½ in τ and the τ operator per spin | laptop, seconds | ≤ 1e−7 relative |
| V3 | the hooks reproduce QE native: libxc PBE vs `input_dft='PBE'`, libxc TPSS vs `'TPSS'`; (i) bulk Si, nspin = 1, fixed occupations; (ii) **bcc Fe, nspin = 2, cold smearing, FM**; and the host's E_xc of the native ground states vs QE's etxc in both cases | node | ΔE_tot ≤ 1e−5 Ry/atom, eigenvalues ≤ 1e−5 Ry, moment ≤ 1e−4 μB (limited by QE's internal density thresholds and QE vs libxc implementation details; the measured differences are recorded) |
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
| M1 step 1, cutoff ladder | 1 atom; E_c^PBE ~100 Ry assumed; up to ×3: 300 Ry, grid 32³ → ~56³, ~1.5·10³ → ~8·10³ PW; 20³ k | 2-7 min at the low rungs, ~10-40 min at ×3 (QE cost ∝ PW count; SCH-SI ~3 s/step at 56³) | 8 rungs × 4 functionals × 3 states (Fe FM, V NM, V FM) ≈ 100 SCFs + d = 8 checks + ripple restarts: ≈ 8-15 node-h (several allocations, or batch, Q7) |
| M1 steps 2-3, Fe, V | 1 atom; ecutwfc from step 1; 20³-24³ k (≈ 250-450 irreducible) | 2-5 s/step, 40-80 steps: 2-7 min at ~100 Ry | k/smearing study (mostly PBE) + 4 functionals × 2 metals × 3 runs: ≈ 2 node-h |
| F1 LC20 | 1-2 atoms; 32³-48³, alkali metals up to ~60³ | 2-15 min (SCH-SI dominates on the large grids) | 11 volumes × 20 solids × 4 functionals ≈ 900 SCFs, packed 4 lanes per node: ≈ 10-60 node-h, plus convergence |
| F2 Fe, Co, Ni | 1-2 atoms | 2-10 min | moments + E(V) scans × 4 functionals: ≈ 5-10 node-h |
| F3 V, Pd | 1 atom | 2-7 min | 3 starts × E(V) scans × 4 functionals: ≈ 3-6 node-h |
| F4 FeO, CoO, NiO | 4 atoms (2 f.u.); ~120 Ry (O and TM semicore); ~72³ (~4·10⁵ points); ~5·10³ PW; 6³-8³ k | ~10 s/step, 80-150 steps: 15-25 min; the `nosym` run several times longer | 3 runs × 3 oxides × 4 functionals + convergence: ≈ 15-30 node-h |

M1 steps 2-3 fit one allocation; the step-1 ladder needs several (its top rungs dominate). F1, F4 and the full F2/F3 scans exceed what a few 4 h interactive allocations can do;
batch jobs would need the user's permission (Q7).

---

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
- Q10. Fixed lattices (M1, F2-F4 moments): low-temperature experimental values with or without zero-point correction? Proposed:
  zero-point-corrected where the source provides it, the uncorrected value reported alongside.
- Q11. "Sol62": Zhang et al. (NJP 2018) has 64 solids. Which 62 do you mean (which two are dropped, and why)?
- Q12. F4: is QE's sphere-integrated site moment (fixed radius per element) acceptable, or do you prefer a Löwdin projection?
- Q13. EOS window: our ±12 % in V (11 points) or Zhang et al.'s ±20 % (7-11 points)? ±12 % keeps the fit near the minimum; ±20 %
  matches their protocol for direct comparison with their tables.
- Q14. Ladder plateau criterion: plateau = the last two rungs within the F2/F3 criteria (moment 0.005 μB, 1 meV/atom) and
  successive differences shrinking as for PBE; drift = anything else, or a lower rippled state in (c). Acceptable, or a different
  threshold?
- Q15. Ripple-seeded restart: amplitudes A (proposed: ripple factors with max |g − 1| = 10⁻³, 10⁻², 5·10⁻²) and wavenumber (the
  largest symmetric star with |k| ≤ 0.8 √ecutwfc, in bohr⁻¹ with ecutwfc in Ry); seed at the density level (proposed, needs only `qepy_set_tau`) or at the
  orbital level (needs write access to QE's wavefunctions)?
- Q16. If the ladder drifts: per-element cutoff definition of the model, or the nonlocal ∇²ln ρ smoothing first (your decision
  after the ladder results)?
