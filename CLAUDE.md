# xcSolidsBench

Solid-state test bench for XC functionals of (ρ_σ, ∇ρ_σ, τ_σ, ∇²ρ_σ, ∇∇ρ_σ) in Quantum ESPRESSO 7.6 via QEpy (plane waves,
SG15 ONCV, no NLCC), collinear spin. Design: DESIGN.md (authoritative; update it with every design change).

## Binding rules
- Workspace rules: /Users/dsambit/Workspace/CLAUDE.md (shared Python env `/Users/dsambit/Workspace/pytools/bin/py`).
- Project rules shared with the xcHoleModelMeta line, where they apply to a host code: the strongxc-v2 rules
  (`/Users/dsambit/Workspace/xcHoleModelMeta-strongxc-v2/rules/{implementation,technical}.md`): reference data from the archive,
  write-once; reference-functional results computed once and banked; potentials checked against finite differences; derivative
  pre-check before benchmark lanes; SCF robustness from cold starts; no grid data on the laptop; failures raise.
- CLEAN ROOM (user 2026-11-10): written from the physics and the public QE / QEpy sources and docs only. Never open, read, copy
  or adapt code from the collaborator's fork (github.com/argha004/xcHoleModelMetaFork) or from any "scanfam" repository.
- `xchole` (/Users/dsambit/Workspace/xcHoleModelMeta-strongxc-v3, branch strongxc-v3): read-only use of its public API; never modify it.
- Principled and minimal: one clear design, not a menu; no branching, fallbacks, clips or floors in numerics; failures raise.
- Magnetic moments and oxide properties are validation / hold-out data: never in an objective, never used to choose parameters.
- Never cite a paper that has not been read; literature and reference data go to the archive (refdata/), write-once.

## Laptop and Perlmutter
- Laptop: light work only (seconds-long checks); no grid data; only JSON/CSV summaries come back.
- Perlmutter: $W=/pscratch/sd/d/dsambit/xc_functional_dev, write only inside $W; code reaches it only via git
  (bare remote `perlmutter` = $W/xcSolidsBench.git); heavy work on compute nodes only.
- Standing permission: 1-node interactive allocations ≤ 4 h on account m2360, at most 2 interactive jobs in total across ALL of
  the user's workstreams (check squeue first); report every launch. Batch jobs, more nodes or longer allocations need the user's
  explicit permission.

## Git
- Branch main only; no new branches (incl. an `archive` branch) without the user's permission.
- Commit by pathspec: `git commit -m "..." -- <paths>`, message ending with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Remotes: `origin` (private GitHub github.com/dsambit/xcSolidsBench), `perlmutter` (bare repo on scratch); the laptop is authoritative.
