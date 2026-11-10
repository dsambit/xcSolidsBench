# xcSolidsBench

A modular solid-state test bench for exchange-correlation functionals of (ρ_σ, ∇ρ_σ, τ_σ, ∇²ρ_σ) and optionally the density
Hessian, run self-consistently (generalized Kohn-Sham, collinear spin) in Quantum ESPRESSO 7.6 through QEpy, with norm-conserving
SG15 ONCV pseudopotentials.

Status: phase 0, design only. See [DESIGN.md](DESIGN.md) for the architecture, the verification tiers, the benchmark families
(first milestone: magnetism of bcc Fe and bcc V; then LC20 lattice constants and bulk moduli, ferromagnetic and paramagnetic
metals, antiferromagnetic transition-metal monoxides, the Zhang et al. 64-solid set), the reference-data plan, the Perlmutter
build plan and the open questions.

Functionals: libxc (PBE, r2SCAN, SCAN, TPSS) and the SCH-SI model sets of the `xchole` package (SCH-SI-A:v3.1, SCH-SI-B:v3.1).
