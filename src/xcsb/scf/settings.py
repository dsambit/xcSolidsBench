"""Global numerical settings (DESIGN.md Sec. 5): the same for every system and functional, recorded in every result.
Per-system physical settings (ecutwfc, k-mesh, degauss) live in the case files."""
from __future__ import annotations

DUAL = 4                     # ecutrho = DUAL * ecutwfc (protocol; 8 only as the quadrature check of M1 step 1)
CONV_THR_BENCH = 1e-8        # Ry per cell, QE's estimated scf accuracy: every benchmark SCF, every system and functional
                             # (user 2026-11-10, M1 onward)
CONV_THR_VERIFY = 1e-10      # Ry per cell: the verification tiers (V3-V6), labelled "verification"
MAXITER = 200
MIXING_MODE = "plain"        # Broyden
MIXING_BETA = 0.3
MIXING_NDIM = 8
SMEARING = "gaussian"        # occupations in [0, 1] keep rho_sigma = sum f |psi|^2 >= 0; cold (Marzari-Vanderbilt) and
                             # Methfessel-Paxton smearing have negative occupations (M0: SG15 Fe gave rho_sigma < 0 with mv)
STARTINGWFC = "atomic+random"
CARRIER_DFT = "TPSS"         # QE-internal meta-GGA: makes QE compute, mix and apply tau; its XC is switched off (exttype 4)
STARTS = ("tpss-carrier", "pbe-scaffold")   # Case.start (DESIGN Sec. 5): the converged native carrier (every study up to
                             # 2026-11-13), or one XC-free diagonalization and the converged scaffold functional through the hooks
SCAFFOLD_FUNCTIONAL = "PBE"  # start 'pbe-scaffold': libxc PBE through the hooks (v_tau = 0), QE's own XC never applied
EXTTYPE_XC = 4               # QEpy: skip QE's XC; v_xc, v_tau and E_xc come from the hook


def as_dict():
    return {k: v for k, v in globals().items() if k.isupper()}
