"""Global numerical settings (DESIGN.md Sec. 5): the same for every system and functional, recorded in every result.
Per-system physical settings (ecutwfc, k-mesh, degauss) live in the case files."""
from __future__ import annotations

DUAL = 4                     # ecutrho = DUAL * ecutwfc (protocol; 8 only as the quadrature check of M1 step 1)
CONV_THR = 1e-10             # Ry, QE's estimated scf accuracy per cell
MAXITER = 200
MIXING_MODE = "plain"        # Broyden
MIXING_BETA = 0.3
MIXING_NDIM = 8
SMEARING = "mv"              # Marzari-Vanderbilt cold smearing
STARTINGWFC = "atomic+random"
CARRIER_DFT = "TPSS"         # QE-internal meta-GGA: makes QE compute, mix and apply tau; its XC is switched off (exttype 4)
EXTTYPE_XC = 4               # QEpy: skip QE's XC; v_xc, v_tau and E_xc come from the hook


def as_dict():
    return {k: v for k, v in globals().items() if k.isupper()}
