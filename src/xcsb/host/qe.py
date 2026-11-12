"""The QE host through QEpy (DESIGN.md Sec. 4): pw.x input, the QEpy session, gathering rho/tau, setting
extpot / extkin / extene. Only this module (and scf.driver) imports QEpy.

Hooked runs use input_dft = settings.CARRIER_DFT (a meta-GGA, so that QE computes, mixes and applies tau) with
QEpy exttype 4 (QE's XC off). Native runs (verification V3) use QE's own functional (exttype 0).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from ..bench.structures import Structure
from ..grid.spectral import SpectralGrid
from ..pseudo import check_element
from . import layout
from ..scf import settings


@dataclass
class Case:
    structure: Structure
    ecutwfc: float                         # Ry
    kmesh: tuple = (8, 8, 8)
    kshift: tuple = (0, 0, 0)
    occupations: str = "smearing"          # "smearing" (cold smearing, settings.SMEARING) or "fixed"
    degauss: float = 0.01                  # Ry
    nspin: int = 1
    starting_magnetization: dict = field(default_factory=dict)   # label -> value in [-1, 1]
    tot_magnetization: float | None = None
    nosym: bool = False
    dual: int = settings.DUAL
    conv_class: str = "bench"              # "bench" (settings.CONV_THR_BENCH) or "verification" (CONV_THR_VERIFY)
    mixing_beta: float = settings.MIXING_BETA
    nbnd: int | None = None
    diago_full_acc: bool = False           # converge the empty bands as tightly as the occupied ones (band gaps)

    @property
    def conv_thr(self) -> float:
        return {"bench": settings.CONV_THR_BENCH, "verification": settings.CONV_THR_VERIFY}[self.conv_class]

    def __post_init__(self):
        if self.conv_class not in ("bench", "verification"):
            raise ValueError(f"conv_class {self.conv_class!r}: 'bench' or 'verification'")
        if self.nspin not in (1, 2):
            raise ValueError("nspin must be 1 or 2 (non-collinear magnetism is out of scope)")
        if self.nspin == 2 and not self.starting_magnetization and self.tot_magnetization is None:
            raise ValueError("nspin = 2 needs starting_magnetization (an explicit 0.0 counts) or tot_magnetization")
        unknown = set(self.starting_magnetization) - set(self.structure.species)
        if unknown:
            raise ValueError(f"starting_magnetization for unknown species labels {sorted(unknown)}")

    def as_dict(self):
        s = self.structure
        return {"structure": {"name": s.name, "lattice_bohr": s.lattice.tolist(), "labels": list(s.labels),
                              "frac": s.frac.tolist(), "elements": s.elements},
                "ecutwfc": self.ecutwfc, "ecutrho": self.dual * self.ecutwfc, "kmesh": list(self.kmesh),
                "kshift": list(self.kshift), "occupations": self.occupations, "smearing": settings.SMEARING,
                "degauss": self.degauss, "nspin": self.nspin, "starting_magnetization": dict(self.starting_magnetization),
                "tot_magnetization": self.tot_magnetization, "nosym": self.nosym, "conv_thr": self.conv_thr,
                "conv_class": self.conv_class,
                "mixing_beta": self.mixing_beta, "nbnd": self.nbnd, "diago_full_acc": self.diago_full_acc}


def pw_input(case: Case, input_dft: str, pseudo_dir: str, outdir: str, prefix: str, maxiter: int = settings.MAXITER,
             restart: str = "") -> str:
    """The pw.x input text. No forces, no stress (not provided by the host)."""
    s = case.structure
    for el in sorted(set(s.elements.values())):
        check_element(el, pseudo_dir)
    from ..pseudo import manifest
    man = manifest()
    sysl = ["  ibrav = 0", f"  nat = {s.nat}", f"  ntyp = {len(s.species)}",
            f"  ecutwfc = {case.ecutwfc:.6f}", f"  ecutrho = {case.dual * case.ecutwfc:.6f}",
            f"  input_dft = '{input_dft}'", f"  nspin = {case.nspin}", f"  occupations = '{case.occupations}'"]
    if case.occupations == "smearing":
        sysl += [f"  smearing = '{settings.SMEARING}'", f"  degauss = {case.degauss:.8f}"]
    for i, lab in enumerate(s.species, 1):
        if lab in case.starting_magnetization:
            sysl.append(f"  starting_magnetization({i}) = {case.starting_magnetization[lab]:.6f}")
    if case.tot_magnetization is not None:
        sysl.append(f"  tot_magnetization = {case.tot_magnetization:.6f}")
    if case.nosym:
        sysl.append("  nosym = .true.")
    if case.nbnd:
        sysl.append(f"  nbnd = {case.nbnd}")
    lines = ["&CONTROL", "  calculation = 'scf'", f"  prefix = '{prefix}'", f"  outdir = '{outdir}'",
             f"  pseudo_dir = '{pseudo_dir}'", "  tprnfor = .false.", "  tstress = .false.", "  disk_io = 'low'", "/",
             "&SYSTEM", *sysl, "/",
             "&ELECTRONS", f"  conv_thr = {case.conv_thr:.3e}", f"  mixing_mode = '{settings.MIXING_MODE}'",
             f"  mixing_beta = {case.mixing_beta:.6f}", f"  mixing_ndim = {settings.MIXING_NDIM}",
             f"  electron_maxstep = {maxiter}",
             *(["  diago_full_acc = .true."] if case.diago_full_acc else []),
             f"  startingwfc = '{'file' if restart else settings.STARTINGWFC}'",
             f"  startingpot = '{'file' if restart == 'files' else 'atomic'}'", "/",
             "ATOMIC_SPECIES"]
    for lab in s.species:
        lines.append(f"  {lab} 1.0 {man['files'][s.elements[lab]]['file']}")     # the mass is unused in an SCF
    lines.append("CELL_PARAMETERS bohr")
    lines += ["  " + " ".join(f"{x:.12f}" for x in row) for row in s.lattice]
    lines.append("ATOMIC_POSITIONS crystal")
    lines += [f"  {lab} " + " ".join(f"{x:.12f}" for x in fr) for lab, fr in zip(s.labels, s.frac)]
    lines.append("K_POINTS automatic")
    lines.append("  " + " ".join(str(int(k)) for k in case.kmesh) + " " + " ".join(str(int(k)) for k in case.kshift))
    return "\n".join(lines) + "\n"


META_DFTS = {"TPSS", "SCAN", "R2SCAN", "M06L", "TB09"}    # QE input_dft names of the meta-GGAs this host may use natively


class QESession:
    """One QEpy Driver (only one can exist per process) and the host-side grid bookkeeping."""

    def __init__(self, case: Case, input_dft: str, workdir: str, comm, iterative: bool, pseudo_dir: str | None = None,
                 outdir: str | None = None, restart: str = "", npool: int = 1):
        import qepy
        from qepy.driver import Driver
        if restart not in ("", "files", "wavefunctions"):
            raise ValueError(f"restart {restart!r}: '' (cold), 'files' (wavefunctions, rho, tau) or 'wavefunctions'")
        self.case, self.comm, self.input_dft = case, comm, input_dft
        if npool < 1 or comm.Get_size() % npool:
            raise ValueError(f"npool {npool} must divide the rank count {comm.Get_size()}")
        self.npool = int(npool)
        # QE's k-point pools (-nk): QEpy reads QE's command line from qepy_sys.command_line (not from the Python argv)
        qepy.qepy_modules.qepy_sys.set_command_line(f"-nk {self.npool}")
        self._meta = input_dft.upper() in META_DFTS
        self.rank = comm.Get_rank()
        pseudo_dir = pseudo_dir or os.environ["XCSB_PSEUDO_DIR"]
        os.makedirs(workdir, exist_ok=True)
        self.inputfile = os.path.join(workdir, "pw.in")
        if self.rank == 0:
            open(self.inputfile, "w").write(pw_input(case, input_dft, pseudo_dir, outdir or os.path.join(workdir, "out"),
                                                     "xcsb", restart=restart))
        comm.Barrier()
        self.driver = Driver(inputfile=self.inputfile, comm=comm, iterative=iterative,
                             logfile=os.path.join(workdir, "pw.out") if self.rank == 0 else None)
        d = self.driver
        self.nspin = int(d.get_number_of_spins())
        if self.nspin != case.nspin:
            raise RuntimeError(f"QE nspin {self.nspin} != case nspin {case.nspin}")
        self.shape = tuple(int(n) for n in d.get_number_of_grid_points(gather=True))
        self.lattice = np.asarray(d.get_ions_lattice(), dtype=float)
        self.ecutrho = float(d.get_ecutrho())
        self.grid = None
        if self.rank == 0:
            self.grid = SpectralGrid(self.lattice, self.shape, self.ecutrho)
            ngm_g = int(d.qepy_modules.gvect.get_ngm_g())
            if self.grid.ngm != ngm_g:
                raise RuntimeError(f"sphere bookkeeping: host counts {self.grid.ngm} G vectors, QE ngm_g = {ngm_g}")

    # -- QE -> contract ---------------------------------------------------------------------------------------------
    def spin_densities(self, need_tau: bool = True):
        """(rho, tau) as (2, n1, n2, n3) Hartree arrays on the root (None elsewhere). Collective.
        need_tau = False (a native non-meta run feeding a tau-free functional): tau is returned as zeros and never read."""
        rho_qe = self.driver.get_density(gather=True)
        if need_tau:
            if not self._meta:
                raise RuntimeError(f"tau requested but input_dft {self.input_dft!r} is not a meta-GGA: QE has no tau")
            kin_qe = self.driver.get_kinetic_energy_density(gather=True)
        if self.rank != 0:
            return None, None
        rho_qe = np.asarray(rho_qe)
        kin_qe = np.asarray(kin_qe) if need_tau else np.zeros_like(rho_qe)
        return layout.spin_channels(self.nspin, rho_qe, kin_qe, self.shape)

    # -- contract -> QE ---------------------------------------------------------------------------------------------
    def set_potentials(self, e_xc_ha, v, v_tau):
        """extpot (+ exttype 4, extene) and extkin from root-side results. Collective."""
        d = self.driver
        if self.rank == 0:
            extpot, extkin, extene = layout.qe_potentials(self.nspin, e_xc_ha, v, v_tau)
        else:
            extpot = d.create_array(gather=True)
            extkin = d.create_array(gather=True)
            extene = None
        extene = self.comm.bcast(extene, root=0)
        d.set_external_potential(extpot, exttype=settings.EXTTYPE_XC, extene=extene, gather=True)
        if self.comm.Get_size() > 1:
            d.qepy_pw.qepy_mod._mp_bcast_group_real_2(extkin)
        d.qepy_pw.qepy_mod.qepy_set_extkin(extkin, gather=True)
        return extene

    def set_spin_densities(self, rho, tau):
        """Overwrite QE's density and kinetic energy density (rho%of_r/of_g, rho%kin_r/kin_g) with root-side contract arrays
        (2, n1, n2, n3) Hartree; then read them back and raise unless QE holds exactly what was set. Collective.
        Used only by the ripple-seeded restart (xcsb.studies.m1_ladder); QE copies rho into the mixer's input (rhoin) at the
        start of the next diagonalization, so the Broyden history pairs the seeded input with its own output."""
        d = self.driver
        if self.rank == 0:
            rq, kq = layout.qe_spin_arrays(self.nspin, rho, tau)
        else:
            rq = d.create_array(gather=True)
            kq = d.create_array(gather=True)
        d.set_density(rq, gather=True)                       # group broadcast to the pool roots, then scatter (QEpy)
        if self.comm.Get_size() > 1:
            d.qepy_pw.qepy_mod._mp_bcast_group_real_2(kq)
        d.qepy_pw.qepy_mod.qepy_set_tau(kq, gather=True)
        r2, t2 = self.spin_densities()
        bad = None
        if self.rank == 0:
            dr = float(np.max(np.abs(r2 - rho)) / np.max(np.abs(rho)))
            dt = float(np.max(np.abs(t2 - tau)) / np.max(np.abs(tau)))
            if not (dr <= 1e-12 and dt <= 1e-12):
                bad = f"set_spin_densities: read-back mismatch (rel. rho {dr:.2e}, tau {dt:.2e})"
        bad = self.comm.bcast(bad, root=0)
        if bad:
            raise RuntimeError(bad)

    # -- results ----------------------------------------------------------------------------------------------------
    def energies(self):
        e = self.driver.qepy_pw.ener
        out = {k: float(getattr(e, "get_" + k)()) for k in ("etot", "eband", "deband", "ehart", "etxc", "ewld", "demet")}
        out["embed_etotal"] = float(self.driver.embed.etotal)
        return out

    def magnetization(self):
        if self.nspin == 1:
            return {"total": 0.0, "absolute": 0.0}
        lm = self.driver.qepy_pw.lsda_mod
        return {"total": float(lm.get_magtot()), "absolute": float(lm.get_absmag())}

    def eigenvalues(self):
        """et (nbnd, nks) in Ry and the k weights (QE's local k list: one pool only)."""
        if self.npool != 1:
            raise RuntimeError("eigenvalues: QE's k list is pool-local; only with npool = 1")
        w = self.driver.qepy_pw.wvfct
        return np.asarray(w.get_array_et()).copy(), np.asarray(self.driver.qepy_pw.klist.get_array_wk()).copy()

    def density_from_orbitals(self, et_ry, kweights):
        """rho, tau := sum_band of QE's current orbitals (startingwfc 'file') with occupations from the given eigenvalues
        (nbnd, nkstot) Ry, the source run's: QE does not read eigenvalues with the wavefunctions. The k list must be the
        source's (checked through the weights). Occupations in [0, 1] (Gaussian smearing; tot_magnetization: two Fermi levels),
        so rho_s >= 0 and tau_s >= tau_W,s by construction. One pool only."""
        if self.npool != 1:
            raise RuntimeError("density_from_orbitals: one pool only")
        w = self.driver.qepy_pw.wvfct
        et = w.get_array_et()
        et_ry = np.asarray(et_ry, dtype=float)
        n = et_ry.shape[1]                                 # nkstot; QE's wk array has the fixed length npk
        kweights = np.asarray(kweights, dtype=float)[:n]
        wk = np.asarray(self.driver.qepy_pw.klist.get_array_wk())[:n]
        if et.shape != et_ry.shape or kweights.shape != wk.shape or not np.allclose(wk, kweights, rtol=0, atol=1e-12):
            raise RuntimeError(f"density_from_orbitals: QE et {et.shape} / k weights differ from the source {et_ry.shape}")
        w.set_array_et(et_ry)
        if not np.array_equal(w.get_array_et(), et_ry):
            raise RuntimeError("density_from_orbitals: setting QE's eigenvalues failed")
        self.driver.qepy_pw.qepy_mod.qepy_calc_density()

    def kpoints_frac(self):
        """The irreducible k points in crystal coordinates (fractions of the reciprocal vectors), (nks, 3); one pool only."""
        if self.npool != 1:
            raise RuntimeError("kpoints_frac: QE's k list is pool-local; only with npool = 1")
        return np.asarray(self.driver.get_ibz_k_points(), dtype=float).copy()

    def fermi_level(self):
        return float(self.driver.get_fermi_level())

    def stop(self):
        self.driver.stop()
