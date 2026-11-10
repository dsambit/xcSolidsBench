"""The SCF driver (DESIGN.md Sec. 5). One SCF per (case, functional); QEpy iterative mode; cold start; strict failures.

    loop: hook (rho, tau -> E_xc, v, v_tau -> extpot, extkin, extene) -> QE diagonalize -> QE mix (rho, tau) -> converged?

Converged = QE's conv_elec (estimated scf accuracy < conv_thr) and |Delta E_tot| < settings.E_TOL between the last two
steps. MAXITER without convergence, a negative spin density, a NaN or any QE error raise; the record is written first
with status "failed" and the history.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import traceback

import numpy as np

from .. import __version__
from ..functional import get as get_functional
from ..host.evaluate import evaluate_on_grid
from ..host.qe import Case, QESession
from ..parallel.mpi import CHUNK
from . import settings


class SCFError(RuntimeError):
    pass


def code_provenance():
    root = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", root, "status", "--porcelain", "--", "."], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    prov = {"xcsb": __version__, "xcsb_commit": commit, "xcsb_dirty": dirty, "settings": settings.as_dict()}
    info = os.path.join(os.environ.get("XCSB_BUILD", ""), "BUILD_INFO.json")
    if os.path.exists(info):
        prov["build"] = json.load(open(info))
    return prov


def _write(path, rec):
    tmp = path + ".tmp"
    json.dump(rec, open(tmp, "w"), indent=1, default=float)
    os.replace(tmp, path)


def run_native(case: Case, input_dft: str, workdir: str, comm, host_functional: str | None = None) -> dict:
    """QE's own functional (exttype 0, QE's SCF). Optionally the host's E_xc of the converged native density
    (`host_functional`, a libxc name) for the layout / unit check of V3."""
    t0 = time.time()
    s = QESession(case, input_dft, workdir, comm, iterative=False)
    s.driver.scf()
    conv = bool(s.driver.qepy_modules.control_flags.get_conv_elec())
    rec = {"mode": "native", "input_dft": input_dft, "case": case.as_dict(), "converged": conv,
           "energies_ry": s.energies(), "magnetization": s.magnetization(), "fermi_ry": s.fermi_level(),
           "provenance": code_provenance()}
    et, wk = s.eigenvalues()
    rec["eigenvalues_ry"], rec["kweights"] = et.tolist(), wk.tolist()
    if host_functional:
        rho, tau = s.spin_densities()
        res = evaluate_on_grid(get_functional(host_functional), s.grid, rho, tau, comm)
        if comm.Get_rank() == 0:
            rec["host_exc_ry_on_native_density"] = 2.0 * res.e_xc
            rec["host_stats"] = res.stats
    rec["seconds"] = time.time() - t0
    s.stop()
    if not conv:
        raise SCFError(f"native {input_dft} SCF did not converge")
    return rec


def run_hooked(case: Case, functional_name: str, workdir: str, comm, record_path: str | None = None,
               maxiter: int = settings.MAXITER, chunk: int = CHUNK) -> dict:
    t0 = time.time()
    rank = comm.Get_rank()
    f = get_functional(functional_name)
    s = QESession(case, settings.CARRIER_DFT, workdir, comm, iterative=True)
    rec = {"mode": "hooked", "functional": functional_name, "functional_provenance": f.provenance,
           "case": case.as_dict(), "provenance": code_provenance(), "history": [], "status": "running"}
    record_path = record_path or os.path.join(workdir, "record.json")
    e_prev = None
    try:
        for it in range(1, maxiter + 1):
            ti = time.time()
            rho, tau = s.spin_densities()
            res = evaluate_on_grid(f, s.grid, rho, tau, comm, chunk)
            th = time.time() - ti
            e_xc = res.e_xc if rank == 0 else None
            s.set_potentials(e_xc, res.v if rank == 0 else None, res.v_tau if rank == 0 else None)
            s.driver.diagonalize()
            s.driver.mix(mix_coef=case.mixing_beta)
            dr2 = float(s.driver.get_scf_error())
            conv = bool(s.driver.check_convergence())
            etot = float(s.driver.embed.etotal)
            h = {"iter": it, "etot_ry": etot, "dr2": dr2, "conv_elec": conv, "e_xc_ry": None if e_xc is None else 2 * e_xc,
                 "t_hook_s": th, "t_iter_s": time.time() - ti, **s.magnetization()}
            if rank == 0:
                h.update(min_rho=res.stats["min_rho"], frac_tau_below_tauW=res.stats["frac_points_tau_below_tauW"])
            rec["history"].append(h)
            if not np.isfinite(etot):
                raise SCFError(f"iteration {it}: E_tot = {etot}")
            if conv:
                de = None if e_prev is None else abs(etot - e_prev)
                if de is None or de > settings.E_TOL:
                    raise SCFError(f"conv_elec at iteration {it} but |dE| = {de} > E_TOL = {settings.E_TOL}")
                break
            e_prev = etot
            if rank == 0 and it % 5 == 0:
                _write(record_path, rec)
        else:
            raise SCFError(f"not converged in {maxiter} iterations (last dr2 {dr2:.3e})")
        rho, tau = s.spin_densities()
        res = evaluate_on_grid(f, s.grid, rho, tau, comm, chunk)
        et, wk = s.eigenvalues()
        rec.update(status="ok", iterations=it, energies_ry=s.energies(), magnetization=s.magnetization(),
                   fermi_ry=s.fermi_level(), eigenvalues_ry=et.tolist(), kweights=wk.tolist())
        if rank == 0:
            rec.update(e_xc_final_ry=2 * res.e_xc, density_stats=res.stats)
    except Exception as exc:
        rec.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        if rank == 0:
            _write(record_path, rec)
        raise
    finally:
        rec["seconds"] = time.time() - t0
    if rank == 0:
        _write(record_path, rec)
    s.stop()
    return rec
