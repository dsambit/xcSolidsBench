"""The SCF driver (DESIGN.md Sec. 5). One SCF per (case, functional); QEpy iterative mode; cold start; strict failures.

    loop: hook (rho, tau -> E_xc, v, v_tau -> extpot, extkin, extene) -> QE diagonalize -> QE mix (rho, tau) -> converged?

Converged = QE's conv_elec (estimated scf accuracy < conv_thr), QE's own criterion; |Delta E_tot| of the last step is
recorded (QEpy's iterative energy lags the density by one step, so it is not a convergence measure). MAXITER without
convergence, a negative spin density, a NaN or any QE error raise; the record is written first
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


def qe_meta_threshold_part(f, grid, rho, tau, nspin):
    """QE's native meta-GGA drivers set e_xc = 0 at every point where rho, |grad rho|^2 or tau is at or below a fixed
    threshold (XClib qe_drivers_mgga.f90: rho_threshold_mgga 1e-12, grho2_threshold_mgga 1e-24, tau_threshold_mgga 1e-12;
    nspin 2: on rho_up + rho_dn, 4 (|grad rho_up|^2 + |grad rho_dn|^2), tau_up + tau_dn). Grid points at centres of symmetry
    have grad rho = 0 exactly. The host E_xc (Ry) at those points, i.e. the expected host - QE difference."""
    from ..host.evaluate import point_inputs
    x, grad = point_inputs(grid, rho, tau, False, False)
    e = f.evaluate(x).extra["e"]
    rt, tt = x.rho.sum(0), x.tau.sum(0)
    if nspin == 1:
        g2 = np.sum(grad.sum(0) ** 2, axis=0).reshape(-1)
    else:
        g2 = 4.0 * (np.sum(grad[0] ** 2, axis=0) + np.sum(grad[1] ** 2, axis=0)).reshape(-1)
    mask = (np.abs(rt) <= 1e-12) | (g2 <= 1e-24) | (np.abs(tt) <= 1e-12)
    return {"npoints": int(np.count_nonzero(mask)), "host_exc_ry_at_points": 2.0 * grid.weight * float(np.sum(e[mask]))}


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
        hf = get_functional(host_functional)
        rho, tau = s.spin_densities(need_tau=bool(getattr(hf, "is_meta", True)))
        res = evaluate_on_grid(hf, s.grid, rho, tau, comm)
        if comm.Get_rank() == 0:
            rec["host_exc_ry_on_native_density"] = 2.0 * res.e_xc
            rec["host_stats"] = res.stats
            if getattr(hf, "is_meta", False):
                rec["qe_meta_threshold"] = qe_meta_threshold_part(hf, s.grid, rho, tau, case.nspin)
    rec["seconds"] = time.time() - t0
    s.stop()
    if not conv:
        raise SCFError(f"native {input_dft} SCF did not converge")
    return rec


def run_hooked(case: Case, functional_name: str, workdir: str, comm, record_path: str | None = None,
               maxiter: int = settings.MAXITER, chunk: int = CHUNK, post=None, deviations: dict | None = None,
               npool: int = 1, seeds=None, warm_start: dict | None = None, previous_attempts: list | None = None) -> dict:
    """post: optional diagnostic called on the root with (grid, rho, tau) of the converged density; its dict is stored
    as rec["post"] (e.g. xcsb.studies.fe_payback.attribution).
    seeds: optional ripple-seeded restarts after convergence (xcsb.studies.m1_ladder.SeedPlan): each seed overwrites QE's
    (rho, tau) with the seeded fields built from the converged ones and re-converges in the same session (QE's orbitals as
    the start, the Broyden history kept); results in rec["ripple"]. The record of the unseeded SCF is complete before.
    warm_start: optional {"from": a converged run directory, "label": text}: a production restart (user-approved per point, never
    a robustness test): that run's QE outdir (wavefunctions, rho, tau) is copied and read by QE (startingwfc and startingpot
    'file'), and the hook starts directly from it (no carrier phase). previous_attempts: earlier failed records of this point,
    kept in rec["previous_attempts"]."""
    t0 = time.time()
    rank = comm.Get_rank()
    f = get_functional(functional_name)
    # Start (the same for every functional): the carrier meta-GGA (QE's own TPSS, exttype 0) is converged first in the
    # SAME QEpy session and loop (diagonalize, mix), from QE's cold start; the hook then takes over from that density.
    # An SCF-converged density is (up to the mixing residual) an output density sum f |psi|^2 >= 0 with occupations in
    # [0, 1] (Gaussian smearing), tau >= tau_W (DESIGN 4.5). Found in M0: the atomic superposition is not positive (SG15
    # Fe: 0.85 e of negative charge); the bare starting orbitals' density is far from self-consistency (Fe: dr2 ~ 7e2 Ry);
    # a save/restart through QE's files changed the density (Fe: 0.054 e negative after reading back).
    warm = None
    if warm_start is not None:
        import shutil
        src_rec = json.load(open(os.path.join(warm_start["from"], "record.json")))
        if src_rec.get("status") != "ok":
            raise SCFError(f"warm start: {warm_start['from']} is not a converged run ({src_rec.get('status')})")
        if rank == 0:
            dst = os.path.join(workdir, "out")
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(os.path.join(warm_start["from"], "out"), dst)
        comm.Barrier()
        warm = {"label": warm_start["label"], "from": os.path.abspath(warm_start["from"]),
                "source": {"etot_ry": src_rec["energies_ry"]["etot"], "magnetization": src_rec["magnetization"],
                           "case": src_rec["case"], "xcsb_commit": src_rec["provenance"].get("xcsb_commit")}}
    s = QESession(case, settings.CARRIER_DFT, workdir, comm, iterative=True, npool=npool, restart=warm is not None)
    if warm is None:
        carrier = {"history": []}
        tc = time.time()
        for it in range(1, maxiter + 1):
            s.driver.diagonalize()
            s.driver.mix(mix_coef=case.mixing_beta)
            dr2 = float(s.driver.get_scf_error())
            conv = bool(s.driver.qepy_modules.control_flags.get_conv_elec())
            carrier["history"].append({"iter": it, "dr2": dr2, **s.magnetization()})
            if conv:
                break
        else:
            raise SCFError(f"start: the carrier {settings.CARRIER_DFT} did not converge in {maxiter} iterations")
        carrier.update(iterations=it, magnetization=s.magnetization(), seconds=time.time() - tc)
        # QEpy marks an exttype-0 run as finished at convergence (rho is then the output density); the hook continues it
        s.driver.embed.finish = False
        start = f"converged {settings.CARRIER_DFT} (QE native, exttype 0) in the same session, then the hook"
    else:
        carrier = {"history": [], "iterations": 0, "skipped": "warm start", "magnetization": s.magnetization()}
        start = warm["label"]
    rec = {"mode": "hooked", "functional": functional_name, "functional_provenance": f.provenance,
           "case": case.as_dict(), "provenance": code_provenance(), "history": [], "status": "running",
           "start": start, "carrier": carrier, "maxiter": maxiter, "deviations": deviations or {},
           "parallel": {"ranks": comm.Get_size(), "npool": npool, "chunk": chunk}}
    if warm is not None:
        rec["warm_start"] = warm
    if previous_attempts:
        rec["previous_attempts"] = previous_attempts
    record_path = record_path or os.path.join(workdir, "record.json")

    def hook_loop(history, label):
        """Hook iterations until QE's conv_elec (read from the flag; QEpy's end_scf runs once, at the very end)."""
        e_prev = None
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
            conv = bool(s.driver.qepy_modules.control_flags.get_conv_elec())
            etot = float(s.driver.embed.etotal)
            h = {"iter": it, "etot_ry": etot, "dr2": dr2, "conv_elec": conv, "e_xc_ry": None if e_xc is None else 2 * e_xc,
                 "t_hook_s": th, "t_iter_s": time.time() - ti, **s.magnetization()}
            if rank == 0:
                h.update(min_rho=res.stats["min_rho"], frac_tau_below_tauW=res.stats["frac_points_tau_below_tauW"])
            history.append(h)
            if not np.isfinite(etot):
                raise SCFError(f"{label} iteration {it}: E_tot = {etot}")
            h["d_etot_ry"] = None if e_prev is None else etot - e_prev
            if conv:
                return it
            e_prev = etot
            if rank == 0 and it % 5 == 0:
                _write(record_path, rec)
        raise SCFError(f"{label}: not converged in {maxiter} iterations (last dr2 {dr2:.3e})")

    try:
        it = hook_loop(rec["history"], "hook")
        rho, tau = s.spin_densities()
        res = evaluate_on_grid(f, s.grid, rho, tau, comm, chunk)
        rec.update(status="ok", iterations=it, energies_ry=s.energies(), magnetization=s.magnetization(),
                   fermi_ry=s.fermi_level())
        if npool == 1:
            et, wk = s.eigenvalues()
            rec.update(eigenvalues_ry=et.tolist(), kweights=wk.tolist(), kpoints_frac=s.kpoints_frac().tolist())
        if rank == 0:
            rec.update(e_xc_final_ry=2 * res.e_xc, density_stats=res.stats)
            if post is not None:                         # a diagnostic: its failure is recorded, the SCF result stands
                try:
                    rec["post"] = post(s.grid, rho, tau)
                except Exception as exc:
                    rec["post"] = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        rec["seconds"] = time.time() - t0
        if seeds is not None:
            if rank == 0:
                _write(record_path, rec)
            rec["ripple"] = {"plan": seeds.describe(s.grid) if rank == 0 else None, "seeds": []}
            base = {"rho": rho, "tau": tau, "e_xc_ry": rec.get("e_xc_final_ry"), "etot_ry": rec["energies_ry"]["etot"],
                    "magnetization": rec["magnetization"]}
            for spec in seeds.specs():
                ts = time.time()
                out = {"seed": spec, "history": []}
                rec["ripple"]["seeds"].append(out)
                try:
                    r1 = t1 = err = None
                    if rank == 0:
                        try:
                            r1, t1 = seeds.apply(s.grid, rho, tau, spec)
                        except Exception as exc:
                            err = f"{type(exc).__name__}: {exc}"
                    err = comm.bcast(err, root=0)
                    if err:
                        raise SCFError(f"seed {spec['label']}: {err}")
                    seeded = evaluate_on_grid(f, s.grid, r1, t1, comm, chunk)        # collective; E_xc of the seeded fields
                    if rank == 0:
                        out["delta_e_xc_seed_ry"] = 2 * seeded.e_xc - base["e_xc_ry"]
                        out["seeded_density_stats"] = seeded.stats
                    s.set_spin_densities(r1, t1)
                    out["iterations"] = hook_loop(out["history"], f"seed {spec['label']}")
                    r2, t2 = s.spin_densities()
                    out.update(status="ok", etot_before_ry=base["etot_ry"], etot_after_ry=float(s.energies()["etot"]),
                               magnetization_before=base["magnetization"], magnetization_after=s.magnetization())
                    out["delta_etot_ry"] = out["etot_after_ry"] - out["etot_before_ry"]
                    if rank == 0:
                        out["amplitudes"] = seeds.amplitudes(s.grid, (rho, tau), (r1, t1), (r2, t2), spec["band"])
                except Exception as exc:                 # one seed's failure is recorded; the next seed starts from the base
                    out.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
                out["seconds"] = time.time() - ts
                if rank == 0:
                    _write(record_path, rec)
    except Exception as exc:
        rec.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        if rank == 0:
            _write(record_path, rec)
        raise
    finally:
        rec.setdefault("seconds", time.time() - t0)            # the unseeded SCF (set before the seeds when it converged)
        rec["seconds_total"] = time.time() - t0
    if rank == 0:
        _write(record_path, rec)
    s.driver.end_scf()
    s.stop()
    return rec
