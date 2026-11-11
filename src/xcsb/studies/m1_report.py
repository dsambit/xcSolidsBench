"""Plain-text tables of the M1 cutoff ladder (xcsb.studies.m1_ladder.table): tables only, no plots."""
from __future__ import annotations


def _f(x, fmt):
    return "-" if x is None else format(x, fmt)


def render(t) -> str:
    out = []
    out.append("# M1 step 1: cutoff ladder (bcc Fe, bcc V; 20^3 k, Gaussian 0.01 Ry)")
    out.append("")
    out.append("## Stage P: PBE scan (E_c^PBE criteria: moment 0.005 muB/atom, E(FM) - E(NM) 1 meV/atom between successive cutoffs)")
    for el, s in t["scan"].items():
        out.append(f"\n{el} (NM reference: {s['nm_state']})")
        out.append("| ecutwfc (Ry) | m_FM (muB) | m_NM (muB) | E_FM - E_NM (meV/atom) | it FM | it NM |")
        out.append("|---|---|---|---|---|---|")
        for r in s["rows"]:
            fm, nm = r["fm"] or {}, r["nm"] or {}
            out.append(f"| {r['ecutwfc']:g} | {_f(fm.get('m'), '.4f')} | {_f(nm.get('m'), '.4f')} | "
                       f"{_f(r.get('de_fm_nm_mev'), '.2f')} | {_f(fm.get('iterations'), 'd')} | {_f(nm.get('iterations'), 'd')} |")
    out.append(f"\nE_c^PBE: {t['ec_pbe']}")
    if not t["ec_pbe"]:
        return "\n".join(out)
    out.append("\n## Stage L: ladder, d = 4 (E per atom; Delta E = E(rung n) - E(rung n-1); concave fraction for each record lambda)")
    for key, rows in t["ladder"].items():
        out.append(f"\n{key}")
        out.append("| r | ecutwfc | E (Ry) | Delta E (meV) | m (muB) | Delta m | |m| | it (carrier + hook) | concave fraction |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            if "etot_ry" not in r:
                out.append(f"| {r['r']:g} | {r['ecutwfc']:g} | {r['status'] or 'missing'} | | | | | | |")
                continue
            cf = ", ".join(f"{v:.3e}" for v in (r.get("concave_fraction") or {}).values()) or "-"
            out.append(f"| {r['r']:g} | {r['ecutwfc']:g} | {r['etot_ry']:.8f} | {_f(r['de_prev_mev'], '.3f')} | {r['m']:.4f} | "
                       f"{_f(r['dm_prev'], '+.4f')} | {r['abs_m']:.4f} | {r['carrier_iterations']} + {r['iterations']} | {cf} |")
    out.append("\n## d = 8 minus d = 4 at the same ecutwfc")
    out.append("| state, functional | r | ecutwfc | Delta E (meV/atom) | Delta m (muB) |")
    out.append("|---|---|---|---|---|")
    for key, rows in t["d8"].items():
        for r in rows:
            out.append(f"| {key} | {r['r']:g} | {r['ecutwfc']:g} | {r['e_d8_minus_d4_mev']:.3f} | {r['m_d8_minus_d4']:+.4f} |")
    out.append("\n## V: E(FM start) - E(NM start)")
    out.append("| functional | r | ecutwfc | Delta E (meV/atom) | m (FM start) | m (NM start) |")
    out.append("|---|---|---|---|---|---|")
    for f, rows in t["v_fm_minus_nm_mev"].items():
        for r in rows:
            out.append(f"| {f} | {r['r']:g} | {r['ecutwfc']:g} | {r['de_mev']:.3f} | {r['m_fm_start']:.4f} | {r['m_nm_start']:.4f} |")
    out.append("\n## Ripple-seeded restarts at the top rung (ratio: seeded star component after / at seeding; 0 decayed, 1 persisted)")
    out.append("| state, functional | seed | status | it | E_after - E_before (meV) | Delta E_xc at seeding (meV) | m before | m after | "
               "ratio rho (up, dn) | ratio lap rho | ratio abs grad rho |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for key, rp in t["ripple"].items():
        for r in rp["rows"]:
            if r["status"] != "ok":
                out.append(f"| {key} | {r['seed']} | {r['status']} | | | | | | {r.get('error', '')} | | |")
                continue
            rt = {k: ", ".join(_f(x, ".3f") for x in v) for k, v in r["ratio"].items()}
            out.append(f"| {key} | {r['seed']} | ok | {r['iterations']} | {r['de_mev']:.4f} | {r['de_xc_seed_mev']:.3f} | "
                       f"{r['m_before']:.4f} | {r['m_after']:.4f} | {rt['rho']} | {rt['lapl_rho']} | {rt['abs_grad_rho']} |")
    return "\n".join(out)


def render_stage_p(t) -> str:
    """The stop report of the first allocation (phase 0 + stage P): EC.json, measured times, revised stage-L estimate."""
    out = ["# M1 phase 0 + stage P report", ""]
    out.append(f"E_c^PBE: {t['ec']['ec_pbe'] if t['ec'] else 'not determined (scan incomplete)'}")
    out.append("\n## Phase 0 (Fe 40 Ry, 8^3 k, 8 ranks)")
    out.append("| run | E (Ry) | m (muB) | hook it | QE s/it | hook s/it | s |")
    out.append("|---|---|---|---|---|---|---|")
    for k, r in t["phase0"].items():
        out.append(f"| {k} | {r['etot_ry']:.8f} | {r['m']:.4f} | {r['iterations']} | {r['qe_s_per_iteration']:.2f} | "
                   f"{r['hook_s_per_iteration']:.2f} | {r['seconds']:.0f} |")
    out.append("\n## Stage P SCF times (PBE, 20^3 k, pools: one rank per pool)")
    out.append("| task | status | ranks | npool | carrier it | hook it | QE s/it | hook s/it | s |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for r in t["scf_times"]:
        if "seconds" not in r:
            out.append(f"| {r['task']} | {r['status'] or 'missing'} | | | | | | | |")
            continue
        out.append(f"| {r['task']} | ok | {r['ranks']} | {r['npool']} | {r['carrier_iterations']} | {r['hook_iterations']} | "
                   f"{r['qe_s_per_iteration']:.2f} | {r['hook_s_per_iteration']:.2f} | {r['seconds']:.0f} |")
    c = t["cost"]
    out.append("\n## Cost model (measured: " + (", ".join(c.get("measured", [])) or "none") + "; the rest assumed)")
    out.append(", ".join(f"{k} = {v:.3g}" for k, v in c.items() if isinstance(v, (int, float))))
    e = t["stage_l_estimate"]
    if e:
        out.append(f"\n## Stage L estimate: {e['node_h_total']:.1f} node-h for {e['n_tasks']} tasks "
                   f"(seed tasks {e['node_h_seed_tasks']:.1f}, d = 8 {e['node_h_d8']:.1f}); longest step {e['longest_step_h']:.2f} h")
    return "\n".join(out)
