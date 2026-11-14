"""Data-driven SUGGESTIONS for three assumptions in config.yaml: Rw, the Passey baseline interval
and the matrix density. Nothing here changes config.yaml; the user decides.

Works on the pipeline output (curves.csv + zone_summary.json), i.e. on QC-cleaned logs.

1. Rw from apparent water resistivity
       Rwa = RT * PHIT^m / a
   In clean, porous, water-bearing rock Rwa equals Rw. Hydrocarbons raise RT and so raise Rwa;
   they never lower it. The low end of the Rwa distribution in clean porous rock is therefore the
   best estimate of Rw: the suggestion is P10, and P50 / P90 are shown so the spread is visible.

2. Passey baseline
   The baseline must be a NON-SOURCE shale in good hole: there DlogR = 0 by definition.
   Organic shale is more resistive (and slower) than lean shale at the same depth, so candidate
   shale intervals in good hole are ranked by LOWEST median resistivity.

3. Matrix density from the density-neutron crossplot (liquid-filled, clean rock)
       phi   = NPHI + nphi_matrix_offset     (neutron porosity in the units of the matrix)
       rhoma = (RHOB - phi * rho_fluid) / (1 - phi)
   Gas lowers the apparent matrix density, residual shale raises it, and a limestone-calibrated
   neutron in sandstone reads slightly low (so rhoma reads slightly low): read it as a guide.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MINERALS = {"quartz (sandstone)": 2.65, "calcite (limestone)": 2.71,
            "dolomite": 2.87, "anhydrite": 2.98}

DEFAULTS = {
    "rw_vsh_max": 0.3, "rw_phi_min": 0.15, "rw_min_thickness": 1.0,
    "rw_min_samples": 100, "rw_hydrocarbon_factor": 3.0,
    "shale_igr_min": 0.5, "shale_nd_sep_min": 0.10, "shale_min_thickness": 3.0, "shale_max_badhole": 0.1,
    "ma_vsh_max": 0.2, "ma_min_samples": 20,
}
PRINT_TOP = 10                       # ranked baseline candidates / Rw intervals printed (all are in the result)
# neighbouring shale windows are merged into one baseline candidate only if they differ by less than this
SIMILAR = {"log10_rt": 0.1, "DT": 5.0, "RHOB": 0.03}


# --------------------------------------------------------------------------- helpers
def _pct(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if not len(x):
        return {"p10": None, "p50": None, "p90": None}
    p10, p50, p90 = np.percentile(x, [10, 50, 90])
    return {"p10": float(p10), "p50": float(p50), "p90": float(p90)}


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as (i0, i1) with i1 exclusive."""
    m = np.asarray(mask, dtype=bool)
    edges = np.diff(np.r_[0, m.astype(int), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _step(depth: np.ndarray) -> float:
    d = np.diff(depth)
    return float(np.median(d)) if len(d) else 0.0


def _thickness(depth: np.ndarray, i0: int, i1: int, step: float) -> float:
    return float(depth[i1 - 1] - depth[i0]) + step


def _zone_at(df: pd.DataFrame, i0: int, i1: int):
    if "ZONE" not in df.columns:
        return None
    z = df["ZONE"].iloc[i0:i1].dropna()
    return str(z.mode().iloc[0]) if len(z) else None


def _badhole(df: pd.DataFrame) -> np.ndarray:
    if "QC_BADHOLE" in df.columns:
        return df["QC_BADHOLE"].fillna(False).astype(bool).to_numpy()
    return np.zeros(len(df), dtype=bool)


def nearest_mineral(rho: float) -> str:
    return min(MINERALS, key=lambda k: abs(MINERALS[k] - rho))


# --------------------------------------------------------------------------- 1. Rw
def suggest_rw(df: pd.DataFrame, a: float, m: float, vsh_max: float = DEFAULTS["rw_vsh_max"],
               phi_min: float = DEFAULTS["rw_phi_min"],
               min_thickness: float = DEFAULTS["rw_min_thickness"],
               min_samples: int = DEFAULTS["rw_min_samples"],
               hc_factor: float = DEFAULTS["rw_hydrocarbon_factor"]) -> dict:
    depth = df["DEPTH"].to_numpy(dtype=float)
    step = _step(depth)
    rt, phi, vsh = (df[c].to_numpy(dtype=float) for c in ("RT", "PHIT", "VSH"))
    with np.errstate(invalid="ignore"):
        ok = (vsh <= vsh_max) & (phi >= phi_min) & (rt > 0) & ~_badhole(df)
        rwa = rt * phi ** m / a

    intervals, keep = [], np.zeros(len(df), dtype=bool)
    for i0, i1 in runs(ok):
        thk = _thickness(depth, i0, i1, step)
        if thk < min_thickness:
            continue
        keep[i0:i1] = True
        intervals.append({"top": float(depth[i0]), "base": float(depth[i1 - 1]), "thickness": thk,
                          "n_samples": int(i1 - i0), "zone": _zone_at(df, i0, i1),
                          "rwa_p50": float(np.median(rwa[i0:i1])),
                          "phit_p50": float(np.median(phi[i0:i1])),
                          "vsh_p50": float(np.median(vsh[i0:i1]))})
    stats = _pct(rwa[keep])
    suggested = stats["p10"]

    # Per zone, so the water / hydrocarbon contrast is visible instead of hidden in one percentile.
    by_zone = []
    if "ZONE" in df.columns:
        zones = df["ZONE"].astype("string").fillna("-").to_numpy()
        for z in pd.unique(zones[keep]):
            sel = keep & (zones == z)
            st = _pct(rwa[sel])
            by_zone.append({"zone": str(z), "n_samples": int(sel.sum()), "rwa": st,
                            "likely_hydrocarbon": bool(suggested and st["p50"] > hc_factor * suggested)})

    # P10 is only Rw if some of the selected rock is water-bearing. Say when that is doubtful.
    warnings = []
    if keep.sum() and keep.sum() < min_samples:
        warnings.append(f"only {int(keep.sum())} samples pass the filter (fewer than {min_samples}): weak statistics")
    if len({i['zone'] for i in intervals}) == 1 and intervals:
        warnings.append(f"all samples are in one zone ({intervals[0]['zone']}); if that zone holds hydrocarbons, "
                        "P10 is NOT Rw")
    if by_zone and not any(not z["likely_hydrocarbon"] for z in by_zone if z["n_samples"] >= 20):
        warnings.append("no zone with 20+ samples has Rwa near P10: no clear water-bearing interval")
    if suggested is not None and stats["p90"] / suggested > 10:
        warnings.append(f"Rwa P90/P10 = {stats['p90'] / suggested:.0f}: mix of water- and hydrocarbon-bearing rock "
                        "(expected); check the zones marked HC")
    return {"criteria": {"vsh_max": vsh_max, "phit_min": phi_min, "min_thickness": min_thickness,
                         "a": a, "m": m, "good_hole_only": True, "hydrocarbon_if_rwa_p50_above": hc_factor},
            "n_samples": int(keep.sum()), "rwa": stats, "suggested_rw": suggested,
            "by_zone": by_zone, "intervals": intervals, "warnings": warnings}


# --------------------------------------------------------------------------- 2. Passey baseline
def passey_candidates(df: pd.DataFrame, method: str = "sonic",
                      igr_min: float = DEFAULTS["shale_igr_min"],
                      nd_sep_min: float = DEFAULTS["shale_nd_sep_min"],
                      min_thickness: float = DEFAULTS["shale_min_thickness"],
                      max_badhole: float = DEFAULTS["shale_max_badhole"]) -> dict:
    curve = {"sonic": "DT", "density": "RHOB"}[method]
    depth = df["DEPTH"].to_numpy(dtype=float)
    step = _step(depth)
    # Shale is picked on the LINEAR gamma-ray index: organic shale is usually the hottest GR and sets
    # gr_shale, so a lean shale can sit at IGR 0.5-0.6, which Larionov squeezes to Vsh 0.3-0.4.
    shale_col = "IGR" if "IGR" in df.columns else "VSH"
    rt, x, igr = (df[c].to_numpy(dtype=float) for c in ("RT", curve, shale_col))
    bad = _badhole(df)
    # GR alone also lets hot (feldspar / mica) sandstone through. Shale shows neutron-density
    # separation (NPHI well above PHID); clean sand does not. Where density was nulled for bad hole
    # the separation is unknown: keep those samples so the washed-out shale shows up as rejected.
    use_sep = "PHID" in df.columns and "NPHI" in df.columns
    with np.errstate(invalid="ignore"):
        shale = (igr >= igr_min) & (rt > 0) & np.isfinite(x)
        if use_sep:
            sep = df["NPHI"].to_numpy(dtype=float) - df["PHID"].to_numpy(dtype=float)
            shale &= (sep >= nd_sep_min) | (np.isnan(sep) & bad)
    facies = df["FACIES"].astype("string").to_numpy() if "FACIES" in df.columns else None

    # A shale run can hold lean shale, organic shale and washouts side by side. Cut each run into
    # windows of min_thickness and merge neighbours only while they look alike, so one candidate
    # never averages a lean shale with an organic shale or with bad hole.
    tol_x = SIMILAR[curve]
    win = max(2, int(round(min_thickness / step))) if step > 0 else 2
    cands = []
    for r0, r1 in runs(shale):
        if _thickness(depth, r0, r1, step) < min_thickness:
            continue
        edges = list(range(r0, r1, win))
        if len(edges) > 1 and r1 - edges[-1] < win:          # fold a short tail into the last window
            edges.pop()
        windows = [(a, b) for a, b in zip(edges, edges[1:] + [r1])]
        groups: list[list[int]] = []
        for a, b in windows:
            stats = (np.log10(np.median(rt[a:b])), np.median(x[a:b]), bad[a:b].mean() > max_badhole)
            if groups:
                g = groups[-1]
                ga, gb = g[0], g[1]
                prev = (np.log10(np.median(rt[ga:gb])), np.median(x[ga:gb]), bad[ga:gb].mean() > max_badhole)
                if (abs(stats[0] - prev[0]) <= SIMILAR["log10_rt"] and abs(stats[1] - prev[1]) <= tol_x
                        and stats[2] == prev[2]):
                    g[1] = b
                    continue
            groups.append([a, b])
        for i0, i1 in groups:
            bh = float(bad[i0:i1].mean())
            cands.append({"top": float(depth[i0]), "base": float(depth[i1 - 1]),
                          "thickness": _thickness(depth, i0, i1, step),
                          "n_samples": int(i1 - i0), "zone": _zone_at(df, i0, i1),
                          "badhole_fraction": bh, "usable": bh <= max_badhole,
                          "rt_p50": float(np.median(rt[i0:i1])),
                          f"{curve.lower()}_p50": float(np.median(x[i0:i1])),
                          "facies": (pd.Series(facies[i0:i1]).dropna().mode().iloc[0]
                                     if facies is not None and pd.Series(facies[i0:i1]).notna().any() else None)})
    usable = sorted((c for c in cands if c["usable"]), key=lambda c: c["rt_p50"])
    for rank, c in enumerate(usable, start=1):
        c["rank"] = rank
    return {"method": method, "curve": curve,
            "criteria": {"shale_curve": shale_col, "igr_min": igr_min,
                         "nd_separation_min": nd_sep_min if use_sep else None,
                         "min_thickness": min_thickness, "max_badhole_fraction": max_badhole},
            "candidates": sorted(cands, key=lambda c: c.get("rank", 10**6)),
            "suggested": usable[0] if usable else None, "shale_mask": shale}


def check_interval(df: pd.DataFrame, interval, curve: str) -> dict | None:
    """How the baseline interval currently in use looks: bad-hole share and medians."""
    if not interval:
        return None
    top, base = map(float, interval)
    sel = (df["DEPTH"] >= top) & (df["DEPTH"] <= base)
    if not sel.any():
        return {"top": top, "base": base, "n_samples": 0}
    sub = df[sel]
    return {"top": top, "base": base, "n_samples": int(sel.sum()),
            "badhole_fraction": float(_badhole(sub).mean()),
            "vsh_p50": float(sub["VSH"].median()), "rt_p50": float(sub["RT"].median()),
            f"{curve.lower()}_p50": float(sub[curve].median())}


def plot_passey(df: pd.DataFrame, result: dict, path, current=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = result["curve"]
    shale = result["shale_mask"]
    d = df["DEPTH"].to_numpy(dtype=float)
    rt = df["RT"].to_numpy(dtype=float)
    x = df[curve].to_numpy(dtype=float)
    unit = "us/ft" if curve == "DT" else "g/cc"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 8), gridspec_kw={"width_ratios": [1.2, 1]})
    sc = ax1.scatter(x[shale], rt[shale], c=d[shale], s=4, cmap="viridis_r", alpha=0.7)
    ax1.set_yscale("log")
    ax1.set_xlabel(f"{curve} [{unit}]")
    ax1.set_ylabel("RT [ohm.m]")
    crit = result["criteria"]
    sep = f", NPHI-PHID >= {crit['nd_separation_min']}" if crit["nd_separation_min"] is not None else ""
    ax1.set_title(f"Shale samples ({crit['shale_curve']} >= {crit['igr_min']}{sep}): RT vs {curve}\n"
                  "lean shale = low-RT trend; organic shale plots ABOVE it", fontsize=10)
    fig.colorbar(sc, ax=ax1, label="depth")
    for c in result["candidates"][:5]:
        if "rank" not in c:
            continue
        ax1.plot(c[f"{curve.lower()}_p50"], c["rt_p50"], "r*", ms=14, mec="k")
        ax1.annotate(f"#{c['rank']} {c['top']:.1f}-{c['base']:.1f}", (c[f"{curve.lower()}_p50"], c["rt_p50"]),
                     xytext=(6, 6), textcoords="offset points", fontsize=8, color="darkred")
    if curve == "DT":
        ax1.invert_xaxis()

    # Passey overlay: log10(RT) and the porosity curve scaled so 1 decade = 50 us/ft (or 0.4 g/cc)
    best = result["suggested"]
    lrt = np.log10(np.where(rt > 0, rt, np.nan))
    if best:
        ref_rt, ref_x = np.log10(best["rt_p50"]), best[f"{curve.lower()}_p50"]
    else:
        ref_rt, ref_x = np.nanmedian(lrt[shale]), np.nanmedian(x[shale])
    scaled = ref_rt + (0.02 * (x - ref_x) if curve == "DT" else -2.5 * (x - ref_x))
    ax2.plot(lrt, d, color="red", lw=0.6, label="log10 RT")
    ax2.plot(scaled, d, color="purple", lw=0.6,
             label=f"{curve} scaled ({'50 us/ft' if curve == 'DT' else '0.4 g/cc'} per decade)")
    ax2.fill_betweenx(d, ax2.get_xlim()[0], ax2.get_xlim()[1], where=shale, color="grey", alpha=0.12, lw=0)
    for c in result["candidates"]:
        if "rank" in c and c["rank"] <= 5:
            ax2.axhspan(c["top"], c["base"], color="gold", alpha=0.4, lw=0)
            ax2.text(ax2.get_xlim()[1], c["top"], f" #{c['rank']}", va="top", fontsize=8)
        elif not c["usable"]:
            ax2.axhspan(c["top"], c["base"], color="red", alpha=0.12, lw=0)
    if current:
        ax2.axhspan(current[0], current[1], fill=False, ec="blue", lw=1.5, ls="--", label="current baseline_interval")
    ax2.invert_yaxis()
    ax2.set_xlabel("log10 ohm.m (curves overlay where DlogR = 0)")
    ax2.set_ylabel("depth")
    ax2.set_title("Overlay referenced to candidate #1\ngold = ranked candidates, red = rejected (bad hole), grey = shale",
                  fontsize=10)
    ax2.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- 3. matrix density
def suggest_matrix_density(df: pd.DataFrame, rho_fluid: float, nphi_offset: float = 0.0,
                           vsh_max: float = DEFAULTS["ma_vsh_max"],
                           min_samples: int = DEFAULTS["ma_min_samples"]) -> dict:
    rhob = df["RHOB"].to_numpy(dtype=float)
    phi = df["NPHI"].to_numpy(dtype=float) + nphi_offset
    vsh = df["VSH"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = (vsh <= vsh_max) & np.isfinite(rhob) & (phi >= 0) & (phi < 0.45) & ~_badhole(df)
        rhoma = (rhob - phi * rho_fluid) / (1 - phi)
    zones = df["ZONE"].astype("string").fillna("-").to_numpy() if "ZONE" in df.columns else np.full(len(df), "all")
    out = []
    for z in pd.unique(zones):
        sel = ok & (zones == z)
        n = int(sel.sum())
        row = {"zone": str(z), "n_clean_samples": n, "clean_fraction": float(sel.sum() / max((zones == z).sum(), 1))}
        if n >= min_samples:
            st = _pct(rhoma[sel])
            row.update({"rhoma": st, "suggested_rho_matrix": round(st["p50"], 2),
                        "nearest_mineral": nearest_mineral(st["p50"])})
        else:
            row.update({"rhoma": None, "suggested_rho_matrix": None, "nearest_mineral": None,
                        "note": f"fewer than {min_samples} clean good-hole samples"})
        out.append(row)
    return {"criteria": {"vsh_max": vsh_max, "rho_fluid": rho_fluid, "nphi_matrix_offset": nphi_offset,
                         "good_hole_only": True, "min_samples": min_samples},
            "zones": out}


# --------------------------------------------------------------------------- report
def _f(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def format_suggestions(rw: dict, passey: dict, current_baseline: dict | None, ma: dict, used: dict,
                       plot_path=None) -> str:
    L = ["=" * 78, "PARAMETER SUGGESTIONS  (printed only; config.yaml is NOT changed)", "=" * 78]

    c = rw["criteria"]
    L.append(f"\n1. Rw FROM Rwa = RT * PHIT^m / a   (a = {c['a']}, m = {c['m']})")
    L.append(f"   samples: Vsh <= {c['vsh_max']}, PHIT >= {c['phit_min']}, good hole, runs >= {c['min_thickness']} m")
    if rw["n_samples"]:
        s = rw["rwa"]
        L.append(f"   {rw['n_samples']} samples   Rwa P10 {_f(s['p10'], 4)}   P50 {_f(s['p50'], 4)}   "
                 f"P90 {_f(s['p90'], 4)} ohm.m")
        L.append(f"   SUGGESTED Rw = {_f(rw['suggested_rw'], 4)} ohm.m (P10: hydrocarbons only raise Rwa)"
                 f"   current sw.rw = {used['rw']}")
        for w in rw["warnings"]:
            L.append(f"   WARNING: {w}")
        if rw["by_zone"]:
            L.append(f"   per zone  {'n':>5} {'P10':>8} {'P50':>8} {'P90':>8}   "
                     f"(HC = Rwa P50 > {c['hydrocarbon_if_rwa_p50_above']:g} x suggested Rw)")
            for z in rw["by_zone"]:
                r = z["rwa"]
                L.append(f"   {z['zone']:>8} {z['n_samples']:>5} {r['p10']:>8.4f} {r['p50']:>8.4f} {r['p90']:>8.4f}"
                         f"   {'HC' if z['likely_hydrocarbon'] else 'water?'}")
        top = sorted(rw["intervals"], key=lambda i: -i["thickness"])[:PRINT_TOP]
        L.append(f"   thickest {len(top)} of {len(rw['intervals'])} intervals:")
        L.append(f"   {'top':>9} {'base':>9} {'thk':>7} {'zone':>5} {'n':>5} {'Rwa P50':>9} {'PHIT':>6} {'Vsh':>5}")
        for i in sorted(top, key=lambda i: i["top"]):
            L.append(f"   {i['top']:>9.2f} {i['base']:>9.2f} {i['thickness']:>7.2f} {str(i['zone']):>5} "
                     f"{i['n_samples']:>5} {i['rwa_p50']:>9.4f} {i['phit_p50']:>6.3f} {i['vsh_p50']:>5.2f}")
        L.append("   Rwa = Rw only in water-bearing rock. PHIT depends on rho_matrix, so check section 3 first.")
    else:
        L.append("   no clean, porous, good-hole interval found: no Rw suggestion (use a water sample)")

    cur = passey["curve"]
    k = f"{cur.lower()}_p50"
    c = passey["criteria"]
    L.append(f"\n2. PASSEY BASELINE  (method {passey['method']}: RT vs {cur})")
    sep = f", NPHI - PHID >= {c['nd_separation_min']}" if c["nd_separation_min"] is not None else ""
    L.append(f"   shale: {c['shale_curve']} >= {c['igr_min']}{sep}, runs >= {c['min_thickness']} m, "
             f"bad hole <= {c['max_badhole_fraction']:.0%}")
    L.append("   ranked by LOWEST median RT (lean, non-source shale); prefer one close in depth to the source interval")
    ranked = [x for x in passey["candidates"] if "rank" in x]
    rejected = [x for x in passey["candidates"] if "rank" not in x]
    L.append(f"   {'rank':>4} {'top':>9} {'base':>9} {'thk':>7} {'zone':>5} {'RT P50':>8} {cur + ' P50':>8} "
             f"{'badhole':>8}  facies")
    for x in ranked[:PRINT_TOP]:
        L.append(f"   {x['rank']:>4} {x['top']:>9.2f} {x['base']:>9.2f} {x['thickness']:>7.2f} {str(x['zone']):>5} "
                 f"{x['rt_p50']:>8.3f} {x[k]:>8.3f} {x['badhole_fraction']:>8.0%}  {x['facies'] or '-'}")
    if len(ranked) > PRINT_TOP:
        L.append(f"   ... {len(ranked) - PRINT_TOP} more ranked candidates (on the plot / in the result)")
    if rejected:
        L.append(f"   {len(rejected)} shale interval(s) rejected for bad hole, "
                 f"{sum(x['thickness'] for x in rejected):.1f} m in total")
    if passey["suggested"]:
        b = passey["suggested"]
        L.append(f"   SUGGESTED toc.baseline_interval = [{b['top']:.1f}, {b['base']:.1f}]   (check it on the plot)")
    else:
        L.append("   no usable shale interval in good hole: no suggestion")
    if current_baseline:
        cb = current_baseline
        if cb.get("n_samples"):
            L.append(f"   current baseline_interval [{cb['top']}, {cb['base']}]: {cb['n_samples']} samples, "
                     f"bad hole {cb['badhole_fraction']:.0%}, Vsh P50 {cb['vsh_p50']:.2f}, RT P50 {cb['rt_p50']:.3f}")
            if cb["badhole_fraction"] > c["max_badhole_fraction"]:
                L.append("   WARNING: the current baseline interval is mostly bad hole")
        else:
            L.append(f"   current baseline_interval [{cb['top']}, {cb['base']}] is outside the processed interval")
    if plot_path:
        L.append(f"   plot: {plot_path}")

    c = ma["criteria"]
    L.append(f"\n3. MATRIX DENSITY FROM DENSITY-NEUTRON  rhoma = (RHOB - phi*{c['rho_fluid']}) / (1 - phi),"
             f" phi = NPHI + {c['nphi_matrix_offset']}")
    L.append(f"   clean rock: Vsh <= {c['vsh_max']}, good hole, liquid-filled assumed   "
             f"current porosity.rho_matrix = {used['rho_matrix']}")
    L.append(f"   {'zone':>5} {'n':>6} {'clean':>6} {'P10':>6} {'P50':>6} {'P90':>6}  suggestion")
    for z in ma["zones"]:
        if z["rhoma"]:
            r = z["rhoma"]
            L.append(f"   {z['zone']:>5} {z['n_clean_samples']:>6} {z['clean_fraction']:>6.0%} {r['p10']:>6.3f} "
                     f"{r['p50']:>6.3f} {r['p90']:>6.3f}  {z['suggested_rho_matrix']:.2f} ({z['nearest_mineral']})")
        else:
            L.append(f"   {z['zone']:>5} {z['n_clean_samples']:>6} {z['clean_fraction']:>6.0%} {'-':>6} {'-':>6} {'-':>6}  "
                     f"none: {z['note']}")
    L.append("   Gas lowers rhoma; residual shale raises it; a limestone-scaled neutron in sandstone reads it low.")
    L.append("=" * 78)
    return "\n".join(L)


def run_suggestions(curves: pd.DataFrame, summary: dict, method: str | None = None, plot_path=None) -> tuple[dict, str]:
    """Everything the script prints. Reads only; writes the plot if plot_path is given."""
    p = summary["parameters"]
    method = method or ("density" if p["toc"]["method"] == "density" else "sonic")
    rw = suggest_rw(curves, p["sw"]["a"], p["sw"]["m"])
    passey = passey_candidates(curves, method)
    current = check_interval(curves, p["toc"].get("baseline_interval"), passey["curve"])
    ma = suggest_matrix_density(curves, p["porosity"]["rho_fluid"], p["porosity"]["nphi_matrix_offset"])
    if plot_path:
        plot_passey(curves, passey, plot_path, p["toc"].get("baseline_interval"))
    used = {"rw": p["sw"]["rw"], "rho_matrix": p["porosity"]["rho_matrix"]}
    text = format_suggestions(rw, passey, current, ma, used, plot_path)
    passey = {k: v for k, v in passey.items() if k != "shale_mask"}
    return {"rw": rw, "passey": passey, "current_baseline": current, "matrix_density": ma}, text
