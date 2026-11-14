"""Compare log-derived TOC and porosity with core, and say which Passey change would remove the TOC bias.

Nothing here changes config.yaml: the script prints and writes calibration_report.json + a crossplot.

Depth matching
    log_depth = core_depth + depth_shift       (shift in the LAS depth unit; + moves core deeper)
    each core sample takes the NEAREST log sample within `tolerance`; otherwise it is unmatched.
    Matches in bad hole are reported but left out of the statistics by default.

Statistics (core = reference, residual = log - core)
    bias = mean(log - core)     RMSE = sqrt(mean((log - core)^2))
    R2   = 1 - sum((log - core)^2) / sum((core - mean(core))^2)     (can be negative: worse than the core mean)

TOC: what would remove the bias (same Passey formula and masking as the pipeline)
    TOC = DlogR * 10^(2.297 - 0.1688 * LOM), 0 where DlogR < 0 or Vsh < toc.min_vsh
    * LOM scales TOC (fixes a PROPORTIONAL error).  Solved so that bias = 0.
    * The baseline shifts DlogR by a constant D (fixes an OFFSET):
          new rt_baseline = rt_baseline * 10^(-D)          (porosity-log baseline kept), or
          new porosity baseline = old - D / c             (RT baseline kept)
      with c = +0.02 (sonic, us/ft), -2.50 (density, g/cc), +4.0 (neutron, v/v).  Solved so that bias = 0.
    Both are printed with the RMSE they would give, so the better fit is visible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .petrophysics import toc_passey

PASSEY_COEF = {"sonic": ("DT", "dt_baseline", 0.02, "us/ft"),
               "density": ("RHOB", "rhob_baseline", -2.50, "g/cc"),
               "neutron": ("NPHI", "nphi_baseline", 4.0, "v/v")}
LOM_RANGE = (2.0, 16.0)              # search range for the LOM that removes the bias
DLOGR_SHIFT_RANGE = (-3.0, 3.0)      # search range for the DlogR offset
LOM_PLAUSIBLE = (6.0, 12.0)          # outside this the LOM is compensating for something else
DEFAULT_TOLERANCE = 0.3              # depth units


# --------------------------------------------------------------------------- core input
def read_core(path) -> pd.DataFrame:
    core = pd.read_csv(path)
    core.columns = [c.strip().upper() for c in core.columns]
    if "DEPTH" not in core.columns:
        raise ValueError("core CSV needs a DEPTH column")
    for c in ("TOC_WTPCT", "PHI"):
        if c not in core.columns:
            core[c] = np.nan
        core[c] = pd.to_numeric(core[c], errors="coerce")
    core["DEPTH"] = pd.to_numeric(core["DEPTH"], errors="coerce")
    bad = core["DEPTH"].isna()
    if bad.any():
        core = core[~bad]
    if core["PHI"].dropna().gt(1.0).any():
        raise ValueError("core PHI must be a fraction (v/v); values above 1 look like percent")
    warnings = []
    if bad.any():
        warnings.append(f"{int(bad.sum())} row(s) without a numeric DEPTH dropped")
    for c in ("TOC_WTPCT", "PHI"):
        neg = core[c] < 0
        if neg.any():
            warnings.append(f"{int(neg.sum())} negative {c} value(s) treated as missing at depths "
                            f"{core.loc[neg, 'DEPTH'].round(2).tolist()[:10]}")
            core.loc[neg, c] = np.nan
    core = core.reset_index(drop=True)
    core.attrs["warnings"] = warnings
    return core


def match_depths(core: pd.DataFrame, curves: pd.DataFrame, depth_shift: float = 0.0,
                 tolerance: float = DEFAULT_TOLERANCE) -> pd.DataFrame:
    """One row per core sample: core values, shifted depth, matched log depth + log values."""
    d_log = curves["DEPTH"].to_numpy(dtype=float)
    shifted = core["DEPTH"].to_numpy(dtype=float) + depth_shift
    idx = np.clip(np.searchsorted(d_log, shifted), 1, len(d_log) - 1)
    left, right = idx - 1, idx
    nearest = np.where(np.abs(d_log[left] - shifted) <= np.abs(d_log[right] - shifted), left, right)
    dist = np.abs(d_log[nearest] - shifted)
    matched = dist <= tolerance

    out = pd.DataFrame({"CORE_DEPTH": core["DEPTH"].to_numpy(), "SHIFTED_DEPTH": shifted,
                        "CORE_TOC": core["TOC_WTPCT"].to_numpy(), "CORE_PHI": core["PHI"].to_numpy(),
                        "MATCHED": matched})
    cols = [c for c in ("DEPTH", "TOC", "DLOGR", "VSH", "PHIT", "PHIE", "QC_BADHOLE", "ZONE") if c in curves.columns]
    logs = curves[cols].iloc[nearest].reset_index(drop=True)
    logs = logs.rename(columns={"DEPTH": "LOG_DEPTH"})
    logs = logs.where(pd.Series(matched, index=logs.index), axis=0)     # unmatched rows -> missing
    out = pd.concat([out, logs], axis=1)
    out["QC_BADHOLE"] = out["QC_BADHOLE"].astype("boolean").fillna(False).astype(bool) \
        if "QC_BADHOLE" in out.columns else False
    return out


# --------------------------------------------------------------------------- statistics
def stats(log: np.ndarray, core: np.ndarray) -> dict:
    log, core = np.asarray(log, dtype=float), np.asarray(core, dtype=float)
    ok = np.isfinite(log) & np.isfinite(core)
    n = int(ok.sum())
    if n == 0:
        return {"n": 0, "bias": None, "rmse": None, "r2": None, "core_mean": None, "log_mean": None}
    res = log[ok] - core[ok]
    ss_tot = float(np.sum((core[ok] - core[ok].mean()) ** 2))
    return {"n": n, "bias": float(res.mean()), "rmse": float(np.sqrt(np.mean(res ** 2))),
            "r2": (1.0 - float(np.sum(res ** 2)) / ss_tot) if n >= 2 and ss_tot > 0 else None,
            "core_mean": float(core[ok].mean()), "log_mean": float(log[ok].mean())}


def _toc_from_dlogr(dlogr, vsh, lom: float, min_vsh: float) -> np.ndarray:
    """Same as petrophysics.stage_toc, from the stored DLOGR."""
    toc = toc_passey(dlogr, lom)
    vsh = np.asarray(vsh, dtype=float)
    with np.errstate(invalid="ignore"):
        toc = np.where(vsh < min_vsh, 0.0, toc)
    return np.where(np.isnan(vsh), np.nan, toc)


def _solve(f, lo: float, hi: float, increasing: bool, iters: int = 80):
    """Root of a monotonic f on [lo, hi] by bisection; None if f has no sign change there."""
    flo, fhi = f(lo), f(hi)
    if flo is None or fhi is None or np.sign(flo) == np.sign(fhi) and flo != 0 and fhi != 0:
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm > 0) == increasing:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def toc_adjustments(m: pd.DataFrame, toc_params: dict) -> dict:
    """LOM and baseline changes that make the mean TOC bias zero on the matched good-hole samples."""
    method = toc_params["method"]
    curve, base_key, coef, unit = PASSEY_COEF[method]
    lom0, min_vsh = float(toc_params["lom"]), float(toc_params["min_vsh"])
    ok = m["CORE_TOC"].notna() & m["DLOGR"].notna() & m["VSH"].notna()
    dl, vsh, core = (m.loc[ok, c].to_numpy(dtype=float) for c in ("DLOGR", "VSH", "CORE_TOC"))
    out = {"method": method, "n": int(ok.sum()), "current": {"lom": lom0, "rt_baseline": toc_params["rt_baseline"],
                                                              base_key: toc_params.get(base_key)}}
    if not ok.any():
        out["note"] = "no matched good-hole core TOC with a log DlogR"
        return out
    n_masked = int((vsh < min_vsh).sum())
    if n_masked:
        out["note_masked"] = (f"{n_masked} core sample(s) sit where Vsh < toc.min_vsh = {min_vsh}: the log TOC is 0 "
                              "there whatever the LOM or baseline, so they keep their bias")

    def bias_lom(lom):
        return float(np.nanmean(_toc_from_dlogr(dl, vsh, lom, min_vsh) - core))

    def bias_shift(d):
        return float(np.nanmean(_toc_from_dlogr(dl + d, vsh, lom0, min_vsh) - core))

    def rmse(toc):
        return float(np.sqrt(np.nanmean((toc - core) ** 2)))

    # LOM: higher LOM -> lower TOC, so bias decreases with LOM
    lom = _solve(bias_lom, *LOM_RANGE, increasing=False)
    if lom is None:
        out["lom"] = {"solvable": False, "reason": f"no LOM in {LOM_RANGE} gives zero bias "
                      "(log TOC is 0 at the core points, or the error is not proportional)"}
    else:
        new = _toc_from_dlogr(dl, vsh, lom, min_vsh)
        out["lom"] = {"solvable": True, "lom": lom, "change": lom - lom0, "rmse_after": rmse(new),
                      "r2_after": stats(new, core)["r2"],
                      "plausible": LOM_PLAUSIBLE[0] <= lom <= LOM_PLAUSIBLE[1]}

    # baseline: a DlogR offset D; higher D -> higher TOC
    d = _solve(bias_shift, *DLOGR_SHIFT_RANGE, increasing=True)
    if d is None:
        out["baseline"] = {"solvable": False, "reason": f"no DlogR offset in {DLOGR_SHIFT_RANGE} gives zero bias"}
    else:
        new = _toc_from_dlogr(dl + d, vsh, lom0, min_vsh)
        b0 = toc_params.get(base_key)
        out["baseline"] = {"solvable": True, "dlogr_offset": d, "rmse_after": rmse(new),
                           "r2_after": stats(new, core)["r2"],
                           "rt_baseline": float(toc_params["rt_baseline"]) * 10 ** (-d),
                           base_key: (float(b0) - d / coef) if b0 is not None else None,
                           "porosity_curve": curve, "unit": unit}
    out["rmse_now"] = rmse(_toc_from_dlogr(dl, vsh, lom0, min_vsh))
    return out


# --------------------------------------------------------------------------- run
def calibrate(core: pd.DataFrame, curves: pd.DataFrame, summary: dict, depth_shift: float = 0.0,
              tolerance: float = DEFAULT_TOLERANCE, phi_curve: str = "PHIT", include_badhole: bool = False) -> dict:
    m = match_depths(core, curves, depth_shift, tolerance)
    use = m["MATCHED"] & (include_badhole | ~m["QC_BADHOLE"])
    toc_rows = use & m["CORE_TOC"].notna()
    phi_rows = use & m["CORE_PHI"].notna()
    params = summary["parameters"]

    report = {
        "inputs": {"depth_shift": depth_shift, "depth_rule": "log_depth = core_depth + depth_shift",
                   "tolerance": tolerance, "depth_unit": summary.get("well", {}).get("depth_unit"),
                   "porosity_curve": phi_curve, "bad_hole_included": include_badhole,
                   "core_warnings": list(core.attrs.get("warnings", []))},
        "matching": {"core_samples": int(len(m)), "matched": int(m["MATCHED"].sum()),
                     "unmatched": int((~m["MATCHED"]).sum()),
                     "matched_in_bad_hole": int((m["MATCHED"] & m["QC_BADHOLE"]).sum()),
                     "unmatched_depths": [float(x) for x in m.loc[~m["MATCHED"], "CORE_DEPTH"]]},
        "toc": {"stats": stats(m.loc[toc_rows, "TOC"], m.loc[toc_rows, "CORE_TOC"]),
                "parameters_used": {k: params["toc"].get(k) for k in
                                    ("method", "lom", "min_vsh", "rt_baseline", "dt_baseline", "rhob_baseline",
                                     "nphi_baseline", "baseline_interval")}},
        "porosity": {"stats": stats(m.loc[phi_rows, phi_curve], m.loc[phi_rows, "CORE_PHI"]),
                     "parameters_used": {k: params["porosity"].get(k) for k in ("rho_matrix", "rho_fluid", "phit_method")}},
        "samples": m.assign(USED_TOC=toc_rows, USED_PHI=phi_rows).replace({np.nan: None}).to_dict(orient="records"),
    }
    if toc_rows.any() and "DLOGR" in m.columns:
        report["toc"]["adjustments"] = toc_adjustments(m[toc_rows], params["toc"])
    return report


# --------------------------------------------------------------------------- output
def _f(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def format_report(r: dict) -> str:
    i, mt = r["inputs"], r["matching"]
    u = (i["depth_unit"] or "").lower()
    L = ["=" * 78, "CORE CALIBRATION  (printed only; config.yaml is NOT changed)", "=" * 78,
         f"depth: log = core + {i['depth_shift']} {u}, nearest log sample within {i['tolerance']} {u}"
         f"{'' if i['bad_hole_included'] else ', bad-hole matches excluded'}",
         f"core samples {mt['core_samples']}: matched {mt['matched']}, unmatched {mt['unmatched']}, "
         f"matched in bad hole {mt['matched_in_bad_hole']}"]
    for w in i.get("core_warnings", []):
        L.append(f"  WARNING (core file): {w}")
    if mt["unmatched"]:
        L.append(f"  unmatched core depths (outside the logs or tolerance): {mt['unmatched_depths'][:10]}"
                 + (" ..." if mt["unmatched"] > 10 else ""))

    def block(name, s, unit):
        L.append(f"\n{name}  (residual = log - core)")
        if not s["n"]:
            L.append("  no matched samples")
            return
        L.append(f"  n {s['n']}   bias {_f(s['bias'])} {unit}   RMSE {_f(s['rmse'])} {unit}   R2 {_f(s['r2'])}   "
                 f"(core mean {_f(s['core_mean'])}, log mean {_f(s['log_mean'])})")

    block("TOC [wt%]", r["toc"]["stats"], "wt%")
    a = r["toc"].get("adjustments")
    if a and a["n"]:
        cur = a["current"]
        L.append(f"  to remove the TOC bias (Passey {a['method']}; current LOM {cur['lom']}, "
                 f"RMSE now {_f(a['rmse_now'])} wt%):")
        lo = a["lom"]
        if lo["solvable"]:
            L.append(f"    LOM {cur['lom']} -> {lo['lom']:.2f}  (change {lo['change']:+.2f})   "
                     f"RMSE after {_f(lo['rmse_after'])}  R2 after {_f(lo['r2_after'])}")
            if not lo["plausible"]:
                L.append(f"      WARNING: LOM {lo['lom']:.2f} is outside {LOM_PLAUSIBLE[0]:g}-{LOM_PLAUSIBLE[1]:g}; "
                         "it is probably compensating for a baseline error. Check LOM against Ro / Tmax.")
        else:
            L.append(f"    LOM: {lo['reason']}")
        b = a["baseline"]
        if b["solvable"]:
            key = next(k for k in b if k.endswith("_baseline") and k != "rt_baseline")
            L.append(f"    OR shift DlogR by {b['dlogr_offset']:+.3f}: rt_baseline {cur['rt_baseline']} -> "
                     f"{b['rt_baseline']:.3f} ohm.m   RMSE after {_f(b['rmse_after'])}  R2 after {_f(b['r2_after'])}")
            if b[key] is not None:
                L.append(f"       (or keep rt_baseline and move {key} {cur.get(key)} -> {b[key]:.3f} {b['unit']})")
        else:
            L.append(f"    baseline: {b['reason']}")
        if a.get("lom", {}).get("solvable") and b["solvable"]:
            better = "LOM" if a["lom"]["rmse_after"] < b["rmse_after"] else "baseline"
            L.append(f"    lower RMSE with the {better} change: the error looks "
                     f"{'proportional' if better == 'LOM' else 'like a constant offset'}.")
        if a.get("note_masked"):
            L.append(f"    note: {a['note_masked']}")
        L.append("    Changing TOC also changes PHIT/PHIE through the kerogen correction: re-run the pipeline.")
    block(f"POROSITY [v/v]  log {r['inputs']['porosity_curve']} vs core PHI", r["porosity"]["stats"], "v/v")
    L.append("=" * 78)
    return "\n".join(L)


def plot_crossplots(r: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(r["samples"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    phi_curve = r["inputs"]["porosity_curve"]
    for ax, (core_c, log_c, label, log_label, s) in zip(axes, (
            ("CORE_TOC", "TOC", "TOC [wt%]", "log TOC [wt%]", r["toc"]["stats"]),
            ("CORE_PHI", phi_curve, "porosity [v/v]", f"log {phi_curve} [v/v]", r["porosity"]["stats"]))):
        if core_c not in df or log_c not in df:
            ax.set_axis_off()
            continue
        x = pd.to_numeric(df[core_c], errors="coerce")
        y = pd.to_numeric(df[log_c], errors="coerce")
        used = df[f"USED_{'TOC' if core_c == 'CORE_TOC' else 'PHI'}"].astype(bool)
        bad = df["QC_BADHOLE"].astype(bool) & x.notna() & y.notna()
        ax.scatter(x[used], y[used], s=22, c="tab:blue", label="used")
        if bad.any():
            ax.scatter(x[bad], y[bad], s=22, facecolors="none", edgecolors="red", label="bad hole (not used)")
        vals = pd.concat([x, y]).dropna()
        if len(vals):
            lo, hi = min(0.0, vals.min()), vals.max() * 1.05 + 1e-9
            ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        ax.set_xlabel(f"core {label}")
        ax.set_ylabel(log_label)
        ax.set_title(f"{label}: n {s['n']}, bias {_f(s['bias'])}, RMSE {_f(s['rmse'])}, R2 {_f(s['r2'], 2)}",
                     fontsize=9)
        ax.grid(True, lw=0.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Core vs log  (log depth = core depth + {r['inputs']['depth_shift']})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
