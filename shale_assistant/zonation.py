"""Zonation and per-zone summary (the JSON handed to the LLM).

Automatic zonation = greedy binary segmentation: repeatedly cut the well where a
cut removes the most within-zone variance of the chosen (standardised) logs,
until max_zones is reached or a cut no longer explains min_gain_fraction of the
total variance. No zone can be thinner than min_thickness.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import __version__
from .petrophysics import FORMULAS

UNITS = {
    "depth_and_thickness": "as in the LAS (see well.depth_unit), measured depth",
    "vsh": "v/v (fraction, 0-1)", "phit": "v/v (fraction)", "phie": "v/v (fraction)",
    "sw": "v/v (fraction, 0-1)", "toc": "wt%", "gr": "API", "rhob": "g/cc", "nphi": "v/v",
    "rt": "ohm.m", "rwa": "ohm.m", "dt": "us/ft", "cali": "in",
    "fractions_and_confidence": "fraction, 0-1",
}


def rnd(x, nd: int):
    """Round for the JSON; NaN / inf / None -> None."""
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def _mean(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


# --------------------------------------------------------------------------- zonation
def _segmentation_matrix(df: pd.DataFrame, curves: list[str], smooth: int) -> np.ndarray:
    cols = []
    for name in curves:
        base = name[:-len("_LOG10")] if name.endswith("_LOG10") else name
        if base not in df.columns:
            continue
        x = df[base].astype(float)
        if name.endswith("_LOG10"):
            x = np.log10(x.where(x > 0))
        if x.notna().sum() < 10:
            continue
        x = x.rolling(smooth, center=True, min_periods=1).median()
        x = x.interpolate(limit_direction="both")
        sd = float(x.std())
        if not sd > 0:
            continue
        cols.append(((x - x.mean()) / sd).to_numpy())
    if not cols:
        raise ValueError("none of zonation.curves has usable data")
    return np.column_stack(cols)


def segment(X: np.ndarray, max_zones: int, min_samples: int, min_gain_fraction: float) -> list[int]:
    """Greedy binary segmentation. Returns boundaries [0, b1, ..., n] (zone i = [b_i, b_i+1))."""
    n = X.shape[0]
    S1 = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, axis=0)])
    S2 = np.concatenate([[0.0], np.cumsum((X ** 2).sum(axis=1))])

    def cost(i, j):
        return (S2[j] - S2[i]) - ((S1[j] - S1[i]) ** 2).sum(axis=-1) / (j - i)

    def best_split(i: int, j: int):
        if j - i < 2 * min_samples:
            return None
        ks = np.arange(i + min_samples, j - min_samples + 1)
        gain = cost(i, j) - cost(i, ks) - cost(ks, j)
        best = int(np.argmax(gain))
        return float(gain[best]), int(ks[best])

    total = float(cost(0, n))
    bounds = [0, n]
    while len(bounds) - 1 < max_zones:
        candidates = []
        for i, j in zip(bounds[:-1], bounds[1:]):
            found = best_split(i, j)
            if found:
                candidates.append(found)
        if not candidates:
            break
        gain, k = max(candidates)
        if total <= 0 or gain < min_gain_fraction * total:
            break
        bounds = sorted(bounds + [k])
    return bounds


def make_zones(df: pd.DataFrame, cfg: dict, step: float) -> list[dict]:
    """Return [{'zone_id', 'name', 'i0', 'i1'}] with i0:i1 the row slice of each zone."""
    zc = cfg["zonation"]
    depth = df["DEPTH"].to_numpy()
    zones = []
    if zc["method"] == "tops":
        if not zc["tops"]:
            raise ValueError("zonation.method is 'tops' but zonation.tops is empty")
        for t in sorted(zc["tops"], key=lambda t: t["top"]):
            i0 = int(np.searchsorted(depth, float(t["top"]), side="left"))
            i1 = int(np.searchsorted(depth, float(t["base"]), side="left"))
            if i1 - i0 >= 2:
                zones.append({"name": str(t["name"]), "i0": i0, "i1": i1})
        if not zones:
            raise ValueError("no zonation.tops interval overlaps the processing interval")
    else:
        X = _segmentation_matrix(df, zc["curves"], int(zc["smooth_window"]))
        min_samples = max(2, int(math.ceil(float(zc["min_thickness"]) / step)))
        bounds = segment(X, int(zc["max_zones"]), min_samples, float(zc["min_gain_fraction"]))
        for k, (i0, i1) in enumerate(zip(bounds[:-1], bounds[1:]), start=1):
            zones.append({"name": f"Z{k}", "i0": i0, "i1": i1})
    for k, z in enumerate(zones, start=1):
        z["zone_id"] = k
    return zones


# --------------------------------------------------------------------------- averages
def petro_averages(sub: pd.DataFrame, params: dict, step: float, toc_cutoff: float) -> dict:
    """Zone averages of the computed curves. Shared by the pipeline and the chat tools,
    so a what-if re-run is summarised exactly like the base case."""
    def phi_weighted(sw_col: str, phi_col: str):
        sw, phi = sub[sw_col].to_numpy(dtype=float), sub[phi_col].to_numpy(dtype=float)
        ok = np.isfinite(sw) & np.isfinite(phi)
        total = phi[ok].sum()
        return float((sw[ok] * phi[ok]).sum() / total) if total > 0 else float("nan")

    toc = sub["TOC"].to_numpy(dtype=float)
    n = len(sub)
    arch_raw = sub["SW_ARCHIE_RAW"].to_numpy(dtype=float)
    arch_phi = sub[params["sw"]["archie_porosity"]].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        above = arch_raw > 1.0
        low_phi = arch_phi < params["sw"]["phi_min"]
    n_raw = int(np.isfinite(arch_raw).sum())
    n_phi = int(np.isfinite(arch_phi).sum())
    return {
        "vsh_linear_mean": rnd(_mean(sub["VSH_LINEAR"]), 3),
        "vsh_larionov_mean": rnd(_mean(sub["VSH_LARIONOV"]), 3),
        "phit_mean": rnd(_mean(sub["PHIT"]), 3),
        "phie_mean": rnd(_mean(sub["PHIE"]), 3),
        "sw_archie_mean": rnd(_mean(sub["SW_ARCHIE"]), 3),
        "sw_simandoux_mean": rnd(_mean(sub["SW_SIMANDOUX"]), 3),
        "sw_archie_porosity_weighted": rnd(phi_weighted("SW_ARCHIE", params["sw"]["archie_porosity"]), 3),
        "sw_simandoux_porosity_weighted": rnd(phi_weighted("SW_SIMANDOUX", params["sw"]["simandoux_porosity"]), 3),
        "toc_mean_wtpct": rnd(_mean(toc), 2),
        "toc_max_wtpct": rnd(np.nanmax(toc), 2) if np.isfinite(toc).any() else None,
        "net_toc_thickness": rnd(float(np.nansum(toc >= toc_cutoff)) * step, 2),
        "rwa_median": rnd(np.nanmedian(sub["RWA"]), 4) if sub["RWA"].notna().any() else None,
        "valid_fraction_porosity": rnd(sub["PHIT"].notna().mean(), 3),
        "valid_fraction_sw": rnd(sub["SW_ARCHIE"].notna().mean(), 3),
        "valid_fraction_toc": rnd(np.isfinite(toc).mean(), 3) if n else None,
        "sw_archie_above_one_fraction": rnd(above.sum() / n_raw, 3) if n_raw else None,
        "low_porosity_fraction": rnd(low_phi.sum() / n_phi, 3) if n_phi else None,
    }


def _log_averages(sub: pd.DataFrame) -> dict:
    out = {}
    for key, col, nd in (("gr_mean", "GR", 1), ("rhob_mean", "RHOB", 3), ("nphi_mean", "NPHI", 3),
                         ("dt_mean", "DT", 1), ("cali_mean", "CALI", 2)):
        out[key] = rnd(_mean(sub[col]), nd) if col in sub.columns else None
    rt = sub["RT"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        lg = np.log10(np.where(rt > 0, rt, np.nan))
    out["rt_geometric_mean"] = rnd(10 ** _mean(lg), 2) if np.isfinite(lg).any() else None
    return out


def _facies_summary(sub: pd.DataFrame, threshold: float) -> dict | None:
    if "FACIES" not in sub.columns:
        return None
    fac = sub["FACIES"].dropna()
    if fac.empty:
        return {"dominant": None, "dominant_fraction": None, "mean_confidence": None,
                "classified_fraction": 0.0, "distribution": {}, "low_confidence": True}
    shares = fac.value_counts(normalize=True)
    mean_conf = float(sub["FACIES_CONF"].mean())
    return {
        "dominant": str(shares.index[0]),
        "dominant_fraction": rnd(shares.iloc[0], 3),
        "mean_confidence": rnd(mean_conf, 3),
        "classified_fraction": rnd(len(fac) / len(sub), 3),
        "distribution": {str(k): rnd(v, 3) for k, v in shares.items()},
        "low_confidence": bool(mean_conf < threshold),
    }


def summarise_zone(sub: pd.DataFrame, zone: dict, params: dict, cfg: dict, step: float,
                   base_depth: float) -> dict:
    fl = cfg["flags"]
    top = float(sub["DEPTH"].iloc[0])
    avg = petro_averages(sub, params, step, float(cfg["cutoffs"]["toc_min_wtpct"]))
    facies = _facies_summary(sub, float(cfg["facies"]["low_confidence_threshold"]))
    qc = {
        "badhole_fraction": rnd(sub["QC_BADHOLE"].mean(), 3),
        "spike_samples": int(sub["QC_SPIKE"].sum()),
        "out_of_range_samples": int(sub["QC_RANGE"].sum()),
        "flatline_samples": int(sub["QC_FLAT"].sum()),
        "undergauge_caliper_samples": int(sub["QC_UNDERGAUGE"].sum()),
    }

    reasons = []
    if qc["badhole_fraction"] is not None and qc["badhole_fraction"] > fl["badhole_fraction"]:
        reasons.append("WASHOUT: bad hole over a large part of the zone; density, neutron, porosity and Sw are unreliable")
    for key, label in (("valid_fraction_porosity", "porosity"), ("valid_fraction_sw", "Sw"),
                       ("valid_fraction_toc", "TOC")):
        if avg[key] is None or avg[key] < fl["min_valid_fraction"]:
            reasons.append(f"MISSING DATA: {label} could be computed on only part of the zone; the average is not representative")
    if avg["sw_archie_above_one_fraction"] is not None and avg["sw_archie_above_one_fraction"] > fl["sw_above_one_fraction"]:
        reasons.append("SW ABOVE 1: Archie gives Sw > 1 on many samples before clipping; Rw, m or the matrix density do not fit this zone")
    if facies is not None:
        if facies["low_confidence"]:
            reasons.append("FACIES: classifier confidence is below the threshold")
        if facies["classified_fraction"] is not None and facies["classified_fraction"] < fl["min_valid_fraction"]:
            reasons.append("FACIES: only part of the zone could be classified (missing or rejected log samples)")

    return {
        "zone_id": zone["zone_id"],
        "name": zone["name"],
        "top": rnd(top, 2),
        "base": rnd(base_depth, 2),
        "thickness": rnd(base_depth - top, 2),
        "n_samples": int(len(sub)),
        "averages": avg,
        "log_averages": _log_averages(sub),
        "facies": facies,
        "qc": qc,
        "low_confidence": bool(reasons),
        "low_confidence_reasons": reasons,
    }


def _compact_qc(rep: dict) -> dict:
    """Short version of the QC report for the LLM (the full report is saved separately)."""
    w = rep["washout"]
    return {
        "depth_issues": rep["depth"]["issues"],
        "warnings": rep["warnings"],
        "curves_not_in_file": rep["curves_not_in_file"],
        "unit_conversions": {k: v["conversion"] for k, v in rep["curves_found"].items() if v["conversion"] != "none"},
        "missing_fraction_in_interval": {k: v["missing_fraction"] for k, v in rep["missing_data"].items()},
        "washout": {k: w.get(k) for k in ("assessed", "bit_size_source", "threshold_in", "samples",
                                          "fraction", "max_overgauge_in", "not_assessed_samples", "note")
                    if w.get(k) is not None},
        "undergauge_caliper_samples": rep.get("undergauge_caliper", {}).get("samples", 0),
        "spike_samples": {k: v["samples"] for k, v in rep["spikes"].items()},
        "flatline_samples": {k: v["samples"] for k, v in rep["flatline"].items()},
        "out_of_range_samples": {k: v["samples"] for k, v in rep["out_of_range"].items()},
        "samples_set_to_nan": rep["samples_set_to_nan"],
        "badhole_action": rep["badhole_action"],
        "spike_action": rep["spike_action"],
    }


def build_summary(df: pd.DataFrame, zones: list[dict], params: dict, cfg: dict, qc_report: dict,
                  well: dict, step: float, facies_info: dict) -> dict:
    """df = cleaned logs + QC flags + computed curves (+ FACIES). Adds a ZONE column to df."""
    depth = df["DEPTH"].to_numpy()
    df["ZONE"] = None
    out_zones = []
    for z in zones:
        sub = df.iloc[z["i0"]:z["i1"]]
        base_depth = float(depth[z["i1"]]) if z["i1"] < len(depth) else float(depth[-1])
        df.iloc[z["i0"]:z["i1"], df.columns.get_loc("ZONE")] = z["name"]
        out_zones.append(summarise_zone(sub, z, params, cfg, step, base_depth))

    def ranking(key: str, reverse: bool) -> list[str]:
        have = [(z["averages"][key], z["name"]) for z in out_zones if z["averages"][key] is not None]
        return [name for _, name in sorted(have, key=lambda t: t[0], reverse=reverse)]

    used = ("vsh_larionov_" + params["vsh"]["larionov_variant"]
            if params["vsh"]["use_for_downstream"] == "larionov" else "vsh_linear")
    toc_key = {"sonic": "dlogr_sonic", "density": "dlogr_density", "neutron": "dlogr_neutron"}[params["toc"]["method"]]
    phit_key = "phit_" + params["porosity"]["phit_method"]
    formulas = {k: FORMULAS[k] for k in ("igr", "vsh_linear", "vsh_larionov_" + params["vsh"]["larionov_variant"],
                                         toc_key, "toc", "phid", "kerogen_volume", "phid_kerogen_corrected",
                                         "phin_kerogen_corrected", phit_key,
                                         "phie_" + params["porosity"]["phie_method"], "sw_archie",
                                         "sw_simandoux", "sw_low_porosity", "rwa")}

    return {
        "schema_version": "1.0",
        "generated_by": f"shale_assistant {__version__} - every number below was computed in Python",
        "well": well,
        "units": UNITS,
        "parameters": {
            **{k: params[k] for k in ("vsh", "toc", "porosity", "sw")},
            "vsh_used_downstream": used,
            "auto_picked": params["auto_picked"],
            "notes": params["notes"],
            "cutoffs": cfg["cutoffs"],
            "low_confidence_thresholds": {**cfg["flags"],
                                          "facies_mean_confidence": cfg["facies"]["low_confidence_threshold"]},
        },
        "formulas": formulas,
        "qc_summary": _compact_qc(qc_report),
        "facies_model": facies_info,
        "zonation": {"method": cfg["zonation"]["method"], "n_zones": len(out_zones),
                     "curves": cfg["zonation"]["curves"] if cfg["zonation"]["method"] == "auto" else None},
        "zones": out_zones,
        "rankings": {
            "toc_mean_wtpct_high_to_low": ranking("toc_mean_wtpct", True),
            "net_toc_thickness_high_to_low": ranking("net_toc_thickness", True),
            "phit_mean_high_to_low": ranking("phit_mean", True),
            "sw_archie_mean_low_to_high": ranking("sw_archie_mean", False),
        },
    }
