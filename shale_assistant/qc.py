"""Data quality control. Runs BEFORE any petrophysical calculation.

Checks
    depth     : reversals, duplicates, irregular step, gaps, header vs data
    missing   : null samples and gaps per curve
    range     : values outside physical limits            -> set to NaN
    flatline  : long runs of identical values (dead tool) -> set to NaN
    washout   : CALI - bit size > threshold  (and |DRHO| > threshold if DRHO exists)
    spikes    : Hampel filter (rolling median / MAD), only features narrower than
                the tools can resolve are called spikes

The report is returned as a dict (saved as JSON) and as text (printed).
"""
from __future__ import annotations

from pathlib import Path

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .io_las import WellLog

LOG_CURVES = ["GR", "CGR", "RHOB", "NPHI", "RT", "DT", "CALI", "BS", "DRHO", "PEF"]
MAX_INTERVALS_IN_REPORT = 8


@dataclass
class QCResult:
    df: pd.DataFrame      # processing interval: cleaned curves + QC_* flag columns
    report: dict
    step: float           # median depth step in the processing interval


# --------------------------------------------------------------------------- helpers
def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) index pairs for every run of True in mask."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    change = np.diff(padded.astype(np.int8))
    return list(zip(np.flatnonzero(change == 1).tolist(), np.flatnonzero(change == -1).tolist()))


def _r(x, nd=2):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def _intervals(depth: np.ndarray, mask: np.ndarray, extra: np.ndarray | None = None) -> list:
    """Longest flagged intervals as [top, base] (plus max of `extra` inside, if given)."""
    runs = sorted(true_runs(mask), key=lambda ab: ab[0] - ab[1])[:MAX_INTERVALS_IN_REPORT]
    out = []
    for a, b in sorted(runs):
        item = [_r(depth[a]), _r(depth[b - 1])]
        if extra is not None:
            item.append(_r(np.nanmax(extra[a:b])))
        out.append(item)
    return out


# --------------------------------------------------------------------------- depth
def depth_qc(depth: np.ndarray, header: dict, tol: float) -> dict:
    d = np.asarray(depth, dtype=float)
    diffs = np.diff(d)
    issues = []
    n_reversals = int((diffs < 0).sum())
    n_duplicates = int((diffs == 0).sum())
    med = float(np.median(np.abs(diffs))) if len(diffs) else float("nan")

    if n_reversals and n_reversals == len(diffs):
        issues.append("depth is recorded bottom-up (decreasing); it was reversed")
    elif n_reversals:
        issues.append(f"{n_reversals} depth reversals; samples were sorted by depth")
    if n_duplicates:
        issues.append(f"{n_duplicates} duplicate depths; first occurrence kept")

    pos = np.abs(diffs)
    irregular = np.abs(pos - med) > tol * med
    gaps = pos > 1.5 * med
    gap_list = [[_r(d[i]), _r(d[i + 1])] for i in np.flatnonzero(gaps)[:MAX_INTERVALS_IN_REPORT]]
    if irregular.sum():
        issues.append(f"{int(irregular.sum())} irregular depth steps (median step {med:.4f})")
    if gaps.sum():
        issues.append(f"{int(gaps.sum())} depth gaps larger than 1.5 x the median step")

    header_ok = True
    step = header.get("STEP")
    if step is not None and step != 0 and np.isfinite(med) and abs(abs(step) - med) > tol * med:
        header_ok = False
        issues.append(f"header STEP {step} does not match the data step {med:.4f}")
    strt, stop = header.get("STRT"), header.get("STOP")
    if len(d) and strt is not None and stop is not None and np.isfinite(med):
        lo, hi = min(strt, stop), max(strt, stop)
        if abs(lo - d.min()) > med or abs(hi - d.max()) > med:
            header_ok = False
            issues.append(f"header STRT/STOP ({strt}, {stop}) do not match the data "
                          f"({d.min():.4f}, {d.max():.4f})")

    return {
        "n_samples": int(len(d)),
        "top": _r(d.min(), 4) if len(d) else None,
        "base": _r(d.max(), 4) if len(d) else None,
        "median_step": _r(med, 4),
        "n_reversals": n_reversals,
        "n_duplicate_depths": n_duplicates,
        "n_irregular_steps": int(irregular.sum()),
        "n_gaps": int(gaps.sum()),
        "gaps": gap_list,
        "header_matches_data": header_ok,
        "issues": issues,
    }


def _sorted_unique(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("DEPTH", kind="mergesort")
    df = df.drop_duplicates(subset="DEPTH", keep="first")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- detectors
def flatline_mask(x: np.ndarray, min_samples: int) -> np.ndarray:
    """True where a value belongs to a run of >= min_samples identical values."""
    x = np.asarray(x, dtype=float)
    out = np.zeros(len(x), dtype=bool)
    if len(x) < 2:
        return out
    same = np.concatenate(([False], np.diff(x) == 0))     # NaN - NaN = NaN -> False
    for a, b in true_runs(same):
        if (b - a + 1) >= min_samples:                    # run of k "same" flags = k + 1 samples
            out[a - 1:b] = True
    return out


def hampel_spikes(x: np.ndarray, window: int, nsigma: float, min_scale: float,
                  max_width: int) -> np.ndarray:
    """Hampel identifier. Only flagged runs of <= max_width samples are kept as spikes."""
    s = pd.Series(np.asarray(x, dtype=float))
    min_periods = max(3, window // 2)
    med = s.rolling(window, center=True, min_periods=min_periods).median()
    dev = (s - med).abs()
    mad = dev.rolling(window, center=True, min_periods=min_periods).median()
    scale = np.maximum(1.4826 * mad.to_numpy(), min_scale)
    with np.errstate(invalid="ignore"):
        flagged = dev.to_numpy() > nsigma * scale
    flagged = np.where(np.isnan(dev.to_numpy()), False, flagged)
    out = np.zeros(len(s), dtype=bool)
    for a, b in true_runs(flagged):
        if (b - a) <= max_width:
            out[a:b] = True
    return out


# --------------------------------------------------------------------------- main
def _display_path(path) -> str:
    """Path relative to the working directory when the file is under it, else just the file name:
    saved reports must not carry a user's home folder or machine layout."""
    p = Path(path)
    try:
        return p.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return p.name


def run_qc(log: WellLog, cfg: dict) -> QCResult:
    qc = cfg["qc"]
    report: dict = {
        "well": log.name,
        "file": _display_path(log.path),
        "depth_unit": log.depth_unit,
        "header": log.header,
        "curves_found": log.mapping,
        "curves_not_in_file": log.missing_curves,
        "warnings": list(log.warnings),
    }

    # ---- depth (whole file)
    report["depth"] = depth_qc(log.df["DEPTH"].to_numpy(), log.header, qc["depth_step_tolerance"])
    full = _sorted_unique(log.df)
    depth_full = full["DEPTH"].to_numpy()
    curves = [c for c in LOG_CURVES if c in full.columns]

    # ---- coverage (whole file)
    coverage = {}
    for c in curves:
        valid = full[c].notna().to_numpy()
        if valid.any():
            idx = np.flatnonzero(valid)
            coverage[c] = {"first_valid_depth": _r(depth_full[idx[0]]),
                           "last_valid_depth": _r(depth_full[idx[-1]]),
                           "missing_fraction_of_file": _r(1 - valid.mean(), 3)}
        else:
            coverage[c] = {"first_valid_depth": None, "last_valid_depth": None,
                           "missing_fraction_of_file": 1.0}
    report["coverage_whole_file"] = coverage

    # ---- processing interval
    required = cfg["well"]["required_curves"]
    absent = [c for c in required if c not in full.columns]
    if absent:
        raise ValueError(f"required curve(s) {absent} not found in {log.path}. "
                         f"Add the mnemonic to the 'curves' aliases in config.yaml.")
    interval = cfg["well"].get("interval")
    if interval:
        top, base = float(interval[0]), float(interval[1])
        reason = "set in config (well.interval)"
    else:
        all_valid = full[required].notna().all(axis=1).to_numpy()
        if not all_valid.any():
            raise ValueError(f"no depth where all required curves {required} have data")
        idx = np.flatnonzero(all_valid)
        top, base = depth_full[idx[0]], depth_full[idx[-1]]
        reason = f"first to last depth where {required} all have data"
    df = full[(full["DEPTH"] >= top) & (full["DEPTH"] <= base)].reset_index(drop=True).copy()
    if len(df) < 10:
        raise ValueError("processing interval holds fewer than 10 samples")
    depth = df["DEPTH"].to_numpy()
    step = float(np.median(np.diff(depth)))
    n = len(df)
    report["processing_interval"] = {"top": _r(depth[0]), "base": _r(depth[-1]),
                                     "n_samples": n, "step": _r(step, 4), "reason": reason}

    # ---- missing data inside the interval (as recorded in the file)
    missing = {}
    for c in curves:
        isnan = df[c].isna().to_numpy()
        runs = true_runs(isnan)
        longest = max(runs, key=lambda ab: ab[1] - ab[0]) if runs else None
        missing[c] = {
            "missing_samples": int(isnan.sum()),
            "missing_fraction": _r(isnan.mean(), 3),
            "n_gaps": len(runs),
            "longest_gap": [_r(depth[longest[0]]), _r(depth[longest[1] - 1])] if longest else None,
        }
    report["missing_data"] = missing

    nulled = {c: {} for c in curves}

    # ---- physical range
    out_of_range = {}
    range_any = np.zeros(n, dtype=bool)
    for c, (lo, hi) in qc["physical_limits"].items():
        if c not in df.columns:
            continue
        x = df[c].to_numpy()
        with np.errstate(invalid="ignore"):
            bad = (x < lo) | (x > hi)
        bad = np.where(np.isnan(x), False, bad)
        out_of_range[c] = {"samples": int(bad.sum()), "limits": [lo, hi],
                           "intervals": _intervals(depth, bad)}
        if bad.any():
            df.loc[bad, c] = np.nan
            nulled[c]["out_of_range"] = int(bad.sum())
            range_any |= bad
    report["out_of_range"] = out_of_range

    # ---- flatline (dead / pegged tool)
    flat = {}
    flat_any = np.zeros(n, dtype=bool)
    for c in qc["flatline_curves"]:
        if c not in df.columns:
            continue
        bad = flatline_mask(df[c].to_numpy(), qc["flatline_min_samples"])
        flat[c] = {"samples": int(bad.sum()), "intervals": _intervals(depth, bad)}
        if bad.any():
            df.loc[bad, c] = np.nan
            nulled[c]["flatline"] = int(bad.sum())
            flat_any |= bad
    report["flatline"] = flat

    # ---- hole condition
    washout = np.zeros(n, dtype=bool)
    undergauge = np.zeros(n, dtype=bool)
    if "CALI" in df.columns and df["CALI"].notna().any():
        cali = df["CALI"].to_numpy()
        if "BS" in df.columns and df["BS"].notna().any():
            bs = df["BS"].ffill().bfill().to_numpy()
            bs_source = "bit-size curve in the LAS"
        else:
            bs = np.full(n, float(qc["bit_size_in"]))
            bs_source = f"config qc.bit_size_in = {qc['bit_size_in']} in (ASSUMED, no bit-size curve)"
        over = cali - bs
        with np.errstate(invalid="ignore"):
            washout = np.where(np.isnan(over), False, over > qc["washout_in"])
            undergauge = np.where(np.isnan(over), False, -over > qc["undergauge_in"])
        report["washout"] = {
            "assessed": True,
            "bit_size_source": bs_source,
            "threshold_in": qc["washout_in"],
            "samples": int(washout.sum()),
            "fraction": _r(washout.mean(), 3),
            "max_overgauge_in": _r(np.nanmax(over)) if np.isfinite(over).any() else None,
            "not_assessed_samples": int(np.isnan(over).sum()),
            "intervals_top_base_maxovergauge": _intervals(depth, washout, over),
        }
        report["undergauge_caliper"] = {
            "threshold_in": qc["undergauge_in"],
            "samples": int(undergauge.sum()),
            "intervals": _intervals(depth, undergauge),
            "note": "caliper reads smaller than the bit: suspect caliper, hole condition unknown there",
        }
    else:
        report["washout"] = {"assessed": False, "samples": 0, "fraction": 0.0,
                             "note": "no caliper curve: washouts CANNOT be detected"}
        report["warnings"].append("no caliper curve: washouts cannot be detected")

    drho_bad = np.zeros(n, dtype=bool)
    if "DRHO" in df.columns and df["DRHO"].notna().any():
        with np.errstate(invalid="ignore"):
            drho_bad = np.where(df["DRHO"].isna(), False, df["DRHO"].abs() > qc["drho_max"])
        report["density_correction"] = {"threshold_gcc": qc["drho_max"], "samples": int(drho_bad.sum()),
                                        "intervals": _intervals(depth, drho_bad)}
    badhole = washout | drho_bad

    # ---- spikes
    sp = qc["spike"]
    spikes = {}
    spike_any = np.zeros(n, dtype=bool)
    for c in sp["curves"]:
        if c not in df.columns:
            continue
        x = df[c].to_numpy()
        if c == "RT":
            with np.errstate(invalid="ignore", divide="ignore"):
                x = np.log10(x)
        bad = hampel_spikes(x, sp["window"], sp["nsigma"], sp["min_scale"].get(c, 0.0), sp["max_width"])
        spikes[c] = {"samples": int(bad.sum()),
                     "first_depths": [_r(v) for v in depth[bad][:MAX_INTERVALS_IN_REPORT]]}
        spike_any |= bad
        if bad.any() and sp["action"] == "set_nan":
            df.loc[bad, c] = np.nan
            nulled[c]["spike"] = int(bad.sum())
    report["spikes"] = spikes
    report["spike_action"] = sp["action"]

    # ---- bad-hole action
    report["badhole_action"] = qc["badhole_action"]
    if qc["badhole_action"] == "null_density_neutron":
        for c in ("RHOB", "NPHI"):
            if c in df.columns:
                hit = badhole & df[c].notna().to_numpy()
                if hit.any():
                    df.loc[hit, c] = np.nan
                    nulled[c]["badhole"] = int(hit.sum())

    report["samples_set_to_nan"] = {c: v for c, v in nulled.items() if v}

    df["QC_WASHOUT"] = washout
    df["QC_UNDERGAUGE"] = undergauge
    df["QC_BADHOLE"] = badhole
    df["QC_SPIKE"] = spike_any
    df["QC_RANGE"] = range_any
    df["QC_FLAT"] = flat_any
    return QCResult(df=df, report=report, step=step)


# --------------------------------------------------------------------------- text report
def format_qc_report(rep: dict) -> str:
    u = rep["depth_unit"].lower()
    L = []
    L.append("=" * 78)
    L.append(f"QC REPORT  -  {rep['well']}   ({rep['file']})")
    L.append("=" * 78)

    L.append("\nCurves found (file mnemonic [unit] -> standard name, conversion):")
    for std, m in rep["curves_found"].items():
        L.append(f"  {m['mnemonic']:<6} [{m['unit_in_file'] or '?':<5}] -> {std:<5} [{m['standard_unit']}]"
                 f"   conversion: {m['conversion']}")
    if rep["curves_not_in_file"]:
        L.append(f"Curves NOT in file: {', '.join(rep['curves_not_in_file'])}")

    d = rep["depth"]
    L.append(f"\nDEPTH ({u}): {d['n_samples']} samples, {d['top']} - {d['base']}, median step {d['median_step']}")
    if d["issues"]:
        for issue in d["issues"]:
            L.append(f"  FLAG  {issue}")
    else:
        L.append("  ok    monotonic, regular step, header matches data")

    L.append("\nCOVERAGE over the whole file:")
    for c, v in rep["coverage_whole_file"].items():
        L.append(f"  {c:<5} valid {v['first_valid_depth']} - {v['last_valid_depth']} {u}"
                 f"   missing {v['missing_fraction_of_file']:.1%} of file")

    p = rep["processing_interval"]
    L.append(f"\nPROCESSING INTERVAL: {p['top']} - {p['base']} {u}  ({p['n_samples']} samples; {p['reason']})")

    L.append("\nMISSING DATA inside the processing interval:")
    for c, v in rep["missing_data"].items():
        tag = "FLAG" if v["missing_samples"] else "ok  "
        gap = f", longest gap {v['longest_gap'][0]} - {v['longest_gap'][1]} {u}" if v["longest_gap"] else ""
        L.append(f"  {tag}  {c:<5} {v['missing_samples']:>6} samples ({v['missing_fraction']:.1%})"
                 f" in {v['n_gaps']} gap(s){gap}")

    L.append("\nOUT OF PHYSICAL RANGE (set to NaN):")
    for c, v in rep["out_of_range"].items():
        tag = "FLAG" if v["samples"] else "ok  "
        L.append(f"  {tag}  {c:<5} {v['samples']:>6} samples outside {v['limits']}")

    L.append("\nFLATLINE / dead tool (set to NaN):")
    for c, v in rep["flatline"].items():
        tag = "FLAG" if v["samples"] else "ok  "
        where = f"  at {v['intervals']}" if v["samples"] else ""
        L.append(f"  {tag}  {c:<5} {v['samples']:>6} samples{where}")

    w = rep["washout"]
    L.append("\nWASHOUT (caliper):")
    if not w["assessed"]:
        L.append(f"  FLAG  {w['note']}")
    else:
        tag = "FLAG" if w["samples"] else "ok  "
        L.append(f"  bit size: {w['bit_size_source']}")
        L.append(f"  {tag}  {w['samples']} samples ({w['fraction']:.1%}) with CALI - bit > {w['threshold_in']} in;"
                 f" max over-gauge {w['max_overgauge_in']} in")
        for top, base, mx in w["intervals_top_base_maxovergauge"]:
            L.append(f"        {top} - {base} {u}   (max +{mx} in)")
        if w["not_assessed_samples"]:
            L.append(f"  FLAG  {w['not_assessed_samples']} samples have no caliper: hole condition unknown")
        ug = rep["undergauge_caliper"]
        if ug["samples"]:
            L.append(f"  FLAG  {ug['samples']} samples with caliper < bit size by > {ug['threshold_in']} in"
                     f" at {ug['intervals']}: suspect caliper")
    if "density_correction" in rep:
        dc = rep["density_correction"]
        tag = "FLAG" if dc["samples"] else "ok  "
        L.append(f"  {tag}  {dc['samples']} samples with |DRHO| > {dc['threshold_gcc']} g/cc")
    L.append(f"  action on bad hole: {rep['badhole_action']}")

    L.append(f"\nSPIKES (Hampel filter; action: {rep['spike_action']}):")
    for c, v in rep["spikes"].items():
        tag = "FLAG" if v["samples"] else "ok  "
        where = f"  first at {v['first_depths']}" if v["samples"] else ""
        L.append(f"  {tag}  {c:<5} {v['samples']:>6} samples{where}")

    if rep["samples_set_to_nan"]:
        L.append("\nSAMPLES SET TO NaN BY QC (curve: reason = count):")
        for c, reasons in rep["samples_set_to_nan"].items():
            L.append(f"  {c:<5} " + ", ".join(f"{k} = {v}" for k, v in reasons.items()))

    if rep["warnings"]:
        L.append("\nWARNINGS:")
        for wtxt in rep["warnings"]:
            L.append(f"  FLAG  {wtxt}")
    L.append("=" * 78)
    return "\n".join(L)
