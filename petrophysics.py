"""Petrophysics. Pure numpy functions + a staged workflow.

Units expected by every function in this module
    GR API | RHOB g/cc | NPHI v/v | RT, Rw, Rsh ohm.m | DT us/ft | TOC wt% | Vsh, PHI, Sw v/v

Workflow order (each stage only needs the stages before it)
    1. Vsh        from gamma ray
    2. TOC        Passey delta-log-R (needs Vsh only for the reservoir mask)
    3. Porosity   density / neutron, kerogen-corrected (needs TOC) and clay-corrected (needs Vsh)
    4. Sw         Archie and Simandoux (needs porosity and Vsh)
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

# The formulas, as text, travel with the results (JSON, README, report).
FORMULAS = {
    "igr": "IGR = (GR - GR_clean) / (GR_shale - GR_clean), clipped to 0-1",
    "vsh_linear": "Vsh = IGR",
    "vsh_larionov_older": "Vsh = 0.33 * (2^(2 * IGR) - 1)   [Larionov 1969, older / consolidated rocks]",
    "vsh_larionov_tertiary": "Vsh = 0.083 * (2^(3.7 * IGR) - 1)   [Larionov 1969, Tertiary / unconsolidated rocks]",
    "dlogr_sonic": "DlogR = log10(RT / RT_baseline) + 0.02 * (DT - DT_baseline)   [DT in us/ft]",
    "dlogr_density": "DlogR = log10(RT / RT_baseline) - 2.50 * (RHOB - RHOB_baseline)   [RHOB in g/cc]",
    "dlogr_neutron": "DlogR = log10(RT / RT_baseline) + 4.0 * (NPHI - NPHI_baseline)   [NPHI in v/v]",
    "toc": "TOC [wt%] = DlogR * 10^(2.297 - 0.1688 * LOM); negative values set to 0; set to 0 where Vsh < min_vsh   [Passey et al. 1990]",
    "phid": "PHID = (rho_matrix - RHOB) / (rho_matrix - rho_fluid)",
    "kerogen_volume": "Vker = k_factor * (TOC / 100) * RHOB / rho_kerogen",
    "phid_kerogen_corrected": "PHID_k = PHID - Vker * (rho_matrix - rho_kerogen) / (rho_matrix - rho_fluid)",
    "phin_kerogen_corrected": "PHIN_k = NPHI + nphi_matrix_offset - Vker * hi_kerogen",
    "phit_density": "PHIT = PHID_k, clipped to 0-phi_max",
    "phit_density_neutron": "PHIT = (PHID_k + PHIN_k) / 2 (liquid) or sqrt((PHID_k^2 + PHIN_k^2) / 2) (gas), clipped to 0-phi_max",
    "phie_density": "PHIE = PHID_k - Vsh * PHID_shale, clipped to 0-phi_max and never above PHIT; "
                    "PHID_shale = (rho_matrix - rho_shale) / (rho_matrix - rho_fluid)",
    "phie_density_neutron": "PHID_c = PHID_k - Vsh * PHID_shale; PHIN_c = PHIN_k - Vsh * PHIN_shale; "
                            "PHIE = (PHID_c + PHIN_c) / 2 (liquid) or sqrt((PHID_c^2 + PHIN_c^2) / 2) (gas), "
                            "clipped to 0-phi_max and never above PHIT; "
                            "PHID_shale = (rho_matrix - rho_shale) / (rho_matrix - rho_fluid)",
    "sw_archie": "Sw = ((a * Rw) / (PHI^m * RT))^(1/n), clipped to 0-1   [Archie 1942]",
    "sw_simandoux": "1/RT = PHI^m * Sw^n / (a * Rw) + Vsh * Sw / Rsh, solved for Sw, clipped to 0-1   [Simandoux 1963]",
    "sw_low_porosity": "Sw = 1 where PHI < phi_min",
    "rwa": "Rwa = RT * PHIT^m / a   (apparent water resistivity; equals Rw in clean water-bearing rock)",
}

OUTPUT_CURVES = ["IGR", "VSH_LINEAR", "VSH_LARIONOV", "VSH", "DLOGR", "TOC", "VKER", "PHID", "PHIN",
                 "PHIT", "PHIE", "SW_ARCHIE", "SW_SIMANDOUX", "SW_ARCHIE_RAW", "SW_SIMANDOUX_RAW", "RWA"]


# =========================================================================== Vsh
def gamma_ray_index(gr, gr_clean: float, gr_shale: float):
    if not gr_shale > gr_clean:
        raise ValueError(f"gr_shale ({gr_shale}) must be larger than gr_clean ({gr_clean})")
    return np.clip((np.asarray(gr, dtype=float) - gr_clean) / (gr_shale - gr_clean), 0.0, 1.0)


def vsh_linear(igr):
    return np.asarray(igr, dtype=float)


def vsh_larionov_older(igr):
    return np.clip(0.33 * (2.0 ** (2.0 * np.asarray(igr, dtype=float)) - 1.0), 0.0, 1.0)


def vsh_larionov_tertiary(igr):
    return np.clip(0.083 * (2.0 ** (3.7 * np.asarray(igr, dtype=float)) - 1.0), 0.0, 1.0)


# =========================================================================== TOC
def delta_log_r(rt, rt_baseline: float, porosity_log, porosity_baseline: float, method: str):
    """Passey delta-log-R. porosity_log is DT (us/ft), RHOB (g/cc) or NPHI (v/v)."""
    rt = np.asarray(rt, dtype=float)
    x = np.asarray(porosity_log, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        res = np.log10(rt / rt_baseline)
    if method == "sonic":
        return res + 0.02 * (x - porosity_baseline)
    if method == "density":
        return res - 2.50 * (x - porosity_baseline)
    if method == "neutron":
        return res + 4.0 * (x - porosity_baseline)
    raise ValueError(f"unknown TOC method {method!r}")


def toc_passey(dlogr, lom: float):
    """TOC in wt%. Negative delta-log-R gives 0."""
    toc = np.asarray(dlogr, dtype=float) * 10.0 ** (2.297 - 0.1688 * lom)
    return np.where(np.isnan(toc), np.nan, np.maximum(toc, 0.0))


# =========================================================================== porosity
def density_porosity(rhob, rho_matrix: float, rho_fluid: float):
    return (rho_matrix - np.asarray(rhob, dtype=float)) / (rho_matrix - rho_fluid)


def kerogen_volume(toc_wtpct, rhob, rho_kerogen: float, k_factor: float):
    """Kerogen volume fraction of bulk rock. TOC in wt% (divided by 100 here)."""
    toc = np.nan_to_num(np.asarray(toc_wtpct, dtype=float), nan=0.0)   # no TOC estimate -> no correction
    return k_factor * (toc / 100.0) * np.asarray(rhob, dtype=float) / rho_kerogen


def combine_density_neutron(phid, phin, fluid_type: str):
    phid = np.asarray(phid, dtype=float)
    phin = np.asarray(phin, dtype=float)
    if fluid_type == "gas":
        d = np.where(np.isnan(phid), np.nan, np.maximum(phid, 0.0))
        nn = np.where(np.isnan(phin), np.nan, np.maximum(phin, 0.0))
        return np.sqrt((d ** 2 + nn ** 2) / 2.0)
    return (phid + phin) / 2.0


def _clip_keep_nan(x, lo, hi):
    x = np.asarray(x, dtype=float)
    return np.where(np.isnan(x), np.nan, np.clip(x, lo, hi))


# =========================================================================== Sw
def sw_archie(rt, phi, rw: float, a: float = 1.0, m: float = 2.0, n: float = 2.0):
    """Archie water saturation, NOT clipped (values > 1 are informative for QC)."""
    rt = np.asarray(rt, dtype=float)
    phi = np.asarray(phi, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((a * rw) / (phi ** m * rt)) ** (1.0 / n)


def sw_simandoux(rt, phi, vsh, rw: float, rsh: float, a: float = 1.0, m: float = 2.0, n: float = 2.0):
    """Simandoux (1963) water saturation, NOT clipped.

    Solves  C*Sw^n + B*Sw - 1/RT = 0  with  C = PHI^m / (a*Rw),  B = Vsh / Rsh.
    Closed form for n = 2, bisection otherwise (the function is monotonic in Sw).
    """
    rt = np.asarray(rt, dtype=float)
    phi = np.asarray(phi, dtype=float)
    vsh = np.asarray(vsh, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = phi ** m / (a * rw)
        b = vsh / rsh
        d = 1.0 / rt
        if abs(n - 2.0) < 1e-12:
            return (-b + np.sqrt(b * b + 4.0 * c * d)) / (2.0 * c)
        hi = (d / c) ** (1.0 / n)            # Archie Sw is an upper bound (f(hi) = B*hi >= 0)
        lo = np.zeros_like(hi)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            f = c * mid ** n + b * mid - d
            go_down = f > 0
            hi = np.where(go_down, mid, hi)
            lo = np.where(go_down, lo, mid)
        sw = 0.5 * (lo + hi)
        return np.where(np.isnan(rt) | np.isnan(phi) | np.isnan(vsh), np.nan, sw)


def apparent_water_resistivity(rt, phi, a: float = 1.0, m: float = 2.0):
    return np.asarray(rt, dtype=float) * np.asarray(phi, dtype=float) ** m / a


# =========================================================================== stages
def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    if name in df.columns:
        return df[name].to_numpy(dtype=float)
    return np.full(len(df), np.nan)


def stage_vsh(df: pd.DataFrame, p: dict) -> dict:
    igr = gamma_ray_index(_col(df, p["gr_curve"]), p["gr_clean"], p["gr_shale"])
    lin = vsh_linear(igr)
    lar = vsh_larionov_older(igr) if p["larionov_variant"] == "older" else vsh_larionov_tertiary(igr)
    used = lin if p["use_for_downstream"] == "linear" else lar
    return {"IGR": igr, "VSH_LINEAR": lin, "VSH_LARIONOV": lar, "VSH": used}


def stage_toc(df: pd.DataFrame, vsh: np.ndarray, p: dict) -> dict:
    method = p["method"]
    curve, base = {"sonic": ("DT", "dt_baseline"), "density": ("RHOB", "rhob_baseline"),
                   "neutron": ("NPHI", "nphi_baseline")}[method]
    dlogr = delta_log_r(_col(df, "RT"), p["rt_baseline"], _col(df, curve), p[base], method)
    toc = toc_passey(dlogr, p["lom"])
    with np.errstate(invalid="ignore"):
        reservoir = vsh < p["min_vsh"]
    toc = np.where(reservoir, 0.0, toc)                   # delta-log-R in reservoir rock is not kerogen
    toc = np.where(np.isnan(vsh), np.nan, toc)
    return {"DLOGR": dlogr, "TOC": toc}


def stage_porosity(df: pd.DataFrame, vsh: np.ndarray, toc: np.ndarray, p: dict) -> dict:
    rhob, nphi = _col(df, "RHOB"), _col(df, "NPHI")
    rma, rfl = p["rho_matrix"], p["rho_fluid"]
    denom = rma - rfl
    ker = p["kerogen"]
    has_neutron = bool(np.isfinite(nphi).any())

    phid = density_porosity(rhob, rma, rfl)
    phin = nphi + p["nphi_matrix_offset"]

    if ker["apply"]:
        vker = kerogen_volume(toc, rhob, ker["rho_kerogen"], ker["k_factor"])
        phid_k = phid - vker * (rma - ker["rho_kerogen"]) / denom
        phin_k = phin - vker * ker["hi_kerogen"]
    else:
        vker = np.where(np.isnan(rhob), np.nan, 0.0)
        phid_k, phin_k = phid, phin

    phid_sh = max((rma - p["rho_shale"]) / denom, 0.0)
    phin_sh = p["nphi_shale"] + p["nphi_matrix_offset"]
    phid_c = phid_k - vsh * phid_sh
    phin_c = phin_k - vsh * phin_sh

    if p["phit_method"] == "density_neutron" and has_neutron:
        phit = combine_density_neutron(phid_k, phin_k, p["fluid_type"])
    else:
        phit = phid_k
    phit = _clip_keep_nan(phit, 0.0, p["phi_max"])

    if p["phie_method"] == "density_neutron" and has_neutron:
        phie = combine_density_neutron(phid_c, phin_c, p["fluid_type"])
    else:
        phie = phid_c
    phie = _clip_keep_nan(phie, 0.0, p["phi_max"])
    phie = np.where(np.isnan(phie) | np.isnan(phit), np.nan, np.minimum(phie, phit))

    return {"VKER": vker, "PHID": phid, "PHIN": phin, "PHIT": phit, "PHIE": phie}


def stage_sw(df: pd.DataFrame, vsh: np.ndarray, por: dict, p: dict) -> dict:
    rt = _col(df, "RT")
    phi_a = por[p["archie_porosity"]]
    phi_s = por[p["simandoux_porosity"]]

    def _finish(raw, phi, also_invalid=None):
        """Clip to 0-1; Sw = 1 where porosity < phi_min; keep the unclipped value for QC."""
        with np.errstate(invalid="ignore"):
            low = phi < p["phi_min"]
        invalid = np.isnan(phi) | np.isnan(rt)
        if also_invalid is not None:
            invalid = invalid | also_invalid
        sw = _clip_keep_nan(raw, 0.0, 1.0)
        sw = np.where(low, 1.0, sw)                       # no pore space to evaluate
        sw = np.where(invalid, np.nan, sw)
        raw = np.where(low | invalid, np.nan, raw)        # unclipped value only where it means something
        return sw, raw

    arch_raw = sw_archie(rt, phi_a, p["rw"], p["a"], p["m"], p["n"])
    sim_raw = sw_simandoux(rt, phi_s, vsh, p["rw"], p["rsh"], p["a"], p["m"], p["n"])
    arch, arch_raw = _finish(arch_raw, phi_a)
    sim, sim_raw = _finish(sim_raw, phi_s, also_invalid=np.isnan(vsh))
    rwa = apparent_water_resistivity(rt, por["PHIT"], p["a"], p["m"])
    return {"SW_ARCHIE": arch, "SW_SIMANDOUX": sim, "SW_ARCHIE_RAW": arch_raw,
            "SW_SIMANDOUX_RAW": sim_raw, "RWA": rwa}


def compute_all(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Run the four stages with fully resolved parameters (no 'auto' left)."""
    v = stage_vsh(df, params["vsh"])
    t = stage_toc(df, v["VSH"], params["toc"])
    por = stage_porosity(df, v["VSH"], t["TOC"], params["porosity"])
    s = stage_sw(df, v["VSH"], por, params["sw"])
    out = {}
    for block in (v, t, por, s):
        out.update(block)
    return pd.DataFrame(out, index=df.index)[OUTPUT_CURVES]


# =========================================================================== parameter resolution
def _median(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.nanmedian(x)) if np.isfinite(x).any() else float("nan")


def resolve_and_compute(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Turn every 'auto' in the config into an explicit number, then compute.

    Returns (curves, params). params is what was actually used, with params['auto_picked']
    explaining how each automatic value was obtained.
    """
    auto: dict[str, str] = {}
    notes: list[str] = []
    depth = df["DEPTH"].to_numpy()

    # ---------------- Vsh
    vc = cfg["vsh"]
    gr_curve = vc["gr_curve"]
    if gr_curve not in df.columns or not df[gr_curve].notna().any():
        if gr_curve != "GR":
            notes.append(f"vsh.gr_curve = {gr_curve} is not in the LAS; GR used instead")
        gr_curve = "GR"
    gr = df[gr_curve].to_numpy(dtype=float)
    p_lo, p_hi = vc["auto_percentiles"]
    vsh_p = {"gr_curve": gr_curve, "larionov_variant": vc["larionov_variant"],
             "use_for_downstream": vc["use_for_downstream"]}
    for key, pct in (("gr_clean", p_lo), ("gr_shale", p_hi)):
        if vc[key] == "auto":
            vsh_p[key] = round(float(np.nanpercentile(gr, pct)), 2)
            auto[f"vsh.{key}"] = f"P{pct} of {gr_curve} over the processing interval"
        else:
            vsh_p[key] = float(vc[key])
    v = stage_vsh(df, vsh_p)

    # ---------------- TOC baselines
    tc = cfg["toc"]
    toc_p = {"method": tc["method"], "lom": float(tc["lom"]), "min_vsh": float(tc["min_vsh"]),
             "baseline_interval": tc.get("baseline_interval")}
    if tc.get("baseline_interval"):
        top, base = tc["baseline_interval"]
        sel = (depth >= top) & (depth <= base)
        how = f"median over baseline_interval {top}-{base}"
        if sel.sum() < 5:
            raise ValueError("toc.baseline_interval holds fewer than 5 samples of this well")
    else:
        with np.errstate(invalid="ignore"):
            sel = v["VSH_LINEAR"] >= 0.6
        how = "median over all samples with linear Vsh >= 0.6 (CRUDE first guess - pick a non-source shale interval)"
    for key, curve, nd in (("rt_baseline", "RT", 3), ("dt_baseline", "DT", 2),
                           ("rhob_baseline", "RHOB", 3), ("nphi_baseline", "NPHI", 3)):
        if tc[key] == "auto":
            value = _median(df[curve].to_numpy(dtype=float)[sel]) if curve in df.columns else float("nan")
            toc_p[key] = round(value, nd) if np.isfinite(value) else float("nan")
            auto[f"toc.{key}"] = how
        else:
            toc_p[key] = float(tc[key])
    needed = {"sonic": "dt_baseline", "density": "rhob_baseline", "neutron": "nphi_baseline"}[tc["method"]]
    for key in ("rt_baseline", needed):
        if not np.isfinite(toc_p[key]):
            raise ValueError(f"could not resolve toc.{key}: the curve has no data where the baseline is picked")
    for key in ("dt_baseline", "rhob_baseline", "nphi_baseline"):      # unused baselines: drop NaN
        if key != needed and not np.isfinite(toc_p[key]):
            toc_p[key] = None
            auto.pop(f"toc.{key}", None)
    t = stage_toc(df, v["VSH"], toc_p)

    # ---------------- shale point (non-organic shale) for the clay correction and Rsh
    pc = cfg["porosity"]
    with np.errstate(invalid="ignore"):
        shale = v["IGR"] >= pc["shale_point_min_igr"]
        lean = np.nan_to_num(t["TOC"], nan=0.0) < pc["shale_point_max_toc"]
    sel = shale & lean
    how = (f"median where IGR >= {pc['shale_point_min_igr']} and TOC < {pc['shale_point_max_toc']} wt% "
           f"({int(sel.sum())} samples)")
    if sel.sum() < 20:
        sel = shale
        how = f"median where IGR >= {pc['shale_point_min_igr']} ({int(sel.sum())} samples; too few lean-shale samples)"
        notes.append("few non-organic shale samples: shale point may be biased by organic shale")
    por_p = {k: pc[k] for k in ("rho_matrix", "rho_fluid", "nphi_matrix_offset", "fluid_type",
                                "phit_method", "phie_method", "phi_max")}
    por_p["kerogen"] = copy.deepcopy(pc["kerogen"])
    for key, curve, nd, fallback in (("rho_shale", "RHOB", 3, pc["rho_matrix"]), ("nphi_shale", "NPHI", 3, 0.0)):
        if pc[key] == "auto":
            value = _median(df[curve].to_numpy(dtype=float)[sel]) if curve in df.columns else float("nan")
            if not np.isfinite(value):
                value = fallback
                notes.append(f"porosity.{key} could not be picked from data; {fallback} used (no clay correction on that log)")
            por_p[key] = round(value, nd)
            auto[f"porosity.{key}"] = how
        else:
            por_p[key] = float(pc[key])
    if por_p["rho_shale"] > por_p["rho_matrix"]:
        notes.append(f"rho_shale ({por_p['rho_shale']}) > rho_matrix ({por_p['rho_matrix']}): "
                     "density clay correction is zero; consider a higher rho_matrix")
    por = stage_porosity(df, v["VSH"], t["TOC"], por_p)

    # ---------------- Sw
    sc = cfg["sw"]
    sw_p = {k: (float(sc[k]) if k in ("a", "m", "n", "rw", "phi_min") else sc[k])
            for k in ("a", "m", "n", "rw", "archie_porosity", "simandoux_porosity", "phi_min")}
    if sc["rsh"] == "auto":
        value = _median(df["RT"].to_numpy(dtype=float)[sel])
        if not np.isfinite(value):
            raise ValueError("could not resolve sw.rsh automatically; set a number in config.yaml")
        sw_p["rsh"] = round(value, 3)
        auto["sw.rsh"] = how.replace("median", "median RT")
    else:
        sw_p["rsh"] = float(sc["rsh"])
    s = stage_sw(df, v["VSH"], por, sw_p)

    out = {}
    for block in (v, t, por, s):
        out.update(block)
    curves = pd.DataFrame(out, index=df.index)[OUTPUT_CURVES]
    params = {"vsh": vsh_p, "toc": toc_p, "porosity": por_p, "sw": sw_p,
              "auto_picked": auto, "notes": notes}
    return curves, params
