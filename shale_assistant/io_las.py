"""Read a LAS file with lasio, map mnemonics to standard names and convert
every curve to the standard units used by the calculations.

Standard units after loading
    GR, CGR : API          CALI, BS : inches       RHOB, DRHO : g/cc
    NPHI    : v/v          RT       : ohm.m        DT         : us/ft
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import lasio
import numpy as np
import pandas as pd

STANDARD_UNITS = {
    "GR": "API", "CGR": "API", "CALI": "in", "BS": "in", "RHOB": "g/cc",
    "DRHO": "g/cc", "NPHI": "v/v", "RT": "ohm.m", "DT": "us/ft", "PEF": "b/e",
}

_GCC = {"G/CC", "G/C3", "G/CM3", "GM/CC", "GM/C3", "G/CM^3", "GR/CC", "G.CM-3"}
_KGM3 = {"KG/M3", "K/M3", "KG/M^3", "KG.M-3"}
_PERCENT = {"%", "PU", "P.U.", "P.U", "PERCENT", "PCT", "LPU", "SPU"}
_FRACTION = {"V/V", "DEC", "DECIMAL", "FRAC", "FRACTION", "M3/M3", "CFCF", "FT3/FT3",
             "V/V_DECIMAL", "DECP"}
_US_FT = {"US/F", "US/FT", "USEC/FT", "USEC/F", "US/FOOT", "µS/FT", "ΜS/FT", "MICROSEC/FT"}
_US_M = {"US/M", "USEC/M", "µS/M", "ΜS/M", "MICROSEC/M"}
_INCH = {"IN", "INCH", "INCHES", "INS", '"'}
_API = {"GAPI", "API", "APIU", "API_UNITS", "GAMMA_API"}
_OHMM = {"OHMM", "OHM.M", "OHM-M", "OHM*M", "OHMS", "OHM", "OHM/M", "OHMM2/M", "OHM.M2/M", "OHM-M2/M", "OHMSM"}
_MM = {"MM"}
_CM = {"CM"}

US_PER_M_TO_US_PER_FT = 0.3048      # 1 ft = 0.3048 m  ->  (us/m) * 0.3048 = us/ft
MM_PER_INCH = 25.4
CM_PER_INCH = 2.54


@dataclass
class WellLog:
    name: str
    path: str
    depth_unit: str                    # "M" or "FT"
    df: pd.DataFrame                   # DEPTH + standard curves in standard units
    mapping: dict                      # standard name -> {mnemonic, unit_in_file, conversion}
    header: dict                       # STRT / STOP / STEP / NULL as written in the file
    missing_curves: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def _norm_unit(unit: str | None) -> str:
    return (unit or "").strip().upper().replace(" ", "")


def convert_to_standard(std: str, values, unit: str | None):
    """Return (values_in_standard_units, conversion_note, warning_or_None)."""
    u = _norm_unit(unit)
    v = np.asarray(values, dtype=float)
    med = float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")

    if std in ("RHOB", "DRHO"):
        if u in _KGM3:
            return v / 1000.0, "kg/m3 -> g/cc (divided by 1000)", None
        if u in _GCC:
            return v, "none", None
        if std == "RHOB" and np.isfinite(med) and med > 100:
            return v / 1000.0, "kg/m3 -> g/cc (divided by 1000)", (
                f"RHOB unit '{unit}' not recognised; median {med:.1f} looks like kg/m3, converted")
        return v, "none", f"{std} unit '{unit}' not recognised; assumed g/cc"

    if std == "NPHI":
        if u in _PERCENT:
            return v / 100.0, "percent -> v/v (divided by 100)", None
        if u in _FRACTION:
            return v, "none", None
        if np.isfinite(med) and med > 1.0:
            return v / 100.0, "percent -> v/v (divided by 100)", (
                f"NPHI unit '{unit}' not recognised; median {med:.2f} looks like percent, converted")
        return v, "none", f"NPHI unit '{unit}' not recognised; assumed v/v"

    if std == "DT":
        if u in _US_M:
            return v * US_PER_M_TO_US_PER_FT, "us/m -> us/ft (multiplied by 0.3048)", None
        if u in _US_FT:
            return v, "none", None
        if np.isfinite(med) and med > 200:
            return v * US_PER_M_TO_US_PER_FT, "us/m -> us/ft (multiplied by 0.3048)", (
                f"DT unit '{unit}' not recognised; median {med:.0f} looks like us/m, converted")
        return v, "none", f"DT unit '{unit}' not recognised; assumed us/ft"

    if std in ("CALI", "BS"):
        if u in _MM:
            return v / MM_PER_INCH, "mm -> in (divided by 25.4)", None
        if u in _CM:
            return v / CM_PER_INCH, "cm -> in (divided by 2.54)", None
        if u in _INCH:
            return v, "none", None
        if np.isfinite(med) and med > 60:
            return v / MM_PER_INCH, "mm -> in (divided by 25.4)", (
                f"{std} unit '{unit}' not recognised; median {med:.0f} looks like mm, converted")
        return v, "none", f"{std} unit '{unit}' not recognised; assumed inches"

    # No conversion exists for these; a wrong unit must at least be visible in the QC report.
    if std in ("GR", "CGR") and u and u not in _API:
        return v, "none", (f"{std} unit '{unit}' is not API. Vsh uses relative gamma ray and still works, but the "
                           "facies classifier expects API: its result is not reliable for this well")
    if std == "RT" and u and u not in _OHMM:
        return v, "none", (f"RT unit '{unit}' is not ohm.m; no conversion applied. If this is a conductivity "
                           "curve, Sw and TOC are wrong: map a resistivity curve instead")

    return v, "none", None


def _depth_unit(las: lasio.LASFile) -> tuple[str, str | None]:
    unit = None
    try:
        unit = las.index_unit
    except Exception:  # pragma: no cover - defensive
        unit = None
    if unit is None:
        try:
            unit = las.well["STRT"].unit
        except Exception:  # pragma: no cover
            unit = None
    u = _norm_unit(unit)
    if u in ("M", "METER", "METERS", "METRE", "METRES"):
        return "M", None
    if u in ("FT", "F", "FEET", "FOOT"):
        return "FT", None
    return "M", f"depth unit '{unit}' not recognised; assumed metres"


def load_las(path: str | Path, cfg: dict) -> WellLog:
    """Read the LAS, keep only the curves named in cfg['curves'], standardise units."""
    path = Path(path)
    las = lasio.read(str(path))

    warnings: list[str] = []
    depth_unit, warn = _depth_unit(las)
    if warn:
        warnings.append(warn)

    depth = np.asarray(las.index, dtype=float)
    available = {}
    for curve in las.curves[1:]:                         # curve 0 is the depth index
        key = curve.mnemonic.upper().split(":")[0]
        available.setdefault(key, curve)                 # first occurrence wins

    data = {"DEPTH": depth}
    mapping, missing = {}, []
    for std, aliases in cfg["curves"].items():
        found = next((available[a.upper()] for a in aliases if a.upper() in available), None)
        if found is None:
            missing.append(std)
            continue
        values, note, warn = convert_to_standard(std, np.asarray(found.data, dtype=float), found.unit)
        if warn:
            warnings.append(warn)
        data[std] = values
        mapping[std] = {
            "mnemonic": found.mnemonic,
            "unit_in_file": found.unit or "",
            "standard_unit": STANDARD_UNITS.get(std, ""),
            "conversion": note,
        }

    def _hdr(mnemonic):
        try:
            return float(las.well[mnemonic].value)
        except Exception:
            return None

    name = path.stem
    for mnemonic in ("WBN", "WELL"):                     # wellbore name first, then well name
        try:
            candidate = str(las.well[mnemonic].value).strip()
        except Exception:
            candidate = ""
        if candidate:
            name = candidate
            break

    header = {"STRT": _hdr("STRT"), "STOP": _hdr("STOP"), "STEP": _hdr("STEP"), "NULL": _hdr("NULL")}
    return WellLog(name=name, path=str(path), depth_unit=depth_unit, df=pd.DataFrame(data),
                   mapping=mapping, header=header, missing_curves=missing, warnings=warnings)
