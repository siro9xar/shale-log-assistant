"""Convert a tabular well-log CSV (one row per depth) into a LAS 2.0 file, without changing any value.

Units come from a data dictionary CSV with columns 'Element Name' and 'Units' (USGS data-release style,
e.g. "grams per cubic centimeter [g/cc]" -> "g/cc"), matched by column name, else by column position.
--unit NAME=UNIT overrides or fills in a unit. The null value is written to the LAS header unchanged.

Example (USGS GC-1 West Woodway):
    python scripts/csv_to_las.py --csv data/usgs_gc1_west_woodway/wireline_geophysical_welllogs.csv \
        --dictionary data/usgs_gc1_west_woodway/wireline_geophysical_welllogs_datadictionary.csv \
        --out data/usgs_gc1_west_woodway/GC-1_West_Woodway.las --well "USGS GC-1 West Woodway" \
        --unit NPHI=PU --source "USGS data release doi:10.5066/P9KZVG7P"
"""
import argparse
import re
import unicodedata
from pathlib import Path

import lasio
import numpy as np
import pandas as pd


# LAS 2.0 is ASCII: write standard ASCII unit names (a UTF-8 "µs/ft" is read back as "Âµs/ft").
ASCII_UNITS = {"µs/ft": "US/F", "μs/ft": "US/F", "Ωm": "OHMM", "Ω·m": "OHMM", "ohm-m": "OHMM",
               "(mΩm)-1": "MMHO/M", "°F": "DEGF", "°C": "DEGC", "feet": "FT", "ft": "FT", "pu": "PU",
               "API": "GAPI", "g/cc": "G/CC", "in": "IN", "lbs": "LBF", "mV": "MV", "ppm": "PPM", "%": "%"}


_ASCII_NFKC = {unicodedata.normalize("NFKC", k): v for k, v in ASCII_UNITS.items()}


def ascii_unit(u: str) -> str:
    key = unicodedata.normalize("NFKC", u.strip())          # OHM SIGN -> GREEK OMEGA, MICRO SIGN -> MU
    u = _ASCII_NFKC.get(key, u.strip())
    if not u.isascii():
        raise SystemExit(f"unit {u!r} is not ASCII; give it with --unit NAME=UNIT")
    return u


def units_from_dictionary(path, columns):
    d = pd.read_csv(path, encoding="utf-8-sig")
    names = [str(n).strip() for n in d["Element Name"]]
    units = []
    for u in d["Units"].astype(str):
        m = re.search(r"\[([^\]]+)\]", u)
        units.append(m.group(1).strip() if m else u.strip())
    by_name = dict(zip(names, units))
    out = {}
    for i, c in enumerate(columns):
        if c in by_name:
            out[c] = by_name[c]
        elif i < len(units):                     # same position in the dictionary (renamed column)
            out[c] = units[i]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dictionary", help="data dictionary CSV with 'Element Name' and 'Units'")
    ap.add_argument("--unit", action="append", default=[], metavar="NAME=UNIT", help="override a curve unit")
    ap.add_argument("--depth", default="DEPTH", help="depth column (default DEPTH)")
    ap.add_argument("--null", type=float, default=-999.25)
    ap.add_argument("--well", default="")
    ap.add_argument("--source", default="", help="text for the ~Other section (where the data came from)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    units = units_from_dictionary(args.dictionary, list(df.columns)) if args.dictionary else {}
    for item in args.unit:
        k, v = item.split("=", 1)
        units[k.strip()] = v.strip()
    units = {k: ascii_unit(v) for k, v in units.items()}

    depth = df[args.depth].to_numpy(dtype=float)
    if not np.all(np.diff(depth) > 0):
        raise SystemExit("depth must be strictly increasing")
    las = lasio.LASFile()
    las.well["NULL"].value = args.null
    if args.well:
        las.well["WELL"].value = args.well
    unit_d = units.get(args.depth, "")
    las.append_curve("DEPT", depth, unit=unit_d, descr="depth")
    for c in df.columns:
        if c == args.depth:
            continue
        values = df[c].to_numpy(dtype=float)
        las.append_curve(c, np.where(values == args.null, np.nan, values), unit=units.get(c, ""))
    las.other = (f"Converted from {Path(args.csv).name} by scripts/csv_to_las.py; values unchanged. "
                 + args.source)
    las.write(args.out, version=2.0, fmt="%.5f")
    back = lasio.read(args.out)                 # round-trip check: same curves, same values
    for c in df.columns:
        if c == args.depth:
            continue
        a = df[c].replace(args.null, np.nan).to_numpy(dtype=float)
        b = back[c]
        if not np.allclose(a, b, equal_nan=True, atol=1e-5):
            raise SystemExit(f"round-trip mismatch in {c}")
    print(f"wrote {args.out}: {len(depth)} samples, {depth[0]}-{depth[-1]} {unit_d}, curves "
          + ", ".join(f"{c} [{units.get(c, '')}]" for c in df.columns if c != args.depth))
    print("round-trip check passed: every value in the LAS equals the CSV")
