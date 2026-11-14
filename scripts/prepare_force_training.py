"""Turn the FORCE 2020 training file into the training table used by train_facies.py.

Output columns (standard units): WELL, DEPTH [m], GR [API], RHOB [g/cc], NPHI [v/v],
RT [ohm.m], DT [us/ft], FACIES [text].  The FORCE file already uses these units.

NOTE: FORCE labels are LITHOLOGY classes interpreted by geologists in conventional
North Sea wells. They are not shale lithofacies from core/XRD. Swap in your own
labelled table (same columns) for a real shale play.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))            # so "shale_assistant" imports from any working directory
import argparse
import zipfile

import pandas as pd

LITHOLOGY = {30000: "Sandstone", 65030: "Sandstone/Shale", 65000: "Shale", 80000: "Marl",
             74000: "Dolomite", 70000: "Limestone", 70032: "Chalk", 88000: "Halite",
             86000: "Anhydrite", 99000: "Tuff", 90000: "Coal", 93000: "Basement"}
COLUMNS = {"WELL": "WELL", "DEPTH_MD": "DEPTH", "GR": "GR", "RHOB": "RHOB", "NPHI": "NPHI",
           "RDEP": "RT", "DTC": "DT", "FORCE_2020_LITHOFACIES_LITHOLOGY": "FACIES"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(ROOT / "data" / "force" / "train.zip"))
    ap.add_argument("--output", default=str(ROOT / "data" / "force" / "training_table.csv"))
    ap.add_argument("--quadrants", nargs="*", default=["15", "16", "25"],
                    help="keep wells whose name starts with these quadrant numbers (default: the "
                         "quadrants around the example well). Pass --quadrants with nothing after it for all wells.")
    args = ap.parse_args()

    src = Path(args.input)
    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as z, z.open(z.namelist()[0]) as fh:
            raw = pd.read_csv(fh, sep=";", usecols=list(COLUMNS))
    else:
        raw = pd.read_csv(src, sep=";", usecols=list(COLUMNS))
    df = raw.rename(columns=COLUMNS)
    df["FACIES"] = df["FACIES"].map(LITHOLOGY)
    if args.quadrants:
        df = df[df["WELL"].str.split("/").str[0].isin(args.quadrants)]
    before = len(df)
    df = df.dropna(subset=["GR", "RHOB", "NPHI", "RT", "DT", "FACIES"])
    df = df[df["RT"] > 0]
    df.to_csv(args.output, index=False)
    print(f"{len(df)} samples from {df['WELL'].nunique()} wells written to {args.output} "
          f"({before - len(df)} rows dropped for missing curves)")
    print(df["FACIES"].value_counts().to_string())
