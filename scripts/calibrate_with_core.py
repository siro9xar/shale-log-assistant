"""Compare log TOC and porosity with core data; print bias, RMSE, R2 and the LOM or baseline change
that would remove the TOC bias.

Core CSV columns: DEPTH, TOC_WTPCT, PHI (PHI as a fraction). TOC_WTPCT or PHI may be empty on any row.
Run scripts/run_pipeline.py first: this reads curves.csv and zone_summary.json from --out.

Writes <out>/calibration_report.json and <out>/calibration_crossplot.png. Never writes config.yaml.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))            # so "shale_assistant" imports from any working directory
import argparse
import json

import pandas as pd

from shale_assistant.core_calibration import (DEFAULT_TOLERANCE, calibrate, format_report, plot_crossplots,
                                              read_core)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core", required=True, help="core CSV: DEPTH, TOC_WTPCT, PHI")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "example"),
                    help="folder with curves.csv and zone_summary.json from run_pipeline.py")
    ap.add_argument("--depth-shift", type=float, default=0.0,
                    help="added to core depth to get log depth, in the LAS depth unit (default 0)")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                    help=f"max distance core -> nearest log sample, LAS depth unit (default {DEFAULT_TOLERANCE})")
    ap.add_argument("--phi-curve", choices=["PHIT", "PHIE"], default="PHIT",
                    help="log porosity to compare with core PHI (default PHIT: core porosity is usually total)")
    ap.add_argument("--include-badhole", action="store_true", help="also use core points that match bad hole")
    args = ap.parse_args()

    out = Path(args.out)
    curves = pd.read_csv(out / "curves.csv")
    summary = json.loads((out / "zone_summary.json").read_text(encoding="utf-8"))
    report = calibrate(read_core(args.core), curves, summary, args.depth_shift, args.tolerance,
                       args.phi_curve, args.include_badhole)
    report["inputs"]["core_file"] = str(args.core)
    (out / "calibration_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    plot_crossplots(report, out / "calibration_crossplot.png")
    print(format_report(report))
    print(f"\nSaved {out / 'calibration_report.json'} and {out / 'calibration_crossplot.png'}")
