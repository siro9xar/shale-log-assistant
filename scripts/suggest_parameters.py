"""Suggest Rw, a Passey baseline interval and matrix densities from the pipeline output.

Prints suggestions only. It never writes config.yaml: copy a value there yourself if you agree.
Run scripts/run_pipeline.py first (this reads curves.csv and zone_summary.json from --out).
The only file written is the RT-vs-DT (or RHOB) shale crossplot PNG.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))            # so "shale_assistant" imports from any working directory
import argparse
import json

import pandas as pd

from shale_assistant.suggest import run_suggestions

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "example"),
                    help="folder with curves.csv and zone_summary.json from run_pipeline.py")
    ap.add_argument("--method", choices=["sonic", "density"],
                    help="Passey crossplot curve: sonic = RT vs DT, density = RT vs RHOB (default: toc.method used in the run)")
    ap.add_argument("--plot", help="crossplot PNG path (default: <out>/passey_baseline_crossplot.png)")
    args = ap.parse_args()

    out = Path(args.out)
    curves = pd.read_csv(out / "curves.csv")
    summary = json.loads((out / "zone_summary.json").read_text(encoding="utf-8"))
    plot = Path(args.plot) if args.plot else out / "passey_baseline_crossplot.png"
    _, text = run_suggestions(curves, summary, args.method, plot)
    print(text)
