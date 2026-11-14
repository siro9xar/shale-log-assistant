"""Check an LLM report against the JSON it was written from.

  1. every number in the report exists in the JSON
  2. Vsh and Sw are between 0 and 1 (zone averages + every depth sample in curves.csv)
  3. every low-confidence zone is flagged in the report
Exit code 0 = passed, 1 = failed.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))            # so "shale_assistant" imports from any working directory
import argparse
import json
import pandas as pd

from shale_assistant.verify import format_verification, verify_report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "example"),
                    help="folder with zone_summary.json, report.md and curves.csv")
    ap.add_argument("--json", help="path to the zone summary (default: <out>/zone_summary.json)")
    ap.add_argument("--report", help="path to the report (default: <out>/report.md)")
    ap.add_argument("--curves", help="path to curves.csv (default: <out>/curves.csv if it exists)")
    ap.add_argument("--allow-rounding", action="store_true",
                    help="accept a report number that is a correctly rounded JSON number")
    args = ap.parse_args()

    out = Path(args.out)
    summary = json.loads(Path(args.json or out / "zone_summary.json").read_text(encoding="utf-8"))
    report_path = Path(args.report or out / "report.md")
    if not report_path.exists():
        sys.exit(f"ERROR: {report_path} does not exist: there is no report to verify. "
                 "Run scripts/generate_report.py first.")
    report = report_path.read_text(encoding="utf-8")
    curves_path = Path(args.curves or out / "curves.csv")
    curves = pd.read_csv(curves_path) if curves_path.exists() else None

    result = verify_report(report, summary, curves, allow_rounding=args.allow_rounding)
    print(format_verification(result))
    sys.exit(0 if result["passed"] else 1)
