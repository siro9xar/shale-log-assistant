"""Send zone_summary.json to the LLM, save report.md, then verify the report."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))            # so "shale_assistant" imports from any working directory
import argparse
import json
import pandas as pd

from shale_assistant.config import load_config
from shale_assistant.llm import generate_report
from shale_assistant.verify import format_verification


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "example"), help="folder holding zone_summary.json")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()

    out = Path(args.out)
    cfg = load_config(args.config)
    summary = json.loads((out / "zone_summary.json").read_text(encoding="utf-8"))
    curves = pd.read_csv(out / "curves.csv") if (out / "curves.csv").exists() else None

    report, verification, calls = generate_report(summary, cfg, curves=curves)
    (out / "report.md").write_text(report + "\n", encoding="utf-8")
    (out / "report_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(report)
    print()
    print(format_verification(verification))
    print(f"\nmodel {cfg['llm']['model']}, {calls} LLM call(s). Saved {out}/report.md and report_verification.json")
    sys.exit(0 if verification["passed"] else 1)
