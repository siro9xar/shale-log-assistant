"""Combine report.md, log_plot.png, the zone table, the confusion matrix and the verification result into one
self-contained HTML file (images embedded), plus a PDF made with matplotlib (no extra installs).

The verification status is stated in plain words at the top: VERIFIED, FAILED VERIFICATION, or NOT VERIFIED
when there is no report.md. Verification is re-run on the current report.md, not read from an old file.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))            # so "shale_assistant" imports from any working directory
import argparse

from shale_assistant.export import export

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "example"),
                    help="folder with zone_summary.json, curves.csv, log_plot.png and (optionally) report.md")
    ap.add_argument("--metrics", default=str(ROOT / "models" / "facies_metrics.json"),
                    help="facies_metrics.json of the model the pipeline used (for the confusion matrix)")
    ap.add_argument("--html", help="output HTML (default <out>/report.html)")
    ap.add_argument("--pdf", help="output PDF (default <out>/report.pdf)")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()

    r = export(args.out, args.metrics, args.html, args.pdf, make_pdf=not args.no_pdf)
    st = r["status"]
    print(st["headline"])
    for line in st["lines"]:
        print(f"  {line}")
    for line in st.get("failures", []):
        print(f"  FAILED: {line}")
    print(f"\nHTML: {r['html']}")
    if r["pdf"]:
        print(f"PDF : {r['pdf']} ({r['pdf_pages']} pages)")
    elif r["pdf_error"]:
        print(f"PDF : not written ({r['pdf_error']})")
    sys.exit({"ok": 0, "none": 0, "fail": 1}[st["level"]])
