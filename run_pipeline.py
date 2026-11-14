"""LAS -> QC report -> petrophysics -> facies -> zones -> zone_summary.json (no LLM involved)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))            # so "shale_assistant" imports from any working directory
import argparse

from shale_assistant.config import load_config
from shale_assistant.pipeline import run


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--las", default=str(ROOT / "data" / "volve" / "15-9-19_SR_COMP.LAS"))
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--model", default=str(ROOT / "models" / "facies_rf.joblib"),
                    help="trained facies model; if the file does not exist the facies step is skipped")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "example"))
    args = ap.parse_args()

    model = Path(args.model)
    if not model.exists():
        print(f"[warning] {model} not found: facies step skipped (run scripts/train_facies.py first)\n")
        model = None
    run(args.las, load_config(args.config), args.out, model_path=model)
