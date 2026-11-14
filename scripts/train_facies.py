"""Train the random-forest lithofacies classifier with a WELL-BASED train/test split.

--table takes the FORCE table from prepare_force_training.py (default) or your own labelled CSV with
the columns  WELL, DEPTH, GR, RHOB, NPHI, RT, DT, FACIES  in standard units
(GR API, RHOB g/cc, NPHI v/v, RT ohm.m, DT us/ft; FACIES = any text label).

5 or more wells: whole wells are held out (facies.test_fraction or facies.test_wells in the config).
2 to 4 wells   : falls back to leave-one-well-out and says so. Rows are never split at random.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))            # so "shale_assistant" imports from any working directory
import argparse

import pandas as pd

from shale_assistant.config import load_config
from shale_assistant.facies import load_training_table, train_classifier


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", default=str(ROOT / "data" / "force" / "training_table.csv"),
                    help="labelled CSV: WELL, DEPTH, GR, RHOB, NPHI, RT, DT, FACIES (default: the FORCE table)")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--out", default=str(ROOT / "models"))
    ap.add_argument("--compare-random-split", action="store_true",
                    help="also score a random row split, to show how much it overstates accuracy")
    args = ap.parse_args()

    cfg = load_config(args.config)
    try:
        table = load_training_table(args.table)
    except ValueError as err:
        sys.exit(f"ERROR: {err}")
    m = train_classifier(table, cfg, args.out, compare_random_split=args.compare_random_split)

    s, t = m["split"], m["test"]
    lowo = s["evaluation"] == "leave_one_well_out"
    if lowo:
        print(f"Split: LEAVE-ONE-WELL-OUT  (fallback: the table has {s['n_test_wells']} labelled wells, "
              f"fewer than the 5 needed to hold wells out)")
        print(f"  {s['note']}")
        print(f"  {'held-out well':<20}{'samples':>9}{'accuracy':>10}{'bal. acc.':>11}{'macro F1':>10}")
        for w in s["per_well"]:
            print(f"  {w['well']:<20}{w['n_samples']:>9}{w['accuracy']:>10.3f}{w['balanced_accuracy']:>11.3f}"
                  f"{w['macro_f1']:>10.3f}")
    else:
        print(f"Split: {s['method']}")
        print(f"  train: {s['n_train_wells']} wells, {s['n_train_samples']} samples")
        print(f"  test : {s['n_test_wells']} wells, {s['n_test_samples']} samples  -> {', '.join(s['test_wells'])}")
    if m["classes_dropped_as_too_rare"]:
        print(f"  classes dropped (too few samples): {m['classes_dropped_as_too_rare']}")
    if m["rows_dropped_missing_features_or_label"]:
        print(f"  rows dropped (missing curve or label): {m['rows_dropped_missing_features_or_label']}")
    for note in m["notes"]:
        print(f"  NOTE: {note}")
    head = "LEAVE-ONE-WELL-OUT (all held-out predictions pooled)" if lowo else "HELD-OUT WELLS"
    print(f"\n{head}: accuracy {t['accuracy']}, balanced accuracy {t['balanced_accuracy']}, macro F1 {t['macro_f1']}")
    if m["cross_validation"]:
        cv = m["cross_validation"]
        print(f"Well-grouped {cv['n_splits']}-fold CV on training wells: accuracy {cv['accuracy_mean']} +/- {cv['accuracy_std']}, "
              f"macro F1 {cv['macro_f1_mean']} +/- {cv['macro_f1_std']}")
    if "random_row_split_for_comparison_only" in m:
        r = m["random_row_split_for_comparison_only"]
        print(f"Random ROW split (leaky, for comparison only): accuracy {r['accuracy']}, macro F1 {r['macro_f1']}")
    print("\nPer class on held-out wells:")
    print(f"  {'class':<18}{'precision':>10}{'recall':>8}{'f1':>7}{'support':>9}")
    for name, v in t["per_class"].items():
        print(f"  {name:<18}{v['precision']:>10.3f}{v['recall']:>8.3f}{v['f1']:>7.3f}{v['support']:>9}")
    cm = m["confusion_matrix"]
    print("\nConfusion matrix (rows = true, columns = predicted):")
    print(pd.DataFrame(cm["matrix"], index=cm["labels"], columns=cm["labels"]).to_string())
    c = m["confidence"]
    print(f"\nConfidence check on held-out wells (threshold {c['threshold']}): accuracy "
          f"{c['test_accuracy_when_confidence_at_or_above_threshold']} when confident, "
          f"{c['test_accuracy_when_confidence_below_threshold']} when not "
          f"({c['share_of_test_samples_below_threshold']:.0%} of samples are below the threshold)")
    print(f"\nSaved to {args.out}/: facies_rf.joblib, facies_metrics.json, confusion_matrix.csv, confusion_matrix.png")
