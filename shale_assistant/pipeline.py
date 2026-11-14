"""End-to-end run: LAS -> QC -> petrophysics -> facies -> zones -> JSON / CSV / plot."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .facies import load_model, predict_facies
from .io_las import load_las
from .petrophysics import resolve_and_compute
from .qc import format_qc_report, run_qc
from .zonation import build_summary, make_zones

INPUT_CURVES = ["GR", "CGR", "CALI", "BS", "RHOB", "NPHI", "RT", "DT", "DRHO", "PEF"]
QC_COLUMNS = ["QC_WASHOUT", "QC_UNDERGAUGE", "QC_BADHOLE", "QC_SPIKE", "QC_RANGE", "QC_FLAT"]


def _facies_info(model_path, metrics_path, cfg) -> dict:
    if model_path is None:
        return {"available": False,
                "note": "no trained facies model was supplied; facies fields are null"}
    info = {"available": True, "model_file": Path(model_path).name,
            "low_confidence_threshold": cfg["facies"]["low_confidence_threshold"],
            "confidence_definition": "mean of the per-sample maximum class probability (share of tree votes); not calibrated"}
    if metrics_path and Path(metrics_path).exists():
        m = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        info.update({
            "classes": m["classes"],
            "features": m["features"],
            "split": ("leave-one-well-out: each well held out once; the saved model is trained on all wells"
                      if m["split"].get("evaluation") == "leave_one_well_out"
                      else "well-based: whole wells held out for testing"),
            "n_train_wells": m["split"]["n_train_wells"],
            "n_test_wells": m["split"]["n_test_wells"],
            "test_accuracy": m["test"]["accuracy"],
            "test_balanced_accuracy": m["test"]["balanced_accuracy"],
            "test_macro_f1": m["test"]["macro_f1"],
        })
    return info


def run(las_path, cfg: dict, out_dir, model_path=None, verbose: bool = True) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. load + QC. The QC report is printed and saved BEFORE any calculation.
    log = load_las(las_path, cfg)
    qc = run_qc(log, cfg)
    qc_text = format_qc_report(qc.report)
    (out_dir / "qc_report.json").write_text(json.dumps(qc.report, indent=2, allow_nan=False), encoding="utf-8")
    (out_dir / "qc_report.txt").write_text(qc_text + "\n", encoding="utf-8")
    if verbose:
        print(qc_text)

    # 2. petrophysics
    df = qc.df
    curves, params = resolve_and_compute(df, cfg)
    df = pd.concat([df, curves], axis=1)

    # 3. facies
    if model_path is not None:
        bundle = load_model(model_path)
        df = pd.concat([df, predict_facies(df, bundle)], axis=1)
        metrics_path = Path(model_path).with_name("facies_metrics.json")
    else:
        metrics_path = None

    # 4. zones + summary
    zones = make_zones(df, cfg, qc.step)
    well = {"name": log.name, "file": Path(log.path).name, "depth_unit": log.depth_unit,
            "depth_reference": "measured depth as recorded in the LAS",
            "top": qc.report["processing_interval"]["top"], "base": qc.report["processing_interval"]["base"],
            "step": round(qc.step, 4), "n_samples": int(len(df)),
            "curves_used": {k: v["mnemonic"] for k, v in log.mapping.items()}}
    summary = build_summary(df, zones, params, cfg, qc.report, well, qc.step,
                            _facies_info(model_path, metrics_path, cfg))

    # 5. save. Full float precision in the CSV so chat what-if re-runs reproduce the JSON exactly.
    (out_dir / "zone_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    df.to_csv(out_dir / "curves.csv", index=False)
    try:
        from .plotting import log_plot
        log_plot(df, summary, out_dir / "log_plot.png")
    except Exception as exc:  # a plotting problem must not lose the numbers
        print(f"[warning] log plot failed: {exc}")

    if verbose:
        print("\nPARAMETERS USED (auto-picked values marked *):")
        for section in ("vsh", "toc", "porosity", "sw"):
            for key, value in params[section].items():
                star = "*" if f"{section}.{key}" in params["auto_picked"] else " "
                print(f"  {star} {section}.{key} = {value}")
        for note in params["notes"]:
            print(f"  NOTE: {note}")
        print("\n" + zone_table(summary))
        print(f"\nSaved to {out_dir}/: qc_report.txt, qc_report.json, zone_summary.json, curves.csv, log_plot.png")
    return summary


def zone_table(summary: dict) -> str:
    def f(x, nd):
        return "  n/a" if x is None else f"{x:.{nd}f}"
    lines = ["ZONE SUMMARY (averages; * = low confidence)",
             f"{'zone':<6}{'top':>9}{'base':>9}{'thk':>7}{'Vsh':>6}{'PHIT':>6}{'PHIE':>6}{'SwA':>6}{'SwS':>6}{'TOC%':>6}  facies (conf)"]
    vsh_key = ("vsh_linear_mean" if summary["parameters"].get("vsh_used_downstream") == "vsh_linear"
               else "vsh_larionov_mean")          # the Vsh the porosity, Sw and TOC steps used
    for z in summary["zones"]:
        a, fa = z["averages"], z["facies"] or {}
        fac = f"{fa.get('dominant')} ({f(fa.get('mean_confidence'), 2).strip()})" if fa else "-"
        lines.append(f"{z['name'] + ('*' if z['low_confidence'] else ''):<6}{z['top']:>9.1f}{z['base']:>9.1f}"
                     f"{z['thickness']:>7.1f}{f(a[vsh_key], 2):>6}{f(a['phit_mean'], 3):>6}"
                     f"{f(a['phie_mean'], 3):>6}{f(a['sw_archie_mean'], 2):>6}{f(a['sw_simandoux_mean'], 2):>6}"
                     f"{f(a['toc_mean_wtpct'], 2):>6}  {fac}")
    return "\n".join(lines)
