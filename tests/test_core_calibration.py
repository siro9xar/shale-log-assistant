"""calibrate_with_core: synthetic core where the right LOM, baseline offset and depth shift are known."""
import json
import subprocess
import sys

import lasio
import numpy as np
import pandas as pd
import pytest

from shale_assistant.config import load_config
from shale_assistant.core_calibration import (_toc_from_dlogr, calibrate, format_report, read_core, stats,
                                              toc_adjustments)
from shale_assistant.petrophysics import toc_passey
from shale_assistant.pipeline import run
from tests.conftest import ROOT

LOM, MIN_VSH, RT_B, DT_B = 10.0, 0.4, 1.0, 80.0
STEP = 0.1524


@pytest.fixture(scope="module")
def logs():
    depth = 1000.0 + STEP * np.arange(700)                       # 1000 - 1106.5
    dlogr = 0.6 + 0.4 * np.sin(depth / 3.0)                      # always > 0, varies with depth
    vsh = np.where((depth > 1060) & (depth < 1070), 0.1, 0.8)    # clean band: TOC masked to 0
    bad = (depth > 1080) & (depth < 1085)
    df = pd.DataFrame({"DEPTH": depth, "DLOGR": dlogr, "VSH": vsh, "QC_BADHOLE": bad,
                       "PHIT": 0.12 + 0.05 * np.sin(depth / 2.0), "ZONE": "Z1"})
    df["PHIE"] = df["PHIT"] - 0.02
    df["TOC"] = _toc_from_dlogr(dlogr, vsh, LOM, MIN_VSH)
    return df


SUMMARY = {"well": {"depth_unit": "M"},
           "parameters": {"toc": {"method": "sonic", "lom": LOM, "min_vsh": MIN_VSH, "rt_baseline": RT_B,
                                  "dt_baseline": DT_B, "rhob_baseline": 2.45, "nphi_baseline": 0.3,
                                  "baseline_interval": None},
                          "porosity": {"rho_matrix": 2.65, "rho_fluid": 1.0, "phit_method": "density"}}}


def _core(logs, idx, toc=None, phi=None, shift=0.0):
    """Core samples at log sample idx; values default to the log values (perfect core)."""
    sub = logs.iloc[idx]
    return pd.DataFrame({"DEPTH": sub["DEPTH"].to_numpy() - shift,
                         "TOC_WTPCT": sub["TOC"].to_numpy() if toc is None else toc,
                         "PHI": sub["PHIT"].to_numpy() if phi is None else phi})


GOOD = np.arange(0, 380, 7)                                      # 1000-1058 m: shale, good hole


def test_perfect_core_has_no_bias(logs):
    r = calibrate(_core(logs, GOOD), logs, SUMMARY)
    for key in ("toc", "porosity"):
        s = r[key]["stats"]
        assert s["n"] == len(GOOD) and s["bias"] == pytest.approx(0) and s["rmse"] == pytest.approx(0)
        assert s["r2"] == pytest.approx(1.0)


def test_stats_formulas():
    s = stats(np.array([2.0, 3.0, 5.0]), np.array([1.0, 3.0, 4.0]))
    assert s["bias"] == pytest.approx(2 / 3)                     # residuals 1, 0, 1
    assert s["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert s["r2"] == pytest.approx(1 - 2 / (14 / 3))            # SS_tot of core = 14/3


def test_lom_that_removes_a_proportional_error(logs):
    true_lom = 8.0                                               # core TOC was really made with LOM 8
    toc = toc_passey(logs["DLOGR"].iloc[GOOD], true_lom)
    r = calibrate(_core(logs, GOOD, toc=toc), logs, SUMMARY)
    assert r["toc"]["stats"]["bias"] < 0                         # log (LOM 10) under-reads
    a = r["toc"]["adjustments"]
    assert a["lom"]["lom"] == pytest.approx(true_lom, abs=1e-6)
    assert a["lom"]["rmse_after"] == pytest.approx(0, abs=1e-9) and a["lom"]["plausible"]
    assert a["baseline"]["solvable"] and a["baseline"]["rmse_after"] > 0.05      # offset cannot fix a scale error
    assert "lower RMSE with the LOM change" in format_report(r)


def test_baseline_that_removes_an_offset(logs):
    d = 0.2                                                      # core TOC made with DlogR + 0.2
    toc = toc_passey(logs["DLOGR"].iloc[GOOD] + d, LOM)
    a = calibrate(_core(logs, GOOD, toc=toc), logs, SUMMARY)["toc"]["adjustments"]
    b = a["baseline"]
    assert b["dlogr_offset"] == pytest.approx(d, abs=1e-6)
    assert b["rt_baseline"] == pytest.approx(RT_B * 10 ** -d, rel=1e-6)          # 0.631 ohm.m
    assert b["dt_baseline"] == pytest.approx(DT_B - d / 0.02, abs=1e-4)          # 70 us/ft
    assert b["rmse_after"] == pytest.approx(0, abs=1e-9)
    assert a["lom"]["rmse_after"] > b["rmse_after"]


@pytest.mark.parametrize("method,key,coef", [("density", "rhob_baseline", -2.5), ("neutron", "nphi_baseline", 4.0)])
def test_baseline_in_other_porosity_units(logs, method, key, coef):
    toc = toc_passey(logs["DLOGR"].iloc[GOOD] + 0.1, LOM)
    p = {**SUMMARY["parameters"]["toc"], "method": method}
    m = calibrate(_core(logs, GOOD, toc=toc), logs, SUMMARY)       # only for the matched table
    a = toc_adjustments(pd.DataFrame({"CORE_TOC": toc, "DLOGR": logs["DLOGR"].iloc[GOOD].to_numpy(),
                                      "VSH": logs["VSH"].iloc[GOOD].to_numpy()}), p)
    assert a["baseline"][key] == pytest.approx(p[key] - 0.1 / coef, abs=1e-5)
    assert m["toc"]["stats"]["n"] == len(GOOD)


def test_depth_shift(logs):
    core = _core(logs, GOOD, shift=1.5)                          # core depths read 1.5 m shallow
    off = calibrate(core, logs, SUMMARY)
    on = calibrate(core, logs, SUMMARY, depth_shift=1.5)
    assert on["toc"]["stats"]["rmse"] == pytest.approx(0, abs=1e-12)
    assert on["toc"]["stats"]["r2"] == pytest.approx(1.0)
    assert off["toc"]["stats"]["rmse"] > 0.1 and off["toc"]["stats"]["r2"] < 0.9


def test_porosity_bias_and_curve_choice(logs):
    phi = logs["PHIT"].iloc[GOOD].to_numpy() - 0.02
    s = calibrate(_core(logs, GOOD, phi=phi), logs, SUMMARY)["porosity"]["stats"]
    assert s["bias"] == pytest.approx(0.02) and s["rmse"] == pytest.approx(0.02)
    s = calibrate(_core(logs, GOOD, phi=phi), logs, SUMMARY, phi_curve="PHIE")["porosity"]["stats"]
    assert s["bias"] == pytest.approx(0.0, abs=1e-12)


def test_empty_values_unmatched_and_bad_hole(logs):
    core = _core(logs, np.r_[GOOD[:10], 535, 540])              # 535/540 fall in the bad-hole band
    core.loc[0:3, "TOC_WTPCT"] = np.nan                          # TOC missing on 4 rows
    core.loc[4:5, "PHI"] = np.nan                                # PHI missing on 2 rows
    core = pd.concat([core, pd.DataFrame({"DEPTH": [900.0, 1000.0 + 10.5 * STEP], "TOC_WTPCT": [2, 2],
                                          "PHI": [0.1, 0.1]})], ignore_index=True)  # above logs / between samples
    r = calibrate(core, logs, SUMMARY, tolerance=0.05)          # 0.076 m to the nearest sample -> unmatched
    mt = r["matching"]
    assert mt["core_samples"] == 14 and mt["unmatched"] == 2 and mt["matched_in_bad_hole"] == 2
    assert r["toc"]["stats"]["n"] == 10 - 4 and r["porosity"]["stats"]["n"] == 10 - 2
    r2 = calibrate(core, logs, SUMMARY, tolerance=0.05, include_badhole=True)
    assert r2["toc"]["stats"]["n"] == 8 and r2["porosity"]["stats"]["n"] == 10


def test_core_in_clean_rock_is_reported_as_unfixable(logs):
    idx = np.r_[GOOD[:20], np.arange(400, 455, 6)]               # 400-455 = clean band (TOC masked)
    toc = logs["TOC"].iloc[idx].to_numpy().copy()
    toc[20:] = 1.0                                               # core finds 1 wt% where the log says 0
    a = calibrate(_core(logs, idx, toc=toc), logs, SUMMARY)["toc"]["adjustments"]
    assert "Vsh < toc.min_vsh" in a["note_masked"]


def test_read_core(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("depth,toc_wtpct\n1000.5,2.1\n1001.0,\n")       # lower case, PHI column missing
    c = read_core(p)
    assert list(c.columns[:3]) == ["DEPTH", "TOC_WTPCT", "PHI"] and c["PHI"].isna().all()
    p.write_text("DEPTH,TOC_WTPCT,PHI\n1000,1,12.5\n")           # percent porosity
    with pytest.raises(ValueError, match="percent"):
        read_core(p)
    p.write_text("DEPTH,TOC_WTPCT,PHI\n1000,-0.2,0.1\n1001,1.0,-0.01\nabc,1,0.1\n")   # impossible values
    c = read_core(p)
    assert len(c) == 2 and c["TOC_WTPCT"].isna().sum() == 1 and c["PHI"].isna().sum() == 1
    w = " | ".join(c.attrs["warnings"])
    assert "negative TOC_WTPCT" in w and "negative PHI" in w and "without a numeric DEPTH" in w


# ----------------------------------------------------------------------------- end to end, real pipeline
@pytest.fixture(scope="module")
def pipeline_out(tmp_path_factory):
    out = tmp_path_factory.mktemp("core")
    rng = np.random.default_rng(5)
    n = 600
    depth = 3000 + STEP * np.arange(n)
    block = np.repeat([0, 1, 2], n // 3)                         # sand, lean shale, organic shale
    las = lasio.LASFile()
    las.well["WELL"].value = "SYN-CORE"
    las.append_curve("DEPT", depth, unit="M")
    for name, data, unit in (("GR", np.choose(block, [30, 110, 180]) + rng.normal(0, 3, n), "GAPI"),
                             ("RHOB", np.choose(block, [2.35, 2.55, 2.42]) + rng.normal(0, 0.01, n), "G/CC"),
                             ("NPHI", np.choose(block, [18, 30, 33]) + rng.normal(0, 0.5, n), "%"),
                             ("RT", 10 ** (np.choose(block, [1.3, 0.3, 1.2]) + rng.normal(0, 0.02, n)), "OHMM"),
                             ("DT", np.choose(block, [80, 90, 105]) + rng.normal(0, 0.5, n), "US/F"),
                             ("CALI", np.full(n, 8.6), "IN")):
        las.append_curve(name, data, unit=unit)
    las.write(str(out / "syn.las"), version=2.0)
    cfg = load_config(ROOT / "config.yaml")
    cfg["toc"]["baseline_interval"] = [3032.0, 3037.0]
    cfg["zonation"]["max_zones"] = 3
    run(out / "syn.las", cfg, out, model_path=None, verbose=False)
    return out, cfg


def test_recomputed_toc_matches_the_pipeline(pipeline_out):
    out, cfg = pipeline_out
    c = pd.read_csv(out / "curves.csv")
    summary = json.loads((out / "zone_summary.json").read_text())
    t = summary["parameters"]["toc"]
    np.testing.assert_allclose(_toc_from_dlogr(c["DLOGR"], c["VSH"], t["lom"], t["min_vsh"]), c["TOC"],
                               equal_nan=True)


def test_script_end_to_end_and_config_untouched(pipeline_out, tmp_path):
    out, _ = pipeline_out
    c = pd.read_csv(out / "curves.csv")
    organic = c[c["TOC"] > 0].iloc[::10]
    core = pd.DataFrame({"DEPTH": organic["DEPTH"] - 0.6096,      # core logged 2 samples shallow
                         "TOC_WTPCT": organic["TOC"] * 1.5, "PHI": organic["PHIT"]})
    core_path = tmp_path / "core.csv"
    core.to_csv(core_path, index=False)
    cfg_path = ROOT / "config.yaml"
    before, mtime = cfg_path.read_bytes(), cfg_path.stat().st_mtime_ns
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "calibrate_with_core.py"), "--core", str(core_path),
                           "--out", str(out), "--depth-shift", "0.6096"], capture_output=True, text=True, check=True)
    assert cfg_path.read_bytes() == before and cfg_path.stat().st_mtime_ns == mtime
    rep = json.loads((out / "calibration_report.json").read_text())          # strict JSON (no NaN)
    assert (out / "calibration_crossplot.png").stat().st_size > 0
    assert rep["porosity"]["stats"]["rmse"] == pytest.approx(0, abs=1e-9)    # shift found the right samples
    lom = rep["toc"]["adjustments"]["lom"]
    assert lom["lom"] == pytest.approx(10.0 - np.log10(1.5) / 0.1688, abs=1e-4)   # x1.5 TOC = LOM -1.04
    assert "LOM 10.0 ->" in proc.stdout and "config.yaml is NOT changed" in proc.stdout
