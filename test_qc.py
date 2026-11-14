import numpy as np
import pandas as pd
import pytest

from shale_assistant.config import load_config
from shale_assistant.io_las import WellLog, convert_to_standard
from shale_assistant.qc import depth_qc, flatline_mask, hampel_spikes, run_qc
from tests.conftest import ROOT


def test_unit_conversions():
    v, note, warn = convert_to_standard("NPHI", np.array([25.0, 30.0]), "%")
    assert v.tolist() == [0.25, 0.30] and warn is None
    v, _, _ = convert_to_standard("RHOB", np.array([2450.0]), "KG/M3")
    assert v[0] == pytest.approx(2.45)
    v, _, _ = convert_to_standard("DT", np.array([300.0]), "US/M")
    assert v[0] == pytest.approx(91.44)                       # 300 us/m * 0.3048 m/ft
    v, _, _ = convert_to_standard("CALI", np.array([215.9]), "MM")
    assert v[0] == pytest.approx(8.5)
    v, _, warn = convert_to_standard("NPHI", np.array([25.0, 30.0]), "")   # unit missing: guessed + warned
    assert v[0] == pytest.approx(0.25) and warn


def test_spike_detector_finds_spike_but_keeps_real_bed():
    x = np.full(200, 60.0) + np.random.default_rng(1).normal(0, 1, 200)
    x[50] = 400.0                                              # 1-sample spike
    x[120:140] += 80.0                                         # 20-sample bed: real geology
    mask = hampel_spikes(x, window=11, nsigma=5.0, min_scale=2.0, max_width=2)
    assert mask[50] and mask.sum() == 1


def test_flatline():
    x = np.r_[np.random.default_rng(2).normal(80, 5, 100), np.full(40, 40.0)]
    m = flatline_mask(x, 30)
    assert m[100:].all() and not m[:100].any()


def test_depth_qc_flags_gap_and_header_mismatch():
    d = np.r_[np.arange(1000, 1010, 0.5), np.arange(1020, 1030, 0.5)]
    rep = depth_qc(d, {"STRT": 1000.0, "STOP": 1029.5, "STEP": 0.25}, 0.05)
    assert rep["n_gaps"] == 1 and not rep["header_matches_data"]


def _synthetic_log():
    n = 400
    depth = 2000 + 0.5 * np.arange(n)
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"DEPTH": depth, "GR": 80 + rng.normal(0, 3, n), "RHOB": 2.5 + rng.normal(0, 0.01, n),
                       "NPHI": 0.2 + rng.normal(0, 0.005, n), "RT": 10 ** (1 + rng.normal(0, 0.02, n)),
                       "DT": 85 + rng.normal(0, 1, n), "CALI": np.full(n, 8.6)})
    df.loc[100:139, "CALI"] = 12.0                             # 40-sample washout
    df.loc[200, "RHOB"] = 1.0                                  # outside physical range
    df.loc[250, "GR"] = 600.0                                  # spike
    df.loc[300:309, "RT"] = np.nan                             # gap
    mapping = {c: {"mnemonic": c, "unit_in_file": "", "standard_unit": "", "conversion": "none"}
               for c in df.columns if c != "DEPTH"}
    return WellLog("SYN-1", "synthetic.las", "M", df, mapping,
                   {"STRT": 2000.0, "STOP": float(depth[-1]), "STEP": 0.5, "NULL": -999.25})


def test_run_qc_flags_everything_before_calculation():
    cfg = load_config(ROOT / "config.yaml")
    res = run_qc(_synthetic_log(), cfg)
    rep = res.report
    assert rep["washout"]["samples"] == 40
    assert rep["out_of_range"]["RHOB"]["samples"] == 1
    assert rep["spikes"]["GR"]["samples"] >= 1
    assert rep["missing_data"]["RT"]["missing_samples"] == 10
    assert res.df.loc[100:139, "RHOB"].isna().all()            # washed-out density is not used
    assert res.df["QC_BADHOLE"].sum() == 40
    assert np.isnan(res.df.loc[250, "GR"])
