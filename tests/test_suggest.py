"""suggest_parameters: synthetic wells where the right Rw, baseline interval and matrix density are known."""
import json
import subprocess
import sys

import lasio
import numpy as np
import pandas as pd
import pytest

from shale_assistant.config import load_config
from shale_assistant.pipeline import run
from shale_assistant.suggest import (nearest_mineral, passey_candidates, run_suggestions, suggest_matrix_density,
                                     suggest_rw)
from tests.conftest import ROOT

RW, A, M = 0.05, 1.0, 2.0
N = 60                                     # samples per bed, 0.5 m step -> 30 m beds


def _bed(name, zone, vsh, phi, rt, dt, rho_ma=2.65, badhole=False, nphi=None):
    rhob = rho_ma * (1 - phi) + phi * 1.0                # liquid-filled, exact
    return pd.DataFrame({"BED": name, "ZONE": zone, "VSH": vsh, "PHIT": phi, "RT": rt, "DT": dt,
                         "RHOB": rhob, "NPHI": phi if nphi is None else nphi,
                         "PHID": (2.65 - rhob) / 1.65, "QC_BADHOLE": badhole}, index=range(N))


@pytest.fixture(scope="module")
def well():
    beds = [
        _bed("wet sand", "Z1", 0.05, 0.25, RW / 0.25 ** 2, 90.0),            # Rwa = Rw exactly
        _bed("lean shale", "Z2", 0.90, 0.10, 2.0, 95.0, nphi=0.30),          # the correct Passey baseline
        _bed("oil sand", "Z3", 0.05, 0.22, 25.0, 88.0),                      # Rwa = 25 * 0.22^2 = 1.21
        _bed("washed lean shale", "Z4", 0.90, 0.10, 1.5, 100.0, badhole=True, nphi=0.30),  # lower RT, bad hole
        _bed("organic shale", "Z5", 0.90, 0.12, 15.0, 110.0, nphi=0.33),
        _bed("limestone", "Z6", 0.05, 0.10, 50.0, 55.0, rho_ma=2.71),
        _bed("dolomite", "Z7", 0.05, 0.08, 80.0, 50.0, rho_ma=2.87),
        # high-GR (feldspar) WET sandstone: GR says shale, lowest RT of all, but no N-D separation
        _bed("hot sand", "Z8", 0.70, 0.25, 0.6, 90.0),
    ]
    df = pd.concat(beds, ignore_index=True)
    df.insert(0, "DEPTH", 1000.0 + 0.5 * np.arange(len(df)))
    return df


def _bed_range(df, name):
    d = df.loc[df["BED"] == name, "DEPTH"]
    return d.min(), d.max()


def test_rw_is_the_low_end_of_rwa_and_shows_both_sands(well):
    r = suggest_rw(well, A, M)
    assert r["suggested_rw"] == pytest.approx(RW)                         # P10 = the wet sand
    assert r["rwa"]["p50"] == pytest.approx((RW + 1.21) / 2)              # 50/50 wet/oil -> midpoint
    assert r["rwa"]["p90"] == pytest.approx(1.21)
    assert [i["zone"] for i in r["intervals"]] == ["Z1", "Z3"]            # shale, tight carbonates excluded
    assert r["intervals"][0]["rwa_p50"] == pytest.approx(RW)
    assert r["intervals"][1]["rwa_p50"] == pytest.approx(1.21)
    assert (r["intervals"][0]["top"], r["intervals"][0]["base"]) == _bed_range(well, "wet sand")


def test_rw_per_zone_marks_the_oil_sand(well):
    r = {z["zone"]: z for z in suggest_rw(well, A, M)["by_zone"]}
    assert not r["Z1"]["likely_hydrocarbon"] and r["Z3"]["likely_hydrocarbon"]


def test_rw_ignores_bad_hole_and_thin_beds(well):
    df = well.copy()
    df.loc[df["BED"] == "wet sand", "QC_BADHOLE"] = True
    r = suggest_rw(df, A, M)
    assert r["suggested_rw"] == pytest.approx(1.21)                       # only the oil sand is left
    # ... and the result says loudly that this is probably not Rw
    assert any("one zone (Z3)" in w and "NOT Rw" in w for w in r["warnings"])
    assert any("fewer than 100" in w for w in r["warnings"])
    df = well.copy()
    df.loc[(df["BED"] == "wet sand") & (df.index % 2 == 0), "VSH"] = 0.5  # cut the wet sand into 0.5 m beds
    assert all(i["zone"] != "Z1" for i in suggest_rw(df, A, M, min_thickness=1.0)["intervals"])


def test_no_clean_rock_gives_no_rw(well):
    r = suggest_rw(well.assign(VSH=0.9), A, M)
    assert r["suggested_rw"] is None and r["n_samples"] == 0 and r["intervals"] == []


@pytest.mark.parametrize("method,key", [("sonic", "dt_p50"), ("density", "rhob_p50")])
def test_passey_baseline_is_the_lean_shale_in_good_hole(well, method, key):
    r = passey_candidates(well, method)
    best = r["suggested"]
    assert (best["top"], best["base"]) == _bed_range(well, "lean shale")
    assert best["rt_p50"] == pytest.approx(2.0) and best["zone"] == "Z2"
    washed = next(c for c in r["candidates"] if c["zone"] == "Z4")
    assert not washed["usable"] and "rank" not in washed                  # lower RT, but rejected: bad hole
    organic = next(c for c in r["candidates"] if c["zone"] == "Z5")
    assert organic["rank"] == 2 and organic["rt_p50"] > best["rt_p50"]
    assert key in best
    assert all(c["zone"] != "Z8" for c in r["candidates"])               # hot sand is not shale


def test_without_nd_separation_a_hot_sand_would_win(well):
    """Why the separation test exists: GR alone ranks the wet hot sand first (lowest RT)."""
    r = passey_candidates(well.drop(columns="PHID"))
    assert r["suggested"]["zone"] == "Z8" and r["criteria"]["nd_separation_min"] is None


def test_matrix_density_per_zone(well):
    r = {z["zone"]: z for z in suggest_matrix_density(well, rho_fluid=1.0)["zones"]}
    assert r["Z1"]["rhoma"]["p50"] == pytest.approx(2.65) and r["Z1"]["nearest_mineral"].startswith("quartz")
    assert r["Z6"]["rhoma"]["p50"] == pytest.approx(2.71) and r["Z6"]["nearest_mineral"].startswith("calcite")
    assert r["Z7"]["rhoma"]["p50"] == pytest.approx(2.87) and r["Z7"]["nearest_mineral"] == "dolomite"
    assert r["Z2"]["suggested_rho_matrix"] is None and "fewer than" in r["Z2"]["note"]   # shale is not clean


def test_matrix_density_uses_the_neutron_offset(well):
    # a limestone-scaled neutron reading 0.04 low in sandstone: without the offset rhoma reads low
    df = well.assign(NPHI=well["NPHI"] - 0.04)
    low = {z["zone"]: z for z in suggest_matrix_density(df, 1.0, 0.0)["zones"]}["Z1"]["rhoma"]["p50"]
    fixed = {z["zone"]: z for z in suggest_matrix_density(df, 1.0, 0.04)["zones"]}["Z1"]["rhoma"]["p50"]
    assert low < 2.62 and fixed == pytest.approx(2.65)


def test_nearest_mineral():
    assert nearest_mineral(2.66).startswith("quartz") and nearest_mineral(2.97) == "anhydrite"


# ----------------------------------------------------------------------------- end to end, real pipeline
@pytest.fixture(scope="module")
def pipeline_out(tmp_path_factory):
    """Synthetic LAS: water sand with known Rw, lean shale, organic shale, through the real pipeline."""
    out = tmp_path_factory.mktemp("suggest")
    rng = np.random.default_rng(11)
    n = 900
    depth = 2000 + 0.1524 * np.arange(n)
    block = np.repeat([0, 1, 2], n // 3)                     # wet sand, lean shale, organic shale
    phi_sand = 0.25
    rhob = np.choose(block, [2.65 * (1 - phi_sand) + phi_sand, 2.55, 2.42]) + rng.normal(0, 0.004, n)
    phid = (2.65 - rhob) / 1.65
    rt = np.where(block == 0, RW / phid ** 2, np.choose(block, [1.0, 2.0, 16.0])) * 10 ** rng.normal(0, 0.005, n)
    las = lasio.LASFile()
    las.well["WELL"].value = "SYN-RW"
    las.append_curve("DEPT", depth, unit="M")
    for name, data, unit in (("GR", np.choose(block, [25, 110, 180]) + rng.normal(0, 2, n), "GAPI"),
                             ("RHOB", rhob, "G/CC"),
                             ("NPHI", 100 * np.choose(block, [phi_sand, 0.30, 0.33]) + rng.normal(0, 0.3, n), "%"),
                             ("RT", rt, "OHMM"),
                             ("DT", np.choose(block, [90, 95, 110]) + rng.normal(0, 0.4, n), "US/F"),
                             ("CALI", 8.6 + rng.normal(0, 0.02, n), "IN")):
        las.append_curve(name, data, unit=unit)
    las.write(str(out / "syn.las"), version=2.0)
    cfg = load_config(ROOT / "config.yaml")
    cfg["toc"]["baseline_interval"] = None
    cfg["zonation"]["max_zones"] = 3
    run(out / "syn.las", cfg, out, model_path=None, verbose=False)
    return out, depth, block


def test_end_to_end_recovers_rw_and_baseline(pipeline_out):
    out, depth, block = pipeline_out
    curves = pd.read_csv(out / "curves.csv")
    summary = json.loads((out / "zone_summary.json").read_text())
    res, text = run_suggestions(curves, summary, plot_path=out / "x.png")
    assert res["rw"]["rwa"]["p50"] == pytest.approx(RW, rel=0.03)
    assert res["rw"]["suggested_rw"] == pytest.approx(RW, rel=0.05)
    best = res["passey"]["suggested"]
    lean = depth[block == 1]
    assert best["top"] >= lean.min() - 1 and best["base"] <= lean.max() + 1
    assert res["matrix_density"]["zones"][0]["suggested_rho_matrix"] == pytest.approx(2.65, abs=0.01)
    assert "SUGGESTED Rw" in text and "config.yaml is NOT changed" in text
    assert (out / "x.png").stat().st_size > 0


def test_script_prints_and_never_writes_config(pipeline_out):
    out, *_ = pipeline_out
    cfg_path = ROOT / "config.yaml"
    before, mtime = cfg_path.read_bytes(), cfg_path.stat().st_mtime_ns
    files_before = {p.name for p in out.iterdir()}
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "suggest_parameters.py"), "--out", str(out)],
                          capture_output=True, text=True, check=True)
    assert "SUGGESTED Rw" in proc.stdout and "PASSEY BASELINE" in proc.stdout and "MATRIX DENSITY" in proc.stdout
    assert cfg_path.read_bytes() == before and cfg_path.stat().st_mtime_ns == mtime
    assert {p.name for p in out.iterdir()} - files_before <= {"passey_baseline_crossplot.png"}
