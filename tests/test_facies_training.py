"""train_facies with a user's own labelled table: well-based split, and leave-one-well-out below 5 wells."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from shale_assistant.config import load_config
from shale_assistant.facies import (REQUIRED_COLUMNS, load_model, load_training_table, predict_facies,
                                    train_classifier)
from tests.conftest import ROOT

# facies: (GR, RHOB, NPHI, log10 RT, DT) centres, well separated
CENTRES = {"Sand": (30, 2.30, 0.20, 1.3, 85), "Shale": (120, 2.55, 0.32, 0.4, 95),
           "Organic shale": (190, 2.40, 0.34, 1.4, 108)}
WELLS = ["W-1", "W-2", "W-3", "W-4", "W-5", "W-6"]
PER_BED = 60


def make_table(wells=WELLS, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k, well in enumerate(wells):
        depth = 1000.0
        for facies, (gr, rhob, nphi, lrt, dt) in CENTRES.items():
            n = PER_BED
            rows.append(pd.DataFrame({
                "WELL": well, "DEPTH": depth + 0.5 * np.arange(n),
                "GR": gr + 3 * k + rng.normal(0, 5, n),                 # small well-to-well shift
                "RHOB": rhob + rng.normal(0, 0.02, n), "NPHI": nphi + rng.normal(0, 0.01, n),
                "RT": 10 ** (lrt + rng.normal(0, 0.05, n)), "DT": dt + rng.normal(0, 1.5, n),
                "FACIES": facies}))
            depth += 0.5 * n
    return pd.concat(rows, ignore_index=True)[REQUIRED_COLUMNS]


@pytest.fixture(scope="module")
def cfg():
    c = load_config(ROOT / "config.yaml")
    c["facies"].update({"n_estimators": 25, "min_class_samples": 20})   # small and fast
    return c


def test_six_wells_hold_out_whole_wells(cfg, tmp_path):
    table = make_table()
    m = train_classifier(table, cfg, tmp_path, compare_random_split=True)
    s = m["split"]
    assert s["evaluation"] == "holdout" and "GroupShuffleSplit on WELL" in s["method"]
    assert s["n_test_wells"] == 2 and s["n_train_wells"] == 4                 # 25% of 6 wells, rounded up
    assert not set(s["train_wells"]) & set(s["test_wells"])
    assert sorted(s["train_wells"] + s["test_wells"]) == WELLS
    assert s["n_test_samples"] == 2 * 3 * PER_BED and s["n_train_samples"] == 4 * 3 * PER_BED
    assert m["test"]["accuracy"] > 0.95 and m["classes"] == sorted(CENTRES)
    assert m["cross_validation"]["grouping"] == "WELL (GroupKFold)" and m["cross_validation"]["n_splits"] == 4
    assert np.array(m["confusion_matrix"]["matrix"]).sum() == s["n_test_samples"]
    for name in ("facies_rf.joblib", "facies_metrics.json", "confusion_matrix.csv", "confusion_matrix.png"):
        assert (tmp_path / name).stat().st_size > 0
    pred = predict_facies(table, load_model(tmp_path / "facies_rf.joblib"))  # saved model is usable
    assert (pred["FACIES"] == table["FACIES"]).mean() > 0.95


def test_test_wells_from_config_are_respected(cfg, tmp_path):
    c = json.loads(json.dumps(cfg))
    c["facies"]["test_wells"] = ["W-2", "W-5"]
    s = train_classifier(make_table(), c, tmp_path)["split"]
    assert s["test_wells"] == ["W-2", "W-5"] and s["evaluation"] == "holdout"


@pytest.mark.parametrize("n_wells", [2, 4])
def test_fewer_than_five_wells_falls_back_to_leave_one_well_out(cfg, tmp_path, n_wells):
    table = make_table(WELLS[:n_wells])
    m = train_classifier(table, cfg, tmp_path)
    s = m["split"]
    assert s["evaluation"] == "leave_one_well_out" and "leave-one-well-out" in s["method"]
    assert f"{n_wells} labelled wells, fewer than 5" in s["method"]
    assert [w["well"] for w in s["per_well"]] == WELLS[:n_wells]              # every well held out once
    assert s["n_test_samples"] == len(table) == np.array(m["confusion_matrix"]["matrix"]).sum()
    pooled = sum(w["accuracy"] * w["n_samples"] for w in s["per_well"]) / len(table)
    assert m["test"]["accuracy"] == pytest.approx(pooled, abs=2e-3)           # pooled = sample-weighted mean
    assert m["cross_validation"] is None and "trained on all" in s["note"]
    model = load_model(tmp_path / "facies_rf.joblib")["model"]
    assert model.n_features_in_ == 5 and m["split"]["n_train_samples"] == len(table)


def test_leave_one_well_out_is_honest_about_an_odd_well(cfg, tmp_path):
    """A well whose shale looks different scores badly when held out - and the pooled score shows it.
    A random row split of the same table hides it (leakage)."""
    table = make_table(WELLS[:4])
    odd = (table["WELL"] == "W-4") & (table["FACIES"] == "Shale")
    table.loc[odd, ["GR", "DT"]] = np.column_stack([np.full(odd.sum(), 185.0), np.full(odd.sum(), 107.0)])
    table.loc[odd, "RT"] = 20.0                                              # W-4 shale looks organic
    m = train_classifier(table, cfg, tmp_path, compare_random_split=True)
    per_well = {w["well"]: w["accuracy"] for w in m["split"]["per_well"]}
    assert per_well["W-4"] < 0.75 and min(per_well[w] for w in WELLS[:3]) > 0.9
    assert m["random_row_split_for_comparison_only"]["accuracy"] > m["test"]["accuracy"] + 0.05


def test_one_well_and_one_class_are_refused(cfg, tmp_path):
    with pytest.raises(ValueError, match="at least 2 labelled wells"):
        train_classifier(make_table(["W-1"]), cfg, tmp_path)
    t = make_table(WELLS[:3])
    with pytest.raises(ValueError, match="fewer than 2 facies classes"):
        train_classifier(t[t["FACIES"] == "Sand"], cfg, tmp_path)


def test_class_in_only_one_well_is_reported(cfg, tmp_path):
    t = make_table(WELLS[:3])
    t = t[(t["FACIES"] != "Organic shale") | (t["WELL"] == "W-1")]
    m = train_classifier(t, cfg, tmp_path)
    assert any("Organic shale" in n and "only one well" in n for n in m["notes"])
    assert m["test"]["per_class"]["Organic shale"]["recall"] == 0.0           # never seen when W-1 is held out


def test_load_training_table(tmp_path):
    t = make_table(WELLS[:2])
    p = tmp_path / "t.csv"
    low = t.rename(columns=str.lower)
    low["extra"] = 1
    low.loc[0, "facies"] = "  "                                               # blank label -> missing
    low.to_csv(p, sep=";", index=False)                                       # lower case, semicolons
    got = load_training_table(p)
    assert set(REQUIRED_COLUMNS) <= set(got.columns) and len(got) == len(t)
    assert got["FACIES"].isna().sum() == 1 and got["GR"].dtype == float

    t.drop(columns=["DT", "FACIES"]).to_csv(p, index=False)
    with pytest.raises(ValueError, match=r"missing column\(s\) \['DT', 'FACIES'\]"):
        load_training_table(p)
    bad = t.assign(NPHI=t["NPHI"] * 100, RHOB=t["RHOB"] * 1000, DT=t["DT"] / 0.3048)
    bad.to_csv(p, index=False)
    with pytest.raises(ValueError) as err:
        load_training_table(p)
    assert all(word in str(err.value) for word in ("percent", "kg/m3", "us/m"))


@pytest.mark.parametrize("n_wells,expect", [(6, "Split: GroupShuffleSplit on WELL"), (4, "LEAVE-ONE-WELL-OUT")])
def test_script_with_own_table(tmp_path, cfg, n_wells, expect):
    table_path, cfg_path, out = tmp_path / "mine.csv", tmp_path / "cfg.yaml", tmp_path / "model"
    make_table(WELLS[:n_wells]).to_csv(table_path, index=False)
    cfg_path.write_text(yaml.safe_dump(cfg))
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "train_facies.py"), "--table", str(table_path),
                           "--config", str(cfg_path), "--out", str(out), "--compare-random-split"],
                          capture_output=True, text=True, check=True)
    assert expect in proc.stdout and "Confusion matrix (rows = true" in proc.stdout
    if n_wells == 4:
        assert "fewer than the 5 needed" in proc.stdout and "held-out well" in proc.stdout
        assert "W-4" in proc.stdout and "trained on all 4 wells" in proc.stdout
    else:
        assert "LEAVE-ONE-WELL-OUT" not in proc.stdout and "test : 2 wells" in proc.stdout
    assert json.loads((out / "facies_metrics.json").read_text())["split"]["n_test_wells"] in (2, 4)
