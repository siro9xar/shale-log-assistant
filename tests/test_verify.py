import pandas as pd

from shale_assistant.verify import check_numbers, check_ranges, extract_numbers, verify_report

SUMMARY = {
    "well": {"name": "15/9-19 SR", "file": "15-9-19_SR_COMP.LAS"},
    "parameters": {"porosity": {"phi_max": 0.45}, "sw": {"rw": 0.02}},
    "zones": [
        {"name": "Z1", "top": 4304.43, "low_confidence": False, "facies": None,
         "averages": {"vsh_larionov_mean": 0.99, "sw_archie_mean": 0.644, "phit_mean": 0.129, "toc_mean_wtpct": 4.55}},
        {"name": "Z2", "top": 4310.38, "low_confidence": True, "facies": None,
         "averages": {"vsh_larionov_mean": 0.88, "sw_archie_mean": 1.0, "phit_mean": 0.024, "toc_mean_wtpct": 1.28}},
    ],
}


def test_extract_numbers_ignores_labels():
    vals = [n["value"] for n in extract_numbers("Z3 has log10 P95 of 0.35 and -1.2, range 3550-3600 m; 4,303.5 m")]
    assert vals == [0.35, -1.2, 3550.0, 3600.0, 4303.5]


def test_good_report_passes():
    report = ("# Report: 15/9-19 SR\n## Zone results\nZ1 starts at 4304.43 m, Vsh 0.99, Sw 0.644, TOC 4.55 wt%.\n"
              "## Low-confidence zones\n- Z2: Sw 1.0 with Rw 0.02 ohm.m, treat with caution.\n")
    v = verify_report(report, SUMMARY)
    assert v["passed"], v


def test_invented_rounded_and_percent_numbers_fail():
    report = ("Z1 has Sw 0.64 and porosity of 12.9% and a permeability of 250 nD.\n"
              "## Low-confidence zones\nZ2 is low confidence.")
    v = verify_report(report, SUMMARY)
    reasons = {u["number"]: u["reason"] for u in v["numbers"]["unverified"]}
    assert not v["passed"]
    assert "rounded" in reasons["0.64"] and "percentage" in reasons["12.9%"] and "not in the JSON" in reasons["250"]
    assert check_numbers("Sw 0.64", [SUMMARY], allow_rounding=True)["passed"]


def test_unflagged_low_confidence_zone_fails():
    v = verify_report("Z1 and Z2 both look good with Vsh 0.99 and 0.88.", SUMMARY)
    assert v["low_confidence_flags"]["not_flagged_in_report"] == ["Z2"] and not v["passed"]


def test_range_check_catches_bad_values():
    bad = {**SUMMARY, "zones": [{**SUMMARY["zones"][0], "averages": {"sw_archie_mean": 1.3, "vsh_larionov_mean": -0.1}}]}
    assert len(check_ranges(bad)["violations"]) == 2
    curves = pd.DataFrame({"VSH": [0.1, 1.2], "SW_ARCHIE": [0.5, 0.9]})
    r = check_ranges(SUMMARY, curves)
    assert not r["passed"] and r["violations"][0]["quantity"] == "VSH"
