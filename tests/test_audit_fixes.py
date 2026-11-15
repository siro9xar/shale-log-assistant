"""Regression tests for the bugs found in the audit."""
import copy
import json
from types import SimpleNamespace

import lasio
import numpy as np
import pytest

from shale_assistant.config import load_config
from shale_assistant.export import zone_rows
from shale_assistant.io_las import convert_to_standard
from shale_assistant.llm import Chat
from shale_assistant.pipeline import run, zone_table
from shale_assistant.tools import Session
from shale_assistant.verify import check_numbers, format_verification, verify_report
from tests.conftest import ROOT

@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    """Synthetic well run through the real pipeline (outputs/*/curves.csv is not in git)."""
    out = tmp_path_factory.mktemp("audit")
    rng = np.random.default_rng(3)
    n = 600
    depth = 3000 + 0.1524 * np.arange(n)
    block = np.repeat([0, 1, 2], n // 3)                       # sand, lean shale, organic shale
    las = lasio.LASFile()
    las.well["WELL"].value = "SYN-AUDIT"
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
    return out


@pytest.fixture(scope="module")
def summary(run_dir):
    return json.loads((run_dir / "zone_summary.json").read_text())


# ----------------------------------------------------------------------------- verify.py
@pytest.mark.parametrize("text,bad", [
    ("Sw is .731 in the shale.", ".731"),                       # leading decimal point was skipped
    ("## 350 mD permeability\ntext", "350"),                    # a number opening a heading was skipped
    ("### 0.731 is the Sw\ntext", "0.731"),
])
def test_numbers_that_used_to_escape_the_check(summary, text, bad):
    r = check_numbers(text, [summary])
    assert not r["passed"] and [u["number"] for u in r["unverified"]] == [bad]


def test_section_numbering_is_still_ignored(summary):
    assert check_numbers("## 77. Data quality\n### 77.3) Hole\n1. first\n2. second\n", [summary])["passed"]


def test_empty_report_is_not_verified(summary):
    s = copy.deepcopy(summary)
    for z in s["zones"]:
        z["low_confidence"] = False                             # nothing left to flag: used to pass
    v = verify_report("  \n", s)
    assert not v["passed"] and v["empty_report"] and "the report is empty" in format_verification(v)
    assert verify_report("No numbers here.", s)["passed"]


# ----------------------------------------------------------------------------- chat: failed tool calls
class FakeClient:
    def __init__(self, responses):
        self.responses, self.messages = list(responses), self

    def create(self, **kwargs):
        return self.responses.pop(0)


def _text(t):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=t)], stop_reason="end_turn")


def _tool(name, **inp):
    return SimpleNamespace(content=[SimpleNamespace(type="tool_use", id="t1", name=name, input=inp)],
                           stop_reason="tool_use")


def test_number_echoed_by_a_failed_tool_call_is_not_verified(run_dir):
    cfg = load_config(ROOT / "config.yaml")
    chat = Chat(Session.load(run_dir), cfg, client=FakeClient([_tool("compute_sw", zone="Z1", rw=-0.7313),
                                                                _text("Sw would be 0.7313.")]))
    answer, check = chat.ask("guess the Sw")
    assert chat.session.calls[0]["is_error"] and "0.7313" in json.dumps(chat.session.calls[0])
    assert not check["passed"] and check["unverified"][0]["number"] == "0.7313"

    new = Session.load(run_dir).dispatch("compute_sw", {"zone": "Z1", "rw": 0.05})[0]["results"][0]["what_if"]
    chat = Chat(Session.load(run_dir), cfg, client=FakeClient([_tool("compute_sw", zone="Z1", rw=0.05),
                                                                _text(f"With Rw 0.05 ohm.m Archie Sw is "
                                                                      f"{new['sw_archie_mean']}.")]))
    assert chat.ask("what if Rw is higher?")[1]["passed"]        # a successful call still verifies


@pytest.mark.parametrize("tool,args", [
    ("compute_sw", {"zone": "Z1", "rw": float("inf")}),
    ("compute_vsh", {"zone": "Z1", "gr_shale": float("inf")}),
    ("compute_vsh", {"zone": "Z1", "gr_clean": float("nan")}),
    ("compute_toc", {"zone": "Z3", "lom": "8; drop"}),
    ("compute_toc", {"zone": "Z3", "nphi_baseline": True}),
    ("compute_porosity", {"zone": "Z1", "nphi_shale": float("-inf")}),
])
def test_tools_reject_non_finite_and_non_numeric_inputs(run_dir, tool, args):
    result, is_error = Session.load(run_dir).dispatch(tool, args)
    assert is_error and "error" in result


# ----------------------------------------------------------------------------- zone table Vsh column
def test_zone_table_shows_the_vsh_that_was_used(summary):
    s = copy.deepcopy(summary)
    z = s["zones"][1]["averages"]
    assert z["vsh_linear_mean"] != z["vsh_larionov_mean"]
    assert f"{z['vsh_larionov_mean']:.2f}" == zone_rows(s)[1][4]
    assert f"{z['vsh_larionov_mean']:.2f}" in zone_table(s).splitlines()[3]
    s["parameters"]["vsh_used_downstream"] = "vsh_linear"
    assert f"{z['vsh_linear_mean']:.2f}" == zone_rows(s)[1][4]
    assert f"{z['vsh_linear_mean']:.2f}" in zone_table(s).splitlines()[3]


# ----------------------------------------------------------------------------- io_las units
def test_gamma_ray_and_resistivity_units_are_checked():
    x = np.array([50.0, 60.0])
    for unit in ("GAPI", "API", ""):
        assert convert_to_standard("GR", x, unit)[2] is None
    v, note, warn = convert_to_standard("GR", x, "CPS")
    assert v.tolist() == x.tolist() and note == "none" and "not API" in warn
    assert convert_to_standard("RT", x, "OHMM")[2] is None and convert_to_standard("RT", x, "ohm.m")[2] is None
    assert "not ohm.m" in convert_to_standard("RT", x, "MMHO/M")[2]
