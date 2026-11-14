"""End-to-end on a synthetic LAS: pipeline -> JSON -> what-if tools -> (mock) LLM report and chat.
No API key is needed: the Anthropic client is replaced by a scripted fake."""
import json
from types import SimpleNamespace

import lasio
import numpy as np
import pandas as pd
import pytest

from shale_assistant import petrophysics as pp
from shale_assistant.config import load_config
from shale_assistant.llm import Chat, generate_report
from shale_assistant.pipeline import run
from shale_assistant.tools import Session
from shale_assistant.verify import check_ranges
from tests.conftest import ROOT


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    rng = np.random.default_rng(7)
    n = 600
    depth = 3000 + 0.1524 * np.arange(n)
    block = np.repeat([0, 1, 2], n // 3)                       # sand, lean shale, organic shale
    gr = np.choose(block, [30, 110, 180]) + rng.normal(0, 3, n)
    rhob = np.choose(block, [2.35, 2.55, 2.42]) + rng.normal(0, 0.01, n)
    nphi = np.choose(block, [18, 30, 33]) + rng.normal(0, 0.5, n)          # percent in the file
    rt = 10 ** (np.choose(block, [1.3, 0.3, 1.2]) + rng.normal(0, 0.02, n))
    dt = np.choose(block, [80, 90, 105]) + rng.normal(0, 0.5, n)
    cali = np.full(n, 8.6)
    cali[250:300] = 11.5                                       # washout inside the lean shale
    las = lasio.LASFile()
    las.well["WELL"].value = "SYN-1"
    las.append_curve("DEPT", depth, unit="M")
    for name, data, unit in (("GR", gr, "GAPI"), ("RHOB", rhob, "G/CC"), ("NPHI", nphi, "%"),
                             ("RT", rt, "OHMM"), ("DT", dt, "US/F"), ("CALI", cali, "IN")):
        las.append_curve(name, data, unit=unit)
    path = out / "syn.las"
    las.write(str(path), version=2.0)
    cfg = load_config(ROOT / "config.yaml")
    cfg["toc"]["baseline_interval"] = [3032.0, 3037.0]         # un-washed part of the lean shale
    cfg["zonation"]["max_zones"] = 3
    summary = run(path, cfg, out, model_path=None, verbose=False)
    return out, cfg, summary


def test_summary_is_sane(outputs):
    out, cfg, summary = outputs
    assert [z["name"] for z in summary["zones"]] == ["Z1", "Z2", "Z3"]
    z1, z2, z3 = summary["zones"]
    assert z1["top"] == pytest.approx(3000.0, abs=0.01) and abs(z1["base"] - 3030.48) < 1.0
    assert z3["averages"]["toc_mean_wtpct"] > 3 > z2["averages"]["toc_mean_wtpct"]      # organic shale found
    assert z1["averages"]["toc_mean_wtpct"] == 0.0                                      # sand: TOC masked
    assert z1["averages"]["phit_mean"] == pytest.approx((2.65 - 2.35) / 1.65, abs=0.005)
    assert z2["low_confidence"] and any("WASHOUT" in r for r in z2["low_confidence_reasons"])
    assert summary["qc_summary"]["unit_conversions"]["NPHI"].startswith("percent")
    assert check_ranges(summary, pd.read_csv(out / "curves.csv"))["passed"]
    json.dumps(summary, allow_nan=False)                                                # strict JSON, no NaN


def test_tool_base_case_reproduces_json_and_what_if_matches_direct_calculation(outputs):
    out, cfg, summary = outputs
    s = Session.load(out)
    res, err = s.dispatch("compute_sw", {"zone": "Z1"})
    assert not err
    for key, value in res["results"][0]["base_case"].items():
        assert value == summary["zones"][0]["averages"][key], key      # re-run from curves.csv == pipeline

    res, err = s.dispatch("compute_sw", {"zone": "Z1", "rw": 0.05})
    assert not err and res["changed_parameters"]["sw.rw"] == {"base": 0.02, "what_if": 0.05}
    r = res["results"][0]
    sub = s.curves[s.curves["ZONE"] == "Z1"]
    direct = np.clip(pp.sw_archie(sub["RT"].to_numpy(), sub["PHIT"].to_numpy(), 0.05), 0, 1)
    assert r["what_if"]["sw_archie_mean"] == round(float(np.nanmean(direct)), 3)
    # Archie with n = 2: Sw scales with sqrt(Rw) -> sqrt(0.05 / 0.02) = 1.5811 while no sample is clipped
    assert r["what_if"]["sw_archie_mean"] / r["base_case"]["sw_archie_mean"] == pytest.approx(1.5811, rel=0.01)
    assert r["change_what_if_minus_base"]["sw_archie_mean"] == pytest.approx(
        r["what_if"]["sw_archie_mean"] - r["base_case"]["sw_archie_mean"], abs=1e-9)


def test_other_tools_and_errors(outputs):
    out, *_ = outputs
    s = Session.load(out)
    por, err = s.dispatch("compute_porosity", {"zone": "all", "rho_matrix": 2.71})
    assert not err and len(por["results"]) == 3
    assert all(r["what_if"]["phit_mean"] > r["base_case"]["phit_mean"] for r in por["results"])
    toc, err = s.dispatch("compute_toc", {"zone": "Z3", "lom": 8})
    assert not err and toc["results"][0]["what_if"]["toc_mean_wtpct"] > toc["results"][0]["base_case"]["toc_mean_wtpct"]
    vsh, err = s.dispatch("compute_vsh", {"zone": "Z1", "method": "linear"})
    assert not err
    assert s.dispatch("rank_zones", {"metric": "toc_mean_wtpct"})[0]["ranking"][0]["zone"] == "Z3"
    assert s.dispatch("compute_sw", {"zone": "Z99", "rw": 0.05})[1]            # unknown zone -> error, not a guess
    assert s.dispatch("compute_sw", {"zone": "Z1", "rw": -1})[1]
    assert s.dispatch("compute_sw", {"zone": "Z1", "rw": "0.05; drop"})[1]


# ----------------------------------------------------------------------------- fake Anthropic client
class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


def _text(t, stop="end_turn"):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=t)], stop_reason=stop)


def test_report_is_verified_and_retried(outputs):
    out, cfg, summary = outputs
    z2 = summary["zones"][1]
    bad = "Z3 looks excellent with a permeability of 350 nD."
    good = (f"# Report\nZ3 has mean TOC {summary['zones'][2]['averages']['toc_mean_wtpct']} wt%.\n"
            f"## Low-confidence zones\n- {z2['name']}: washout.\n")
    client = FakeClient([_text(bad), _text(good)])
    report, verification, calls = generate_report(summary, cfg, client=client)
    assert calls == 2 and verification["passed"] and report == good.strip()
    assert client.requests[0]["model"] == "claude-fable-5-2" and "temperature" not in client.requests[0]
    assert "350" in client.requests[1]["messages"][-1]["content"]              # the bad number was sent back


def test_chat_runs_python_tool_for_what_if(outputs):
    out, cfg, summary = outputs
    session = Session.load(out)
    tool_call = SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", id="toolu_1", name="compute_sw", input={"zone": "Z1", "rw": 0.05})],
        stop_reason="tool_use")
    expected, _ = Session.load(out).dispatch("compute_sw", {"zone": "Z1", "rw": 0.05})
    new = expected["results"][0]["what_if"]["sw_archie_mean"]
    client = FakeClient([tool_call, _text(f"With Rw 0.05 ohm.m, Archie Sw in Z1 becomes {new}.")])
    chat = Chat(session, cfg, client=client)
    answer, check = chat.ask("what if Rw is 0.05 in Z1?")
    assert str(new) in answer and check["passed"]
    sent_back = chat.messages[2]["content"][0]              # user, assistant(tool_use), user(tool_result)
    assert sent_back["type"] == "tool_result" and sent_back["tool_use_id"] == "toolu_1"
    assert json.loads(sent_back["content"]) == expected                        # the LLM saw Python's numbers
    assert [t["name"] for t in client.requests[0]["tools"]][2] == "compute_sw"

    client = FakeClient([_text("Sw would be about 0.731.")])                   # a made-up number is caught
    _, check = Chat(session, cfg, client=client).ask("guess Sw for Rw 0.07")
    assert not check["passed"] and check["unverified"][0]["number"] == "0.731"
