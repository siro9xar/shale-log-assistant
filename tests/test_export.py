"""export_report: status at the top in plain words, re-verification, self-contained HTML, PDF."""
import json
import re
import shutil
import subprocess
import sys

import lasio
import numpy as np
import pytest

from shale_assistant.config import load_config
from shale_assistant.export import export, markdown_to_html
from shale_assistant.pipeline import run
from tests.conftest import ROOT


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("export")
    rng = np.random.default_rng(7)
    n = 600
    depth = 3000 + 0.1524 * np.arange(n)
    block = np.repeat([0, 1, 2], n // 3)
    cali = np.full(n, 8.6)
    cali[250:300] = 11.5                                       # washout -> Z2 is low confidence
    las = lasio.LASFile()
    las.well["WELL"].value = "SYN-EXPORT"
    las.append_curve("DEPT", depth, unit="M")
    for name, data, unit in (("GR", np.choose(block, [30, 110, 180]) + rng.normal(0, 3, n), "GAPI"),
                             ("RHOB", np.choose(block, [2.35, 2.55, 2.42]) + rng.normal(0, 0.01, n), "G/CC"),
                             ("NPHI", np.choose(block, [18, 30, 33]) + rng.normal(0, 0.5, n), "%"),
                             ("RT", 10 ** (np.choose(block, [1.3, 0.3, 1.2]) + rng.normal(0, 0.02, n)), "OHMM"),
                             ("DT", np.choose(block, [80, 90, 105]) + rng.normal(0, 0.5, n), "US/F"),
                             ("CALI", cali, "IN")):
        las.append_curve(name, data, unit=unit)
    las.write(str(out / "syn.las"), version=2.0)
    cfg = load_config(ROOT / "config.yaml")
    cfg["toc"]["baseline_interval"] = [3032.0, 3037.0]
    cfg["zonation"]["max_zones"] = 3
    run(out / "syn.las", cfg, out, model_path=None, verbose=False)
    return out


def _copy(run_dir, tmp_path, report=None):
    d = tmp_path / "run"
    shutil.copytree(run_dir, d)
    if report is not None:
        (d / "report.md").write_text(report, encoding="utf-8")
    return d


def _good_report(d):
    s = json.loads((d / "zone_summary.json").read_text())
    z1, z2, z3 = s["zones"]
    return (f"# Report\n\nZ3 has mean TOC **{z3['averages']['toc_mean_wtpct']}** wt%.\n\n"
            f"| Zone | TOC |\n|---|---|\n| Z1 | {z1['averages']['toc_mean_wtpct']} |\n\n"
            "## Low-confidence zones\n- " + "\n- ".join(z["name"] for z in s["zones"] if z["low_confidence"]) + "\n")


def _status_block(html_text):
    body = html_text.split("<main>", 1)[1]
    return body.split("</section>", 1)[0]


def test_no_report_is_not_verified_at_the_top(run_dir, tmp_path):
    d = _copy(run_dir, tmp_path)
    r = export(d)
    page = r["html"].read_text()
    assert r["status"]["level"] == "none"
    top = _status_block(page)
    assert "NOT VERIFIED - there is no LLM report" in top
    assert page.index("NOT VERIFIED") < page.index("Zone table") < page.index("Log plot")
    assert r["pdf"].stat().st_size > 0 and r["pdf_pages"] >= 3


def test_passing_report_is_verified(run_dir, tmp_path):
    d = _copy(run_dir, tmp_path)
    (d / "report.md").write_text(_good_report(d))
    r = export(d)
    assert r["status"]["level"] == "ok", r["status"]
    top = _status_block(r["html"].read_text())
    assert "VERIFIED - the report passed every automatic check" in top and "FAILED" not in top
    assert "<strong>" in r["html"].read_text() and "<table>" in r["html"].read_text()


def test_failing_report_lists_each_bad_number(run_dir, tmp_path):
    d = _copy(run_dir, tmp_path)
    (d / "report.md").write_text(_good_report(d).replace("## Low-confidence zones\n- Z2", "## Other\n- Z2")
                                 + "\nZ3 is excellent, permeability 350 nD and TOC 12.5%.\n")
    # a stale verification file that says "passed" must not be trusted
    (d / "report_verification.json").write_text(json.dumps({"passed": True}))
    r = export(d)
    st = r["status"]
    assert st["level"] == "fail"
    joined = " | ".join(st["failures"])
    assert "'350'" in joined and "'12.5%'" in joined and "Z2 is not flagged" in joined
    top = _status_block(r["html"].read_text())
    assert "FAILED VERIFICATION - do not use this report" in top and "350" in top


def test_report_text_cannot_inject_html(run_dir, tmp_path):
    d = _copy(run_dir, tmp_path, "# Report\n\n<script>alert(1)</script> and <img src=x onerror=alert(2)>\n")
    page = export(d, make_pdf=False)["html"].read_text()
    assert "<script>alert" not in page and "<img src=x" not in page and "&lt;script&gt;" in page


def test_html_is_self_contained(run_dir, tmp_path):
    page = export(_copy(run_dir, tmp_path), make_pdf=False)["html"].read_text()
    assert "data:image/png;base64," in page
    assert not re.search(r"""(src|href)=['"](?!data:)""", page)          # no external files or links


def test_confusion_matrix_from_metrics_and_mismatch_warning(run_dir, tmp_path):
    d = _copy(run_dir, tmp_path)
    metrics = {"classes": ["A", "B"], "features": ["GR"], "split": {"method": "m", "n_test_wells": 2,
               "n_test_samples": 10}, "test": {"accuracy": 0.8, "balanced_accuracy": 0.75, "macro_f1": 0.7,
               "per_class": {"A": {"recall": 0.9}, "B": {"recall": 0.6}}},
               "confusion_matrix": {"labels": ["A", "B"], "matrix": [[5, 1], [1, 3]]}}
    mp = tmp_path / "m.json"
    mp.write_text(json.dumps(metrics))
    s = json.loads((d / "zone_summary.json").read_text())
    s["facies_model"] = {"available": True, "classes": ["A", "B"], "test_accuracy": 0.5}
    (d / "zone_summary.json").write_text(json.dumps(s))
    page = export(d, mp, make_pdf=False)["html"].read_text()
    assert "class='cm'" in page and ">0.900<" in page
    assert "do not match the model recorded in zone_summary.json" in page


def test_markdown_subset():
    h = markdown_to_html("# T\n\nA **b** and `c`.\n\n1. one\n2. two\n\n> note\n\n```\nx < 1\n```\n")
    assert "<h2>T</h2>" in h and "<strong>b</strong>" in h and "<code>c</code>" in h
    assert "<ol><li>one</li><li>two</li></ol>" in h and "<blockquote>note</blockquote>" in h
    assert "<pre>x &lt; 1</pre>" in h


def test_script_exit_codes(run_dir, tmp_path):
    d = _copy(run_dir, tmp_path)
    cmd = [sys.executable, str(ROOT / "scripts" / "export_report.py"), "--out", str(d), "--no-pdf"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0 and p.stdout.startswith("NOT VERIFIED")
    (d / "report.md").write_text("Z1 has TOC 99.9 wt%.\n")
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 1 and p.stdout.startswith("FAILED VERIFICATION") and "99.9" in p.stdout
