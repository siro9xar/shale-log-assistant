"""One self-contained HTML file (and a PDF made with matplotlib, no extra installs) per well:
verification status in plain words at the top, then the report, zone table, log plot and confusion matrix.

The verification is RE-RUN here on report.md + zone_summary.json + curves.csv; a saved
report_verification.json is not trusted, because report.md may have been edited after it was written.
"""
from __future__ import annotations

import base64
import html
import json
import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from .verify import format_verification, verify_report

ZONE_COLUMNS = [  # (header, getter, decimals)
    ("Zone", lambda z: z["name"] + (" *" if z["low_confidence"] else ""), None),
    ("Top", lambda z: z["top"], 1), ("Base", lambda z: z["base"], 1), ("Thk", lambda z: z["thickness"], 1),
    ("Vsh", lambda z: z["averages"]["vsh_larionov_mean"], 2),
    ("PHIT", lambda z: z["averages"]["phit_mean"], 3), ("PHIE", lambda z: z["averages"]["phie_mean"], 3),
    ("Sw Archie", lambda z: z["averages"]["sw_archie_mean"], 2),
    ("Sw Simandoux", lambda z: z["averages"]["sw_simandoux_mean"], 2),
    ("TOC wt%", lambda z: z["averages"]["toc_mean_wtpct"], 2),
    ("Facies", lambda z: (z["facies"] or {}).get("dominant") or "-", None),
    ("Conf.", lambda z: (z["facies"] or {}).get("mean_confidence"), 2),
]


# --------------------------------------------------------------------------- inputs
def collect(out_dir, metrics_path=None) -> dict:
    out = Path(out_dir)
    summary = json.loads((out / "zone_summary.json").read_text(encoding="utf-8"))
    curves = pd.read_csv(out / "curves.csv") if (out / "curves.csv").exists() else None
    report_path = out / "report.md"
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else None
    verification = verify_report(report, summary, curves) if report is not None else None
    metrics, metrics_note = None, None
    if metrics_path and Path(metrics_path).exists():
        metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        fm = summary.get("facies_model") or {}
        same = (fm.get("test_accuracy") == metrics["test"]["accuracy"]
                and fm.get("classes") == metrics["classes"])
        if fm.get("available") and not same:
            metrics_note = ("WARNING: these facies metrics do not match the model recorded in zone_summary.json "
                            f"(accuracy {metrics['test']['accuracy']} vs {fm.get('test_accuracy')}): "
                            "they may describe a different model")
    elif (summary.get("facies_model") or {}).get("available"):
        metrics_note = f"facies metrics file not found ({metrics_path}); no confusion matrix shown"
    return {"out": out, "summary": summary, "report": report, "verification": verification,
            "metrics": metrics, "metrics_note": metrics_note,
            "log_plot": out / "log_plot.png" if (out / "log_plot.png").exists() else None}


def status(d: dict) -> dict:
    """Plain-words verification status: level ok / fail / none, a headline and detail lines."""
    v = d["verification"]
    if d["report"] is None:
        return {"level": "none", "headline": "NOT VERIFIED - there is no LLM report for this well.",
                "lines": ["report.md does not exist in this folder, so there is nothing to verify.",
                          "Everything below is Python output (QC, calculations, zone table, plots); "
                          "no text was written by the LLM.",
                          "Run scripts/generate_report.py, then export again."]}
    n, r, f = v["numbers"], v["ranges"], v["low_confidence_flags"]
    lines = [f"Numbers in the report found in the JSON: {n['numbers_verified']} of {n['numbers_found']}.",
             f"Vsh and Sw within 0-1: {'yes' if r['passed'] else 'NO'} "
             f"({r['zone_values_checked']} zone values, {r['depth_samples_checked']} depth samples checked).",
             f"Low-confidence zones flagged in the report: "
             f"{len(f['low_confidence_zones']) - len(f['not_flagged_in_report'])} of {len(f['low_confidence_zones'])}."]
    if v["passed"]:
        return {"level": "ok", "headline": "VERIFIED - the report passed every automatic check.",
                "lines": lines + ["This checks that the numbers come from Python; it does not make the "
                                  "interpretation or the assumptions correct."]}
    bad = [f"'{u['number']}' - {u['reason']} - \u2026{u['context']}\u2026" for u in n["unverified"]]
    bad += [f"{x['where']}: {x['quantity']} outside {x['allowed']}" for x in r["violations"]]
    bad += [f"low-confidence zone {z} is not flagged in the report" for z in f["not_flagged_in_report"]]
    return {"level": "fail", "headline": "FAILED VERIFICATION - do not use this report as it is.",
            "lines": lines, "failures": bad}


# --------------------------------------------------------------------------- tiny Markdown -> HTML
def _inline(text: str) -> str:
    t = html.escape(text, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![*\w])\*([^*\n]+)\*(?![*\w])", r"<em>\1</em>", t)
    return t


def markdown_to_html(md: str) -> str:
    """Headings, paragraphs, bullet/numbered lists, pipe tables, block quotes, code fences, bold/italic/code.
    All text is HTML-escaped: the report cannot inject markup."""
    out, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.strip().startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            out.append("<pre>" + html.escape("\n".join(lines[i + 1:j])) + "</pre>")
            i = j + 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            lvl = min(len(m.group(1)) + 1, 6)          # report '#' becomes <h2>: the page owns <h1>
            out.append(f"<h{lvl}>{_inline(m.group(2).strip())}</h{lvl}>")
            i += 1
            continue
        if line.lstrip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?[\s:\-|]+\|?\s*$", lines[i + 1]):
            def cells(s):
                return [c.strip() for c in s.strip().strip("|").split("|")]
            head = cells(line)
            j = i + 2
            rows = []
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows.append(cells(lines[j]))
                j += 1
            out.append("<table><thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in head) + "</tr></thead><tbody>"
                       + "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in rows)
                       + "</tbody></table>")
            i = j
            continue
        if re.match(r"^\s*([-*+]|\d+[.)])\s+", line):
            ordered = bool(re.match(r"^\s*\d+[.)]\s+", line))
            items = []
            while i < len(lines) and re.match(r"^\s*([-*+]|\d+[.)])\s+", lines[i]):
                items.append(re.sub(r"^\s*([-*+]|\d+[.)])\s+", "", lines[i]))
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{_inline(x)}</li>" for x in items) + f"</{tag}>")
            continue
        if line.startswith(">"):
            block = []
            while i < len(lines) and lines[i].startswith(">"):
                block.append(lines[i].lstrip(">").strip())
                i += 1
            out.append(f"<blockquote>{_inline(' '.join(block))}</blockquote>")
            continue
        para = []
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,6}\s|\s*\||\s*([-*+]|\d+[.)])\s|>|```)", lines[i]):
            para.append(lines[i].strip())
            i += 1
        out.append(f"<p>{_inline(' '.join(para))}</p>")
    return "\n".join(out)


# --------------------------------------------------------------------------- HTML
def _fmt(v, nd):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return f"{v:.{nd}f}" if nd is not None and isinstance(v, (int, float)) else str(v)


def zone_rows(summary) -> list[list[str]]:
    return [[_fmt(get(z), nd) for _, get, nd in ZONE_COLUMNS] for z in summary["zones"]]


def _img(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


CSS = """
:root{--fg:#1d2329;--muted:#5b6670;--line:#d8dde2;--bg:#fff;--ok:#1e7a3c;--okbg:#e6f4ea;--fail:#b3261e;
--failbg:#fde8e7;--none:#8a5a00;--nonebg:#fff4dc;--th:#f3f5f7}
body{font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--fg);background:var(--bg);margin:0}
main{max-width:1100px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:20px;margin:32px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
h3{font-size:17px;margin:20px 0 6px}.muted{color:var(--muted);font-size:13px}
.status{border:2px solid;border-radius:8px;padding:14px 16px;margin:16px 0 8px}
.status h2{border:0;margin:0 0 6px;padding:0;font-size:20px}
.status.ok{border-color:var(--ok);background:var(--okbg)}.status.ok h2{color:var(--ok)}
.status.fail{border-color:var(--fail);background:var(--failbg)}.status.fail h2{color:var(--fail)}
.status.none{border-color:var(--none);background:var(--nonebg)}.status.none h2{color:var(--none)}
.status ul{margin:6px 0 0 18px;padding:0}
.scroll{overflow-x:auto}table{border-collapse:collapse;margin:8px 0;font-size:13px}
th,td{border:1px solid var(--line);padding:4px 7px;text-align:right;white-space:nowrap}
th{background:var(--th)}td:first-child,th:first-child{text-align:left}
td.lc{color:var(--fail);font-weight:600}img{max-width:100%;height:auto;border:1px solid var(--line)}
pre{background:var(--th);padding:10px;overflow-x:auto;font-size:12px}code{background:var(--th);padding:0 3px}
blockquote{border-left:3px solid var(--line);margin:8px 0;padding:2px 12px;color:var(--muted)}
.cm td{font-variant-numeric:tabular-nums}
@media print{.status{break-inside:avoid}h2{break-after:avoid}img{break-inside:avoid}}
"""


def build_html(d: dict) -> str:
    s, st = d["summary"], status(d)
    w = s["well"]
    unit = (w.get("depth_unit") or "").lower()
    E = html.escape
    P = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
         f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>{E(w['name'])} report</title><style>{CSS}</style></head><body><main>",
         f"<h1>{E(w['name'])}</h1>",
         f"<div class='muted'>{E(w.get('file', ''))} &middot; {w['top']} - {w['base']} {E(unit)} &middot; "
         f"generated by {E(str(s.get('generated_by', '')))}</div>",
         f"<section class='status {st['level']}'><h2>{E(st['headline'])}</h2><ul>"
         + "".join(f"<li>{E(x)}</li>" for x in st["lines"]) + "</ul>"]
    if st.get("failures"):
        P.append("<h3>What failed</h3><ul>" + "".join(f"<li>{E(x)}</li>" for x in st["failures"]) + "</ul>")
    P.append("</section>")
    n_lc = sum(z["low_confidence"] for z in s["zones"])
    P.append(f"<p class='muted'>{n_lc} of {len(s['zones'])} zones are low confidence (marked *). "
             "Assumptions are listed in the report and in zone_summary.json; nothing here is calibrated "
             "against core unless the report says so.</p>")

    P.append("<h2>Zone table</h2><div class='scroll'><table><thead><tr>"
             + "".join(f"<th>{E(h)}{' (' + E(unit) + ')' if h in ('Top', 'Base', 'Thk') else ''}</th>"
                       for h, _, _ in ZONE_COLUMNS) + "</tr></thead><tbody>")
    for z, row in zip(s["zones"], zone_rows(s)):
        P.append("<tr>" + "".join(f"<td{' class=lc' if k == 0 and z['low_confidence'] else ''}>{E(c)}</td>"
                                  for k, c in enumerate(row)) + "</tr>")
    P.append("</tbody></table></div>")
    lc = [z for z in s["zones"] if z["low_confidence"]]
    if lc:
        P.append("<h3>Low-confidence zones</h3><ul>" + "".join(
            f"<li><strong>{E(z['name'])}</strong>: {E('; '.join(z['low_confidence_reasons']))}</li>" for z in lc)
                 + "</ul>")

    P.append("<h2>Report</h2>")
    P.append(markdown_to_html(d["report"]) if d["report"] is not None
             else "<p class='muted'>No report.md in this folder.</p>")

    P.append("<h2>Log plot</h2>")
    P.append(f"<img src='{_img(d['log_plot'])}' alt='log plot of {E(w['name'])}'>" if d["log_plot"]
             else "<p class='muted'>log_plot.png not found.</p>")

    P.append("<h2>Facies classifier: confusion matrix</h2>")
    if d["metrics_note"]:
        P.append(f"<p><strong>{E(d['metrics_note'])}</strong></p>")
    m = d["metrics"]
    if m:
        t, sp = m["test"], m["split"]
        P.append(f"<p>{E(sp['method'])}. Accuracy {t['accuracy']}, balanced accuracy {t['balanced_accuracy']}, "
                 f"macro F1 {t['macro_f1']} on {sp['n_test_wells']} held-out wells "
                 f"({sp['n_test_samples']} samples). Rows = true class, columns = predicted.</p>")
        labels, cm = m["confusion_matrix"]["labels"], np.array(m["confusion_matrix"]["matrix"])
        rs = cm.sum(axis=1, keepdims=True)
        frac = np.divide(cm, rs, out=np.zeros(cm.shape), where=rs > 0)
        P.append("<div class='scroll'><table class='cm'><thead><tr><th>true \\ predicted</th>"
                 + "".join(f"<th>{E(l)}</th>" for l in labels) + "<th>recall</th></tr></thead><tbody>")
        for i, lab in enumerate(labels):
            cells = "".join(f"<td style='background:rgba(31,111,180,{0.85 * frac[i, j]:.2f});"
                            f"color:{'#fff' if frac[i, j] > 0.5 else 'inherit'}'>{cm[i, j]}</td>"
                            for j in range(len(labels)))
            rec = t["per_class"].get(lab, {}).get("recall")
            P.append(f"<tr><th>{E(lab)}</th>{cells}<td>{_fmt(rec, 3)}</td></tr>")
        P.append("</tbody></table></div>")
    P.append("<h2>Verification detail</h2>")
    P.append(f"<pre>{E(format_verification(d['verification']))}</pre>" if d["verification"]
             else "<p class='muted'>No report, so no verification was run.</p>")
    P.append("</main></body></html>")
    return "\n".join(P)


# --------------------------------------------------------------------------- PDF (matplotlib only)
def build_pdf(d: dict, path) -> int:
    """A4 portrait pages: status + zone table, log plot, confusion matrix, report text. Returns page count."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    A4 = (8.27, 11.69)
    s, st = d["summary"], status(d)
    colour = {"ok": "#1e7a3c", "fail": "#b3261e", "none": "#8a5a00"}[st["level"]]
    pages = 0

    def text_page(pdf, title, body_lines, first=None):
        nonlocal pages
        fig = plt.figure(figsize=A4)
        y = 0.95
        if first:
            first(fig)
            y = first.y
        fig.text(0.06, y, title, fontsize=13, weight="bold", va="top")
        y -= 0.035
        for ln in body_lines:
            if y < 0.05:
                pdf.savefig(fig)
                plt.close(fig)
                pages += 1
                fig = plt.figure(figsize=A4)
                y = 0.95
            fig.text(0.06, y, ln, fontsize=8.5, va="top", family="monospace")
            y -= 0.0165
        pdf.savefig(fig)
        plt.close(fig)
        pages += 1

    with PdfPages(path) as pdf:
        # page 1: status + zone table
        fig = plt.figure(figsize=A4)
        fig.text(0.06, 0.965, s["well"]["name"], fontsize=16, weight="bold", va="top")
        box = [st["headline"]] + st["lines"] + [("  - " + x) for x in st.get("failures", [])[:12]]
        wrapped = [w for ln in box for w in textwrap.wrap(ln, 95)]
        top, h = 0.93, 0.02 * len(wrapped) + 0.02
        fig.patches.append(plt.Rectangle((0.05, top - h), 0.9, h, transform=fig.transFigure, fill=False,
                                         ec=colour, lw=2))
        for k, ln in enumerate(wrapped):
            fig.text(0.07, top - 0.012 - 0.02 * k, ln, fontsize=10 if k else 12, va="top",
                     color=colour if k == 0 else "black", weight="bold" if k == 0 else "normal")
        ax = fig.add_axes([0.05, 0.05, 0.9, top - h - 0.09])
        ax.axis("off")
        ax.set_title("Zone table (* = low confidence)", loc="left", fontsize=11)
        unit = (s["well"].get("depth_unit") or "").lower()
        hdr = [h + (f" ({unit})" if h in ("Top", "Base", "Thk") else "") for h, _, _ in ZONE_COLUMNS]
        tb = ax.table(cellText=zone_rows(s), colLabels=hdr, loc="upper center", cellLoc="right")
        tb.auto_set_font_size(False)
        tb.set_fontsize(6.5)
        tb.scale(1, 1.3)
        for (r, c), cell in tb.get_celld().items():
            if r == 0:
                cell.set_facecolor("#f3f5f7")
            elif c == 0 and s["zones"][r - 1]["low_confidence"]:
                cell.get_text().set_color("#b3261e")
        pdf.savefig(fig)
        plt.close(fig)
        pages += 1

        if d["log_plot"]:
            img = plt.imread(d["log_plot"])
            fig = plt.figure(figsize=(A4[1], A4[0]))          # landscape for the wide plot
            ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)
            pages += 1

        m = d["metrics"]
        if m:
            labels, cm = m["confusion_matrix"]["labels"], np.array(m["confusion_matrix"]["matrix"])
            rs = cm.sum(axis=1, keepdims=True)
            frac = np.divide(cm, rs, out=np.zeros(cm.shape), where=rs > 0)
            fig, ax = plt.subplots(figsize=A4)
            ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(labels)), labels, fontsize=8)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6.5,
                            color="white" if frac[i, j] > 0.5 else "black")
            t = m["test"]
            note = f"\n{d['metrics_note']}" if d["metrics_note"] else ""
            ax.set_title(f"Confusion matrix ({m['split']['n_test_wells']} held-out wells; rows = true)\n"
                         f"accuracy {t['accuracy']}, balanced accuracy {t['balanced_accuracy']}, "
                         f"macro F1 {t['macro_f1']}{note}", fontsize=9)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            pages += 1

        body = []
        for ln in (d["report"] or "No report.md in this folder.").splitlines():
            body += textwrap.wrap(ln, 100) or [""]
        text_page(pdf, "Report (report.md, as written)", body)
        if d["verification"]:
            vlines = []
            for ln in format_verification(d["verification"]).splitlines():
                vlines += textwrap.wrap(ln, 100) or [""]
            text_page(pdf, "Verification detail", vlines)
    return pages


def export(out_dir, metrics_path=None, html_path=None, pdf_path=None, make_pdf=True) -> dict:
    d = collect(out_dir, metrics_path)
    html_path = Path(html_path or d["out"] / "report.html")
    html_path.write_text(build_html(d), encoding="utf-8")
    result = {"html": html_path, "status": status(d), "pdf": None, "pdf_pages": 0, "pdf_error": None}
    if make_pdf:
        pdf_path = Path(pdf_path or d["out"] / "report.pdf")
        try:
            result["pdf_pages"] = build_pdf(d, pdf_path)
            result["pdf"] = pdf_path
        except Exception as err:                      # the HTML is the main product; say why the PDF failed
            result["pdf_error"] = f"{type(err).__name__}: {err}"
    return result
