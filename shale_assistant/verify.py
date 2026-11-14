"""Verification. Two questions, answered by Python, not by the LLM:

1. Does every number in the LLM text exist in the JSON it was given
   (plus, for chat, in the Python tool results of that conversation)?
2. Are Vsh and Sw between 0 and 1 (zone averages, and every depth sample if
   curves.csv is supplied)?  Also: is every low-confidence zone flagged in the report?

Matching is strict by default: 0.30 matches 0.3, but 0.35 does not match 0.347.
A percentage made from a fraction (0.35 -> 35%) is reported as a conversion,
because converting is calculating.
"""
from __future__ import annotations

import math
import re

NUMBER = re.compile(
    r"(?<![A-Za-z0-9_.])"                       # not glued to a word, e.g. Z3, log10, P95
    r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:[eE]([-+]?\d+))?"
    r"(\s?%)?")
LIST_MARKER = re.compile(r"^(\s*(?:>\s*)?)\d+[.)]\s+", re.M)          # "1. item"
HEADING_NUMBER = re.compile(r"^(\s*#{1,6}\s*)\d+(?:\.\d+)*[.)]?\s+", re.M)  # "## 2. Title"
SIGN_CHARS = "-−"
RANGE_KEYS_01 = ("vsh_", "sw_")


def _key(x: float) -> float:
    return float(f"{x:.10g}")


def extract_numbers(text: str, strip: list[str] | None = None) -> list[dict]:
    """Every number in text as {raw, value, decimals, percent, context}."""
    for s in sorted(set(strip or []), key=len, reverse=True):
        if s:
            text = text.replace(s, " ")
    text = HEADING_NUMBER.sub(r"\1", text)
    text = LIST_MARKER.sub(r"\1", text)
    found = []
    for m in NUMBER.finditer(text):
        body, exp, pct = m.group(1), m.group(2), m.group(3)
        try:
            value = float(body.replace(",", "") + (f"e{exp}" if exp else ""))
        except ValueError:
            continue
        i = m.start()
        negative = (i > 0 and text[i - 1] in SIGN_CHARS
                    and (i == 1 or not (text[i - 2].isalnum() or text[i - 2] in ")]")))
        if negative:
            value = -value
        decimals = len(body.split(".")[1]) if "." in body and not exp else (0 if not exp else None)
        found.append({"raw": ("-" if negative else "") + m.group(0).strip(), "value": value,
                      "decimals": decimals, "percent": bool(pct),
                      "context": " ".join(text[max(0, i - 45):m.end() + 25].split())})
    return found


def collect_numbers(obj) -> set[float]:
    """All numbers in a JSON-like object: numeric leaves and numbers inside strings and keys."""
    out: set[float] = set()

    def add(x: float):
        if math.isfinite(x):
            out.add(_key(x))
            out.add(_key(abs(x)))

    def walk(o):
        if isinstance(o, bool) or o is None:
            return
        if isinstance(o, (int, float)):
            add(float(o))
        elif isinstance(o, str):
            for n in extract_numbers(o):
                add(n["value"])
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(str(k))
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)

    walk(obj)
    return out


def check_numbers(text: str, sources: list, strip: list[str] | None = None,
                  allow_rounding: bool = False) -> dict:
    allowed: set[float] = set()
    for src in sources:
        allowed |= collect_numbers(src)
    numbers = extract_numbers(text, strip)
    unverified = []
    for n in numbers:
        v = n["value"]
        if _key(v) in allowed or _key(abs(v)) in allowed:
            continue
        if allow_rounding and n["decimals"] is not None:
            half = 0.5 * 10 ** (-n["decimals"]) + 1e-12
            if any(abs(a - abs(v)) <= half for a in allowed):
                continue
        reason = "not in the JSON"
        if n["percent"] and _key(abs(v) / 100.0) in allowed:
            reason = "percentage converted from a fraction in the JSON (a calculation)"
        elif n["decimals"] is not None:
            half = 0.5 * 10 ** (-n["decimals"]) + 1e-12
            if any(abs(a - abs(v)) <= half for a in allowed):
                reason = "rounded version of a JSON number (copy it exactly)"
        unverified.append({"number": n["raw"], "reason": reason, "context": n["context"]})
    return {"numbers_found": len(numbers), "numbers_verified": len(numbers) - len(unverified),
            "unverified": unverified, "passed": not unverified}


def check_ranges(summary: dict, curves=None) -> dict:
    """Vsh and Sw must be within 0-1; porosity within 0-phi_max; fractions within 0-1."""
    violations, checked = [], 0
    phi_max = summary["parameters"]["porosity"]["phi_max"]
    for z in summary["zones"]:
        for key, value in z["averages"].items():
            if value is None:
                continue
            if key.startswith(RANGE_KEYS_01) or key.endswith("_fraction"):
                lo, hi = 0.0, 1.0
            elif key in ("phit_mean", "phie_mean"):
                lo, hi = 0.0, phi_max
            elif key.startswith("toc_"):
                lo, hi = 0.0, 100.0
            else:
                continue
            checked += 1
            if not lo <= value <= hi:
                violations.append({"where": f"zone {z['name']}", "quantity": key, "value": value,
                                   "allowed": [lo, hi]})
        fac = z.get("facies") or {}
        for key in ("dominant_fraction", "mean_confidence", "classified_fraction"):
            if fac.get(key) is not None:
                checked += 1
                if not 0.0 <= fac[key] <= 1.0:
                    violations.append({"where": f"zone {z['name']}", "quantity": f"facies.{key}",
                                       "value": fac[key], "allowed": [0.0, 1.0]})
    samples = 0
    if curves is not None:
        for col in ("VSH_LINEAR", "VSH_LARIONOV", "VSH", "SW_ARCHIE", "SW_SIMANDOUX"):
            if col not in curves.columns:
                continue
            x = curves[col].dropna()
            samples += len(x)
            bad = x[(x < 0.0) | (x > 1.0)]
            if len(bad):
                violations.append({"where": "curves.csv", "quantity": col, "n_samples_outside": int(len(bad)),
                                   "min": float(x.min()), "max": float(x.max()), "allowed": [0.0, 1.0]})
    return {"zone_values_checked": checked, "depth_samples_checked": samples,
            "violations": violations, "passed": not violations}


LOW_CONF = re.compile(r"low[\s\-‑]*confidence", re.I)


def check_low_confidence_flagged(report: str, summary: dict) -> dict:
    """Each zone with low_confidence = true must be named in a paragraph, table row or section
    that says 'low confidence'."""
    flagged = [z["name"] for z in summary["zones"] if z["low_confidence"]]
    pool = []
    in_section, level = False, 0
    for line in report.splitlines():
        heading = re.match(r"^\s*(#{1,6})\s", line)
        if heading:
            if LOW_CONF.search(line):
                in_section, level = True, len(heading.group(1))
            elif in_section and len(heading.group(1)) <= level:
                in_section = False
        if in_section or LOW_CONF.search(line):
            pool.append(line)
    for para in re.split(r"\n\s*\n", report):
        if LOW_CONF.search(para):
            pool.append(para)
    text = "\n".join(pool)
    missing = [name for name in flagged
               if not re.search(r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])", text)]
    return {"low_confidence_zones": flagged, "not_flagged_in_report": missing, "passed": not missing}


def strip_list(summary: dict, extra: list[str] | None = None) -> list[str]:
    """Names that contain digits but are labels, not data (well name, file name, zone names)."""
    items = [summary["well"]["name"], summary["well"]["file"]] + [z["name"] for z in summary["zones"]]
    return items + list(extra or [])


def verify_report(report: str, summary: dict, curves=None, allow_rounding: bool = False) -> dict:
    numbers = check_numbers(report, [summary], strip_list(summary), allow_rounding)
    ranges = check_ranges(summary, curves)
    flags = check_low_confidence_flagged(report, summary)
    return {"passed": numbers["passed"] and ranges["passed"] and flags["passed"],
            "strict_matching": not allow_rounding,
            "numbers": numbers, "ranges": ranges, "low_confidence_flags": flags}


def format_verification(v: dict) -> str:
    n, r, f = v["numbers"], v["ranges"], v["low_confidence_flags"]
    L = ["=" * 78, f"VERIFICATION: {'PASSED' if v['passed'] else 'FAILED'}", "=" * 78,
         f"[{'ok' if n['passed'] else 'FAIL'}] numbers in the report found in the JSON: "
         f"{n['numbers_verified']} of {n['numbers_found']}"]
    for u in n["unverified"]:
        L.append(f"       {u['number']:>10}  {u['reason']}   ...{u['context']}...")
    L.append(f"[{'ok' if r['passed'] else 'FAIL'}] Vsh and Sw within 0-1 (and porosity, TOC, fractions in range): "
             f"{r['zone_values_checked']} zone values, {r['depth_samples_checked']} depth samples checked")
    for viol in r["violations"]:
        L.append(f"       {viol}")
    L.append(f"[{'ok' if f['passed'] else 'FAIL'}] low-confidence zones flagged in the report: "
             f"{len(f['low_confidence_zones']) - len(f['not_flagged_in_report'])} of {len(f['low_confidence_zones'])}")
    if f["not_flagged_in_report"]:
        L.append(f"       not flagged: {', '.join(f['not_flagged_in_report'])}")
    return "\n".join(L)
