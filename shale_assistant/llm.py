"""LLM layer (Anthropic API). The LLM writes words; it never produces a number of its own.

Report : the zone-summary JSON goes in, a petrophysical report comes out, and the
         report is checked by verify.py. If the check fails the model is asked once
         more to fix the listed numbers.
Chat   : the model gets tools (tools.py). A what-if question makes it call a tool,
         Python re-runs the calculation, and the model reports the returned numbers.
"""
from __future__ import annotations

import json
import os

from .tools import TOOLS, Session
from .verify import check_numbers, format_verification, strip_list, verify_report

REPORT_SYSTEM_PROMPT = """You are a senior petrophysicist writing an interpretation report for a well evaluated \
as an unconventional (shale) reservoir. You are given one JSON document. Every number in it was calculated \
by Python.

Rules. They are checked by a script after you answer.
- Use ONLY numbers that appear in the JSON. Copy each number exactly as it is written there: same digits, \
no rounding, no unit conversion, no turning fractions into percentages.
- NEVER calculate. No sums, differences, ratios, averages, percentages, counts or estimates of your own. \
If a number you would like is not in the JSON, say it is not available. Compare zones in words \
(higher, lower, highest) using the values and the "rankings" block.
- Flag every zone whose "low_confidence" is true. Mark it in the zone table and list it in the section \
"Low-confidence zones" with the reasons given in "low_confidence_reasons". Do not draw firm conclusions \
from a low-confidence zone.
- State the assumptions from "parameters" (matrix and fluid density, Rw, a, m, n, baselines, LOM, \
gamma-ray end points) and say which ones were picked automatically ("auto_picked").
- Do not invent anything that is not in the JSON: no formation names, ages, fluid contacts, core data, \
test results, permeability or reserves.
- Refer to zones by their "name". Do not number the headings and do not use numbered lists.

Write the report in Markdown with these sections, in this order:
# Petrophysical interpretation report: <well name>
## Summary
## Data quality
## Assumptions and parameters
## Zone results   (a table: zone, top, base, thickness, Vsh, PHIT, PHIE, Sw Archie, Sw Simandoux, TOC, \
dominant facies, facies confidence, flag)
## Interpretation   (organic richness, porosity, saturation, facies; which zones look best and why)
## Low-confidence zones
## Limitations and recommended next steps
"""

CHAT_SYSTEM_PROMPT = """You are a petrophysics assistant for one well. The zone-summary JSON below was \
calculated by Python. You also have tools that re-run the Python calculations.

Rules:
- You never calculate. Every number you state must be copied exactly from the JSON or from a tool \
result in this conversation. No arithmetic, rounding, unit conversion or percentages of your own.
- For any "what if" question (a different Rw, m, n, a, Rsh, matrix density, gamma-ray end point, LOM, \
baseline ...) call the matching tool. The tool returns the base case, the what-if case and the change, \
so report those; do not subtract yourself.
- If the user asks for a number that neither the JSON nor a tool can provide, say so instead of estimating.
- When a zone has low_confidence = true, say so whenever you quote its numbers.
- Keep answers short. Name the zone, the parameter that changed, and the units.

ZONE-SUMMARY JSON:
"""


def make_client():
    """Anthropic client. Needs the ANTHROPIC_API_KEY environment variable."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Create a key in the Claude Console and run\n"
                           "  export ANTHROPIC_API_KEY=sk-ant-...        (Windows PowerShell: $env:ANTHROPIC_API_KEY=\"sk-ant-...\")")
    import anthropic
    return anthropic.Anthropic()


def _kwargs(cfg: dict) -> dict:
    llm = cfg["llm"]
    kw = {"model": llm["model"], "max_tokens": int(llm["max_tokens"])}
    if llm.get("temperature") is not None:
        kw["temperature"] = float(llm["temperature"])
    return kw


def _text(response) -> str:
    return "".join(block.text for block in response.content if getattr(block, "type", None) == "text").strip()


# --------------------------------------------------------------------------- report
def generate_report(summary: dict, cfg: dict, client=None, curves=None) -> tuple[str, dict, int]:
    """Return (report_markdown, verification, n_llm_calls)."""
    client = client or make_client()
    messages = [{"role": "user", "content":
                 "Write the petrophysical report for this well from the JSON below.\n\n"
                 + json.dumps(summary, indent=1)}]
    retries = int(cfg["llm"].get("max_verification_retries", 0))
    calls = 0
    while True:
        response = client.messages.create(system=REPORT_SYSTEM_PROMPT, messages=messages, **_kwargs(cfg))
        calls += 1
        report = _text(response)
        if getattr(response, "stop_reason", None) == "max_tokens":
            report += "\n\n> NOTE: the report was cut off at llm.max_tokens; raise it in config.yaml."
        verification = verify_report(report, summary, curves)
        if verification["passed"] or calls > retries:
            return report, verification, calls
        problems = [f"- {u['number']}: {u['reason']} (near: \"{u['context']}\")"
                    for u in verification["numbers"]["unverified"]]
        if verification["low_confidence_flags"]["not_flagged_in_report"]:
            problems.append("- these low-confidence zones are not flagged in a low-confidence section: "
                            + ", ".join(verification["low_confidence_flags"]["not_flagged_in_report"]))
        messages += [{"role": "assistant", "content": report},
                     {"role": "user", "content":
                      "The automatic check found problems. Rewrite the whole report and fix them. "
                      "Remove or replace each number below with a value copied exactly from the JSON, "
                      "or describe it in words.\n" + "\n".join(problems)}]


# --------------------------------------------------------------------------- chat
class Chat:
    def __init__(self, session: Session, cfg: dict, client=None, on_tool=None):
        self.session = session
        self.cfg = cfg
        self.client = client or make_client()
        self.on_tool = on_tool                      # callback(name, input, result, is_error) for display
        self.messages: list = []
        self.user_texts: list[str] = []
        self.system = CHAT_SYSTEM_PROMPT + json.dumps(session.summary, indent=1)

    def ask(self, question: str, max_tool_rounds: int = 8) -> tuple[str, dict]:
        """Return (answer, number_check). Numbers may come from the JSON, a tool result or the user."""
        self.user_texts.append(question)
        self.messages.append({"role": "user", "content": question})
        for _ in range(max_tool_rounds + 1):
            response = self.client.messages.create(system=self.system, messages=self.messages,
                                                   tools=TOOLS, **_kwargs(self.cfg))
            self.messages.append({"role": "assistant", "content": response.content})
            if getattr(response, "stop_reason", None) != "tool_use":
                break
            results = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                result, is_error = self.session.dispatch(block.name, dict(block.input or {}))
                if self.on_tool:
                    self.on_tool(block.name, block.input, result, is_error)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": json.dumps(result), "is_error": is_error})
            self.messages.append({"role": "user", "content": results})
        answer = _text(response)
        sources = [self.session.summary, self.session.calls, self.user_texts]
        check = check_numbers(answer, sources, strip_list(self.session.summary))
        return answer, check


__all__ = ["REPORT_SYSTEM_PROMPT", "CHAT_SYSTEM_PROMPT", "make_client", "generate_report", "Chat",
           "format_verification"]
