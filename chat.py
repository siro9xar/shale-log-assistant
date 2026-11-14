"""Chat with the well. What-if questions make the LLM call Python tools (it never calculates).

Example questions:
    which zone has the highest TOC?
    what if Rw is 0.05 in Z8?
    what happens to porosity in all zones if matrix density is 2.71?
Type  exit  to leave.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))            # so "shale_assistant" imports from any working directory
import argparse
import json

from shale_assistant.config import load_config
from shale_assistant.llm import Chat
from shale_assistant.tools import Session



def show_tool(name, tool_input, result, is_error):
    args = ", ".join(f"{k}={v!r}" for k, v in (tool_input or {}).items())
    print(f"  [python] {name}({args})" + ("  -> ERROR: " + result.get("error", "") if is_error else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "example"),
                    help="folder with zone_summary.json and curves.csv")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--no-llm", metavar="TOOL_CALL_JSON",
                    help="run one tool directly without the LLM, e.g. "
                         "'{\"tool\": \"compute_sw\", \"input\": {\"zone\": \"Z8\", \"rw\": 0.05}}'")
    args = ap.parse_args()

    session = Session.load(args.out)
    if args.no_llm:
        call = json.loads(args.no_llm)
        result, _ = session.dispatch(call["tool"], call.get("input", {}))
        print(json.dumps(result, indent=2))
        raise SystemExit(0)

    chat = Chat(session, load_config(args.config), on_tool=show_tool)
    print(f"Well {session.summary['well']['name']}: {len(session.summary['zones'])} zones loaded. Ask a question (exit to quit).")
    while True:
        try:
            question = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break
        answer, check = chat.ask(question)
        print(f"\nassistant> {answer}")
        if check["passed"]:
            print(f"  [check] all {check['numbers_found']} numbers come from the JSON, a Python tool result or your question")
        else:
            print("  [check] WARNING - numbers that are NOT in the JSON or a tool result:")
            for u in check["unverified"]:
                print(f"          {u['number']}  ({u['reason']})  ...{u['context']}...")
