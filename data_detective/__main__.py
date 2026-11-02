"""Command-line Data Detective - no API key, no server needed.

  python -m data_detective "Why did ROAS drop in Q3 2025?"     one question
  python -m data_detective                                     interactive session (follow-ups work)
  python -m data_detective --sql "Top sales by region"         also print the SQL behind each piece of evidence
  python -m data_detective --json "Revenue by year"            raw JSON (same payload as POST /api/ask)
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap

from .agent import DataDetective

GREEN, RED, AMBER, DIM, BOLD, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"


def _color(enabled: bool):
    return (lambda code, s: f"{code}{s}{RESET}") if enabled else (lambda code, s: s)


def _table(columns: list, rows: list, limit: int = 8) -> str:
    def cell(v):
        if isinstance(v, float):
            return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.3f}"
        return "" if v is None else str(v)
    body = [[cell(v) for v in r] for r in rows[:limit]]
    widths = [max(len(str(c)), *(len(r[i]) for r in body)) if body else len(str(c)) for i, c in enumerate(columns)]
    line = lambda r: "  ".join(str(v).ljust(w) if i == 0 else str(v).rjust(w) for i, (v, w) in enumerate(zip(r, widths)))
    return "\n".join([line(columns), line(["-" * w for w in widths]), *map(line, body)])


def render(r: dict, show_sql: bool, color: bool) -> str:
    c = _color(color)
    out = [c(BOLD, r["headline"]), textwrap.fill(r["summary"], 100), ""]
    for f in r["findings"]:
        mark = c(GREEN, "[ok]") if f["verified"] else c(RED, "[??]")
        cites = c(DIM, " [" + ", ".join(f["evidence"]) + "]")
        out.append(textwrap.fill(f"{mark} {f['text']}{cites}", 100, subsequent_indent="     "))
    for cv in r["caveats"]:
        out.append(textwrap.fill(c(AMBER, "  ! " + cv), 100, subsequent_indent="    "))
    if r.get("table"):
        out += ["", c(DIM, f"Data preview ({r['table']['evidence']}):"), _table(r["table"]["columns"], r["table"]["rows"])]
    if show_sql:
        for e in r["evidence"]:
            out += ["", c(DIM, f"-- {e['id']}: {e['purpose']} ({e['row_count']} rows, {e['elapsed_ms']} ms)"), e["sql"]]
    out += ["", c(DIM, f"engine: {r['engine']} | {len(r['evidence'])} queries | {r['elapsed_ms']} ms")]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m data_detective", description="Ask the marketing data a question.")
    ap.add_argument("question", nargs="*", help="question to answer; omit for an interactive session")
    ap.add_argument("--sql", action="store_true", help="print the validated SQL behind each piece of evidence")
    ap.add_argument("--json", action="store_true", help="print the raw JSON response")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args(argv)
    color = sys.stdout.isatty() and not args.no_color
    detective = DataDetective()

    def answer(q: str, history: list[str]) -> None:
        r = detective.ask(q, history)
        print(json.dumps(r, default=str) if args.json else render(r, args.sql, color))

    if args.question:
        answer(" ".join(args.question), [])
        return 0
    print(f"Data Detective ({detective.brain.name}). Ask a question; follow-ups use the conversation. "
          "Commands: /new resets context, /quit exits.")
    history: list[str] = []
    while True:
        try:
            q = input("\n? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if q in ("/quit", "/exit", "quit", "exit"):
            return 0
        if q == "/new":
            history = []
            print("Context cleared.")
            continue
        if len(q) < 3:
            continue
        answer(q, history)
        history = (history + [q])[-5:]


if __name__ == "__main__":
    sys.exit(main())
