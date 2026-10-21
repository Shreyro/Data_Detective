"""Claude 'brain': the LLM plans each SQL step and writes the evidence-cited explanation.

Same interface as OfflineBrain. The LLM never touches the database directly: every SQL string
goes through validator.validate() and a read-only DuckDB connection in agent.py.
"""
from __future__ import annotations

import json
import re

from .config import MODEL
from .models import Narrative, PlanContext, Proposal

PLAN_SYSTEM = """You are Data Detective, an autonomous analytics agent for a marketing team.
You answer a business question by running a SHORT sequence of DuckDB SQL queries, one per turn.

Reply with ONE JSON object and nothing else:
  {"action": "query", "purpose": "<short human-readable goal>", "sql": "<one SELECT statement>"}
or, when the evidence is sufficient:
  {"action": "finish"}

Rules:
- Query only the table `marketing_performance`. One SELECT (CTEs allowed). No DDL/DML, no file functions.
- Compute ratios from SUMs (e.g. SUM(revenue) / NULLIF(SUM(spend), 0)), never AVG of per-row ratios.
- Alias every output column with a clear snake_case name. Use `month`/`quarter`/`week` columns for time grouping.
- Use at most {max_steps} queries in total. Never repeat a query.
- For "why / what changed / drop / spike" questions: (1) monthly series to locate the anomaly,
  (2) break the anomalous period down by channel or region versus the prior period,
  (3) drill into the biggest driver. Compare against the same period last year to rule out seasonality.
- For plain ranking questions: run the ranking, then (if useful) a monthly trend for context.
- If a previous query failed, fix it using the error message."""

NARRATE_SYSTEM = """You are Data Detective writing up your findings for a business reader.
Use ONLY the evidence tables provided. Reply with ONE JSON object and nothing else:
{"headline": "<one sentence answer>",
 "summary": "<2-3 sentences>",
 "findings": [{"text": "<one finding with concrete numbers>", "evidence": ["E1"]}],
 "caveats": ["<limitations / what the data cannot tell>"],
 "highlight": {"evidence": "E1", "x": "YYYY-MM-DD"}}   // optional: a date to mark on a time-series chart

Rules:
- 3-5 findings. Every number must appear in, or be directly computed from, the cited evidence tables.
- Cite evidence ids that support each finding. Format money like $1.23M and ratios like 3.21x.
- Describe where/when something changed; do not claim causes the data cannot show - label them hypotheses.
- Keep each finding under 40 words."""


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError(f"No JSON object in model reply: {text[:120]!r}")
    return json.loads(m.group(0))


def _evidence_block(ctx: PlanContext, rows: int) -> str:
    if not ctx.evidence:
        return "(none yet)"
    out = []
    for e in ctx.evidence:
        out.append(f"[{e.id}] {e.purpose}\nSQL:\n{e.sql}\nRows ({len(e.df)} total, first {min(rows, len(e.df))}):\n{e.df.head(rows).to_csv(index=False)}")
    return "\n".join(out)


class ClaudeBrain:
    def __init__(self, model: str = MODEL, client=None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client, self.model = client, model
        self.name = model

    def _ask(self, system: str, user: str, max_tokens: int = 1800) -> str:
        msg = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

    def next_query(self, ctx: PlanContext) -> Proposal | None:
        user = f"Question: {ctx.question}\n\nSchema:\n{ctx.schema_prompt}\n\nEvidence so far:\n{_evidence_block(ctx, 15)}\n\n"
        user += f"This is query {len(ctx.evidence) + 1} of at most {ctx.max_steps}.\n"
        if ctx.last_error:
            user += f"\nYour previous SQL was rejected.\nSQL:\n{ctx.last_error['sql']}\nProblems: {'; '.join(ctx.last_error['errors'])}\nReturn a corrected query.\n"
        reply = _extract_json(self._ask(PLAN_SYSTEM.replace("{max_steps}", str(ctx.max_steps)), user))
        if reply.get("action") == "finish" and ctx.evidence:
            return None
        sql = reply.get("sql")
        if not sql:
            raise ValueError("Model returned no SQL.")
        return Proposal(sql=sql, purpose=reply.get("purpose") or "Analysis query")

    def narrate(self, ctx: PlanContext) -> Narrative:
        user = f"Question: {ctx.question}\n\nEvidence:\n{_evidence_block(ctx, 40)}"
        r = _extract_json(self._ask(NARRATE_SYSTEM, user, 2000))
        hl = r.get("highlight") or {}
        return Narrative(
            headline=str(r.get("headline", "")), summary=str(r.get("summary", "")),
            findings=[{"text": str(f.get("text", "")), "evidence": list(f.get("evidence", []))} for f in r.get("findings", [])],
            caveats=[str(c) for c in r.get("caveats", [])],
            highlights={hl["evidence"]: hl["x"]} if hl.get("evidence") and hl.get("x") else {},
        )
