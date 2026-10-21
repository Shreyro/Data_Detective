"""The agent loop: plan -> validate -> execute -> (repair) -> explain -> verify."""
from __future__ import annotations

import time

import duckdb

from . import config
from .db import Warehouse
from .evidence import evidence_to_json, verify_claim, df_to_records
from .models import Evidence, PlanContext
from .offline import OfflineBrain
from .validator import validate
from .viz import make_chart


def make_brain():
    if config.FORCE_OFFLINE or not __import__("os").getenv("ANTHROPIC_API_KEY"):
        return OfflineBrain()
    from .claude import ClaudeBrain

    return ClaudeBrain()


class DataDetective:
    def __init__(self, warehouse: Warehouse | None = None, brain=None):
        self.wh = warehouse or Warehouse()
        self.brain = brain or make_brain()
        self.fallback = OfflineBrain()

    def ask(self, question: str) -> dict:
        t_start = time.perf_counter()
        trace: list[dict] = []
        evidence: list[Evidence] = []
        brain = self.brain
        engine = brain.name

        def log(step: str, status: str, detail: str, ms: float | None = None):
            trace.append({"step": step, "status": status, "detail": detail, "ms": None if ms is None else round(ms, 1)})

        log("Inspect dataset", "ok", f"{self.wh.profile()['rows']:,} rows, {self.wh.profile()['date_min']} to {self.wh.profile()['date_max']}")
        ctx = PlanContext(question, self.wh.schema_prompt(), self.wh.profile(), evidence, 0, config.MAX_STEPS)

        while len(evidence) < config.MAX_STEPS:
            ctx.last_error, proposal, accepted = None, None, False
            for attempt in range(config.MAX_REPAIRS + 1):
                t0 = time.perf_counter()
                try:
                    proposal = brain.next_query(ctx)
                except Exception as e:  # LLM/network/JSON failure
                    if brain is not self.fallback and not evidence:
                        log("Plan", "warn", f"{engine} unavailable ({type(e).__name__}); falling back to offline planner")
                        brain, engine = self.fallback, self.fallback.name
                        continue
                    log("Plan", "error", f"Planner failed: {e}")
                    proposal = None
                    break
                if proposal is None:
                    break
                v = validate(proposal.sql, self.wh)
                if not v.ok:
                    log("Validate SQL", "retry", "; ".join(v.errors))
                    ctx.last_error = {"sql": proposal.sql, "errors": v.errors}
                    continue
                try:
                    df, ms = self.wh.run(v.sql)
                except duckdb.Error as e:
                    log("Execute SQL", "retry", str(e).splitlines()[0])
                    ctx.last_error = {"sql": v.sql, "errors": [str(e).splitlines()[0]]}
                    continue
                ev = Evidence(f"E{len(evidence) + 1}", proposal.purpose, v.sql, df, ms)
                evidence.append(ev)
                log(f"Query {ev.id}", "ok", f"{proposal.purpose} -> {len(df)} rows" + (f" ({'; '.join(v.warnings)})" if v.warnings else ""), ms)
                accepted = True
                break
            if proposal is None or not accepted:
                break

        if not evidence:
            return {"question": question, "engine": engine, "ok": False, "headline": "I couldn't answer that",
                    "summary": "No valid query could be produced for this question. Try rephrasing, e.g. 'Top sales by region' or 'Why did ROAS drop in Q3 2025?'.",
                    "findings": [], "caveats": [], "evidence": [], "charts": [], "table": None, "trace": trace,
                    "elapsed_ms": round((time.perf_counter() - t_start) * 1000)}

        t0 = time.perf_counter()
        try:
            narrative = brain.narrate(ctx)
            if not narrative.findings:
                raise ValueError("empty narrative")
        except Exception as e:
            if brain is self.fallback:
                raise
            log("Explain", "warn", f"{engine} narration failed ({type(e).__name__}); using offline explainer")
            narrative, engine = self.fallback.narrate(ctx), self.fallback.name
        by_id = {e.id: e for e in evidence}
        findings, unverified = [], 0
        for f in narrative.findings:
            cited = [by_id[i] for i in f["evidence"] if i in by_id] or evidence
            ok, bad = verify_claim(f["text"], cited)
            unverified += 0 if ok else 1
            findings.append({**f, "evidence": [e.id for e in cited], "verified": ok, "unverified_numbers": bad})
        log("Explain & verify", "ok" if not unverified else "warn",
            f"{len(findings) - unverified}/{len(findings)} findings fully traceable to query results", (time.perf_counter() - t0) * 1000)

        charts = [c for e in evidence if (c := make_chart(e, narrative.highlights.get(e.id)))]
        return {
            "question": question, "engine": engine, "ok": True,
            "headline": narrative.headline, "summary": narrative.summary,
            "findings": findings, "caveats": narrative.caveats,
            "evidence": [evidence_to_json(e) for e in evidence],
            "charts": charts,
            "table": {"evidence": evidence[0].id, **df_to_records(evidence[0].df, 8)},
            "trace": trace, "elapsed_ms": round((time.perf_counter() - t_start) * 1000),
        }
