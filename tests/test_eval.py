"""Eval set for the offline brain: question -> expected interpretation and answer properties.

Runs with no API key. Every case also requires all findings to be traceable to query results.
"""
import pytest

from data_detective.offline import parse, resolve


# question, expected mode, expected metric, substrings that must appear in the findings/headline
CASES = [
    ("Show me top sales by region", "rank", "revenue", ["North America"]),
    ("Which channel is most efficient?", "rank", "roas", ["Email has the highest ROAS"]),
    ("Bottom 3 regions by ROAS", "rank", "roas", ["lowest ROAS"]),
    ("Which campaigns have the highest CAC in 2025?", "rank", "cac", ["Display"]),
    ("Monthly revenue trend by channel", "trend", "revenue", ["Email was highest"]),
    ("Revenue by year", "trend", "revenue", ["in 2023", "in 2025"]),
    ("Quarterly ROAS", "trend", "roas", ["Q1 2023"]),
    ("Weekly clicks for Email in 2024", "trend", "clicks", ["the week of 2024-"]),
    ("What is total revenue?", "total", "revenue", ["Total revenue was $123.50M"]),
    ("How much did we spend in 2024?", "total", "spend", ["Total spend (2024)"]),
    ("Compare 2024 vs 2025 revenue by channel", "compare", "revenue", ["in 2024", "in 2025"]),
    ("Show me CTR year over year by channel", "compare", "ctr", ["2024", "2025"]),
    ("Why did ROAS drop in Q3 2025?", "diagnose", "roas", ["Jul 2025", "Paid Social", "APAC and LATAM"]),
    ("Why did CAC spike in Q3 2025?", "diagnose", "cac", ["CAC rose", "Paid Social"]),
    ("What happened to Email CTR in March 2024?", "diagnose", "ctr", ["Mar 2024", "broad-based: every region"]),
    ("Why did revenue spike in November 2024?", "diagnose", "revenue", ["mostly seasonal"]),
]


@pytest.mark.parametrize("question,mode,metric,expect", CASES, ids=[c[0] for c in CASES])
def test_eval_case(agent, wh, question, mode, metric, expect):
    s = parse(question, wh.profile())
    assert (s.mode, s.metric) == (mode, metric)
    r = agent.ask(question)
    assert r["ok"], r["summary"]
    text = r["headline"] + " " + r["summary"] + " " + " ".join(f["text"] for f in r["findings"])
    for e in expect:
        assert e in text, f"{e!r} not in: {text}"
    bad = [(f["text"], f["unverified_numbers"]) for f in r["findings"] if not f["verified"]]
    assert not bad, bad
    assert all(e["sql"].lstrip().upper().startswith(("SELECT", "WITH")) for e in r["evidence"])


def test_email_incident_is_not_blamed_on_a_small_segment(agent):
    """The Email CTR collapse hit every region/device equally; the agent must not single out one segment."""
    r = agent.ask("What happened to Email CTR in March 2024?")
    text = " ".join(f["text"] for f in r["findings"])
    assert "concentrated" not in text and "broad-based" in r["summary"]


def test_seasonal_spike_is_not_reported_as_incident(agent):
    r = agent.ask("Why did revenue spike in November 2024?")
    assert "seasonal" in r["summary"] and not any("Hypothesis" in c for c in r["caveats"])


# ---- follow-up questions -----------------------------------------------------------------
def test_followup_filter_dimension_and_time(wh):
    p = wh.profile()
    h = ["Revenue by region"]
    s = resolve("what about Europe only?", h, p)
    assert s.filters == {"region": "Europe"} and s.metric == "revenue" and s.dim != "region"
    s = resolve("break that down by device", h + ["what about Europe only?"], p)
    assert (s.mode, s.dim, s.filters) == ("rank", "device", {"region": "Europe"})
    s = resolve("and in 2024?", h + ["what about Europe only?", "break that down by device"], p)
    assert (s.dim, s.year, s.filters) == ("device", 2024, {"region": "Europe"})


def test_followup_switches_metric_and_keeps_worsening_direction(wh):
    s = resolve("what about CAC?", ["Why did ROAS drop in Q3 2025?"], wh.profile())
    assert (s.mode, s.metric, s.direction, s.year, s.quarter) == ("diagnose", "cac", "up", 2025, 3)


def test_new_standalone_question_ignores_history(wh):
    s = resolve("Monthly ROAS trend", ["Revenue in Europe by device"], wh.profile())
    assert s.filters == {} and s.mode == "trend" and s.metric == "roas"


def test_followup_end_to_end(agent):
    r = agent.ask("what about Europe only?", ["Show me top sales by region"])
    assert r["ok"] and "(Europe)" in r["headline"]
    assert any("follow-up" in t["detail"] for t in r["trace"] if t["step"] == "Understand question")


def test_small_talk_is_not_answered_with_previous_question(agent):
    assert agent.ask("thanks", ["Revenue by region"])["ok"] is False
