import json
from types import SimpleNamespace

import pandas as pd

from data_detective.agent import DataDetective
from data_detective.claude import ClaudeBrain
from data_detective.evidence import verify_claim
from data_detective.models import Evidence


def test_ranking_question(agent):
    r = agent.ask("Show me top sales by region")
    assert r["ok"] and r["evidence"][0]["columns"][0] == "region"
    assert r["charts"] and all(f["verified"] for f in r["findings"])


def test_diagnosis_finds_planted_anomaly(agent):
    r = agent.ask("Why did ROAS drop in Q3 2025?")
    assert r["ok"] and len(r["evidence"]) == 3
    text = " ".join(f["text"] for f in r["findings"])
    assert "Jul 2025" in text and "Paid Social" in text      # planted: Paid Social fatigue from Jul 2025
    assert all(f["verified"] for f in r["findings"])


def test_email_anomaly_march_2024(agent):
    r = agent.ask("Why did ROAS drop in March 2024?")
    assert "Email" in " ".join(f["text"] for f in r["findings"])


def test_unintelligible_question(agent):
    assert agent.ask("asdf qwerty")["ok"] is False


def test_user_text_is_never_executed_as_sql(agent):
    r = agent.ask("DROP TABLE campaigns; revenue by channel")
    assert r["ok"]  # rule engine builds its own SQL; nothing destructive can run
    assert all(e["sql"].lstrip().upper().startswith(("SELECT", "WITH")) for e in r["evidence"])


def test_grounding_flags_hallucinated_numbers():
    ev = Evidence("E1", "x", "", pd.DataFrame({"region": ["A", "B"], "rev": [100.0, 50.0]}), 1)
    assert verify_claim("A has 100 in revenue and is 100% above B.", [ev])[0]
    ok, bad = verify_claim("A earned $777.5 in revenue.", [ev])
    assert not ok and bad


# ---- Claude brain with a scripted fake client (no network) -------------------------------
class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls += 1
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.replies.pop(0))])


def test_claude_brain_repairs_bad_sql_and_flags_hallucination(wh):
    replies = [
        json.dumps({"action": "query", "purpose": "bad", "sql": "SELECT channel, SUM(rev) FROM marketing_performance GROUP BY 1"}),
        json.dumps({"action": "query", "purpose": "Revenue by channel", "sql": "SELECT channel, SUM(revenue) AS total_revenue FROM marketing_performance GROUP BY 1 ORDER BY 2 DESC"}),
        json.dumps({"action": "finish"}),
        json.dumps({"headline": "Email wins", "summary": "s", "caveats": [],
                    "findings": [{"text": "Email generated $999.99B in revenue.", "evidence": ["E1"]}]}),
    ]
    a = DataDetective(wh, ClaudeBrain(client=FakeClient(replies)))
    r = a.ask("revenue by channel")
    assert r["ok"] and len(r["evidence"]) == 1
    assert any(t["status"] == "retry" for t in r["trace"])             # repair loop fired
    assert r["findings"][0]["verified"] is False                        # fabricated number caught


def test_falls_back_to_offline_when_llm_fails(wh):
    class Boom:
        messages = SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("no network")))

    a = DataDetective(wh, ClaudeBrain(client=Boom()))
    r = a.ask("Show me top sales by region")
    assert r["ok"] and r["engine"] == "offline-rules"
    assert any(t["status"] == "warn" for t in r["trace"])
