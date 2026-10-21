from fastapi.testclient import TestClient

from data_detective import api
from data_detective.agent import DataDetective
from data_detective.offline import OfflineBrain


def client(wh):
    api._agent = DataDetective(wh, OfflineBrain())
    return TestClient(api.app)


def test_endpoints(wh):
    c = client(wh)
    assert c.get("/api/health").json()["status"] == "ok"
    assert c.get("/api/schema").json()["profile"]["rows"] > 100_000
    assert len(c.get("/api/examples").json()) >= 4
    assert "Data Detective" in c.get("/").text
    assert c.get("/vendor/plotly.min.js").status_code == 200


def test_ask(wh):
    r = client(wh).post("/api/ask", json={"question": "Top channels by roas"})
    j = r.json()
    assert r.status_code == 200 and j["ok"] and j["charts"] and j["trace"]


def test_input_validation(wh):
    c = client(wh)
    assert c.post("/api/ask", json={"question": "hi"}).status_code == 422
    assert c.post("/api/ask", json={"question": "x" * 600}).status_code == 422
