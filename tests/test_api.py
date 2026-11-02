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


def test_ask_with_history(wh):
    c = client(wh)
    j = c.post("/api/ask", json={"question": "break that down by device", "history": ["Revenue in Europe"]}).json()
    assert j["ok"] and j["evidence"][0]["columns"][0] == "device" and "Europe" in j["headline"]
    assert c.post("/api/ask", json={"question": "revenue", "history": ["x" * 600]}).status_code == 422


def test_cli_one_shot(wh, capsys, monkeypatch):
    from data_detective import __main__ as cli

    monkeypatch.setattr(cli, "DataDetective", lambda: DataDetective(wh, OfflineBrain()))
    assert cli.main(["--sql", "--no-color", "Top", "sales", "by", "region"]) == 0
    out = capsys.readouterr().out
    assert "North America has the highest revenue" in out and "[ok]" in out and "SELECT" in out
