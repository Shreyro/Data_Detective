"""FastAPI app. Run:  uvicorn data_detective.api:app --reload"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .agent import DataDetective

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Data Detective", version="1.0.0",
              description="Autonomous analytics agent over synthetic marketing data.")
_agent: DataDetective | None = None


def agent() -> DataDetective:
    global _agent
    if _agent is None:
        _agent = DataDetective()
    return _agent


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=config.MAX_QUESTION_CHARS)


EXAMPLES = [
    "Show me top sales by region",
    "Why did ROAS drop in Q3 2025?",
    "What happened to Email CTR in March 2024?",
    "Which campaigns have the highest CAC in 2025?",
    "Monthly revenue trend by channel",
    "Compare conversion rate by device",
]


@app.get("/api/health")
def health():
    return {"status": "ok", "engine": agent().brain.name}


@app.get("/api/schema")
def schema():
    wh = agent().wh
    return {"columns": wh.schema(), "profile": wh.profile()}


@app.get("/api/examples")
def examples():
    return EXAMPLES


@app.post("/api/ask")
def ask(req: AskRequest):
    try:
        return agent().ask(req.question.strip())
    except Exception as e:  # never leak a stack trace to the browser
        raise HTTPException(status_code=500, detail=f"Agent error: {type(e).__name__}: {e}")


@app.get("/vendor/plotly.min.js", include_in_schema=False)
def plotly_js():
    """Serve the plotly.js bundle that ships inside the plotly Python package (no CDN needed)."""
    import plotly

    return FileResponse(Path(plotly.__file__).parent / "package_data" / "plotly.min.js", media_type="text/javascript")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
