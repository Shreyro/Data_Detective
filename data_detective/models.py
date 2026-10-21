"""Small shared data classes."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Evidence:
    id: str
    purpose: str
    sql: str
    df: pd.DataFrame
    elapsed_ms: float


@dataclass
class Proposal:
    sql: str
    purpose: str


@dataclass
class Narrative:
    headline: str
    summary: str
    findings: list[dict] = field(default_factory=list)   # {"text": str, "evidence": ["E1"]}
    caveats: list[str] = field(default_factory=list)
    highlights: dict = field(default_factory=dict)       # evidence id -> x value to mark on chart


@dataclass
class PlanContext:
    question: str
    schema_prompt: str
    profile: dict
    evidence: list[Evidence]
    step: int
    max_steps: int
    last_error: dict | None = None   # {"sql": str, "errors": [str]}
