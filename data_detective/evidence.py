"""Traceable evidence: serialisation and a groundedness check for every claim.

Each finding cites evidence ids (E1, E2...). `verify_claim` extracts every number in the
finding text and checks it against values that appear in - or can be directly derived from -
the cited result tables (cells, sums, shares, period-over-period changes, ...), using the
rounding precision of the number as tolerance. Findings with numbers we cannot trace are
flagged `verified: false` in the UI, which catches LLM hallucinations.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from .models import Evidence

PREVIEW_ROWS = 50


def df_to_records(df: pd.DataFrame, limit: int = PREVIEW_ROWS) -> dict:
    out = json.loads(df.head(limit).to_json(orient="split", date_format="iso", index=False))
    # trim midnight timestamps to plain dates
    out["data"] = [[(c[:10] if isinstance(c, str) and c.endswith("T00:00:00.000") else c) for c in row] for row in out["data"]]
    return {"columns": out["columns"], "rows": out["data"]}


def evidence_to_json(ev: Evidence) -> dict:
    return {
        "id": ev.id, "purpose": ev.purpose, "sql": ev.sql,
        "row_count": int(len(ev.df)), "elapsed_ms": round(ev.elapsed_ms, 1),
        **df_to_records(ev.df),
    }


# ---------------------------------------------------------------- grounding
_STRIP = [
    re.compile(r"\[E\d+(?:\s*,\s*E\d+)*\]"),            # citation tags
    re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b"),          # ISO dates
    re.compile(r"\b[A-Z][a-z]{2} \d{4}\b"),             # "Jul 2025"
    re.compile(r"\bQ[1-4]\b"),                          # quarters
    re.compile(r"\bE\d+\b"),
]
_NUM = re.compile(r"(?<![\w.])\$?(\d[\d,]*(?:\.\d+)?)\s*(K|M|B|%|x|pp|pts)?(?![\w])", re.I)
_MULT = {"k": 1e3, "m": 1e6, "b": 1e9}


def _candidates(df: pd.DataFrame) -> np.ndarray:
    vals: list[np.ndarray] = [np.array([float(len(df))])]
    num = df.select_dtypes("number").iloc[:200]
    cols = {}
    for c in num.columns:
        a = num[c].dropna().to_numpy(dtype=float)
        if a.size == 0:
            continue
        cols[c] = a
        vals += [np.abs(a), np.abs(a) * 100, np.array([a.sum(), a.mean(), a.max(), a.min()]), np.abs(a.sum()) * np.array([1, 100])]
        if a.size > 1:
            prev, cur = a[:-1], a[1:]
            d = cur - prev
            with np.errstate(all="ignore"):
                pc = np.where(prev != 0, d / prev, np.nan)
                pc2 = np.where(cur != 0, -d / cur, np.nan)
                ratio = np.where(cur != 0, prev / cur, np.nan)
            vals += [np.abs(d), np.abs(d) * 100, np.abs(pc) * 100, np.abs(pc2) * 100, np.abs(ratio), np.abs(1 / ratio)]
            if a[0] != 0:
                vals.append(np.array([abs(a[-1] / a[0] - 1) * 100, abs(a[-1] - a[0]), abs(a[-1] / a[0])]))
            if a.min() != 0:
                vals.append(np.array([a.max() / a.min()]))
            # seasonally adjusted change: pct change minus the same change 12 rows earlier
            if a.size > 13:
                with np.errstate(all="ignore"):
                    pcs = np.where(prev != 0, d / prev, np.nan)
                vals.append(np.abs((pcs[12:] - pcs[:-12]) * 100))
                vals.append(np.abs(pcs[:-12] * 100))
        if (a >= 0).all() and a.sum() > 0:
            s = np.sort(a)[::-1]
            vals += [a / a.sum() * 100, np.cumsum(s) / a.sum() * 100, np.cumsum(s) ]
            # share of *total change* style comparisons
    names = list(cols)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            ax, ay = cols[x], cols[y]
            n = min(len(ax), len(ay))
            d = ay[:n] - ax[:n]
            with np.errstate(all="ignore"):
                pc = np.where(ax[:n] != 0, d / ax[:n], np.nan)
                rr = np.where(ax[:n] != 0, ay[:n] / ax[:n], np.nan)
            vals += [np.abs(d), np.abs(d) * 100, np.abs(pc) * 100, np.abs(rr), np.abs(d).sum(keepdims=True), np.abs(d.sum(keepdims=True)),
                     np.abs(d) / max(np.abs(d.sum()), 1e-12) * 100]
    out = np.concatenate([np.atleast_1d(v) for v in vals])
    return out[np.isfinite(out)]


def verify_claim(text: str, evidence: list[Evidence]) -> tuple[bool, list[str]]:
    """Return (all numbers traceable, list of untraceable number tokens)."""
    t = text
    for rx in _STRIP:
        t = rx.sub(" ", t)
    if not evidence:
        return False, ["no evidence cited"]
    cands = np.concatenate([_candidates(e.df) for e in evidence])
    bad: list[str] = []
    for m in _NUM.finditer(t):
        raw, suffix = m.group(1), (m.group(2) or "").lower()
        token = m.group(0).strip()
        has_unit = bool(suffix) or token.startswith("$")
        if "," in raw:
            v = float(raw.replace(",", ""))
        else:
            v = float(raw)
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        if not has_unit and decimals == 0 and "," not in raw and (v <= 12 or 1900 <= v <= 2100):
            continue
        mult = _MULT.get(suffix, 1.0)
        value, tol = v * mult, 0.5 * 10 ** (-decimals) * mult * 1.001 + 1e-9
        targets = [value]
        if suffix in ("%", "pp", "pts"):
            targets.append(value)  # candidates already include x100 forms
        if not any(np.any(np.abs(cands - tg) <= tol) for tg in targets):
            bad.append(token)
    return (not bad), bad
