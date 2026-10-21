"""Semantic layer: one definition of every KPI, plus number formatting."""
from __future__ import annotations

# kind drives formatting; bad_when_up flips the 'worsening' direction for diagnosis
METRICS: dict[str, dict] = {
    "revenue":     dict(num="revenue", den=None, kind="money", label="revenue", additive=True,
                        aliases=["revenue", "sales", "income", "turnover"]),
    "spend":       dict(num="spend", den=None, kind="money", label="spend", additive=True,
                        aliases=["spend", "cost", "budget", "investment"]),
    "impressions": dict(num="impressions", den=None, kind="count", label="impressions", additive=True,
                        aliases=["impression", "reach", "sends"]),
    "clicks":      dict(num="clicks", den=None, kind="count", label="clicks", additive=True,
                        aliases=["clicks"]),
    "conversions": dict(num="conversions", den=None, kind="count", label="conversions", additive=True,
                        aliases=["conversions", "orders", "purchases"]),
    "roas":        dict(num="revenue", den="spend", kind="ratio", label="ROAS", additive=False,
                        aliases=["roas", "return on ad spend", "return on spend"]),
    "ctr":         dict(num="clicks", den="impressions", kind="pct", label="CTR", additive=False,
                        aliases=["ctr", "click-through", "click through"]),
    "cvr":         dict(num="conversions", den="clicks", kind="pct", label="conversion rate", additive=False,
                        aliases=["conversion rate", "cvr"]),
    "cpc":         dict(num="spend", den="clicks", kind="money", label="CPC", additive=False, bad_when_up=True,
                        aliases=["cpc", "cost per click"]),
    "cac":         dict(num="spend", den="conversions", kind="money", label="CAC", additive=False, bad_when_up=True,
                        aliases=["cac", "cost per acquisition", "acquisition cost", "cost per conversion", "cpa"]),
}
# check specific/compound names before generic ones ("conversion rate" before "conversions")
MATCH_ORDER = ["cvr", "ctr", "roas", "cac", "cpc", "revenue", "spend", "impressions", "clicks", "conversions"]


def expr(metric: str, cond: str | None = None) -> str:
    m = METRICS[metric]
    s = (lambda c: f"SUM({c}) FILTER (WHERE {cond})") if cond else (lambda c: f"SUM({c})")
    return f"{s(m['num'])} / NULLIF({s(m['den'])}, 0)" if m["den"] else s(m["num"])


def alias(metric: str) -> str:
    return f"total_{metric}" if METRICS[metric]["additive"] else metric


def fmt(v: float, kind: str) -> str:
    if v is None or v != v:
        return "n/a"
    a = abs(v)
    sign = "-" if v < 0 else ""
    if kind == "pct":
        return f"{v * 100:.2f}%"
    if kind == "ratio":
        return f"{v:.2f}x"
    pre = "$" if kind == "money" else ""
    if a >= 1e9:
        return f"{sign}{pre}{a / 1e9:.2f}B"
    if a >= 1e6:
        return f"{sign}{pre}{a / 1e6:.2f}M"
    if a >= 1e4:
        return f"{sign}{pre}{a / 1e3:.1f}K"
    return f"{sign}{pre}{a:,.2f}" if kind == "money" else f"{sign}{a:,.0f}"
