"""Offline 'brain': a deterministic, rule-based planner + narrator.

It lets the whole project run (and be tested in CI) with no API key. It turns a natural-language
question into a Spec, generates SQL from the semantic layer, and runs a short investigation:

  rank      "top sales by region"            ranking (+ share of total) -> monthly context trend
  trend     "monthly revenue by channel"     time series -> split by a dimension
  total     "how much did we spend in 2024"  KPI summary -> breakdown of the main metric
  compare   "2024 vs 2025 revenue by channel" period-vs-period by dimension -> overall totals
  diagnose  "why did ROAS drop in Q3 2025?"  find the (seasonally adjusted) anomalous month ->
                                             attribute the change to segments -> drill into the
                                             driver, or confirm the change is broad-based

Follow-up questions ("what about Europe?", "break that down by device") are resolved against the
earlier questions in the conversation. The Claude brain (claude.py) plugs into the same interface.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

import pandas as pd

from .metrics import METRICS, MATCH_ORDER, alias, expr, fmt
from .models import Narrative, PlanContext, Proposal

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
DIM_WORDS = {
    "channel": "channel", "channels": "channel",
    "region": "region", "regions": "region", "country": "region", "countries": "region",
    "market": "region", "markets": "region", "geo": "region", "geography": "region",
    "device": "device", "devices": "device", "platform": "device", "platforms": "device",
    "campaign": "campaign_name", "campaigns": "campaign_name",
    "objective": "objective", "objectives": "objective", "goal": "objective", "goals": "objective",
}
DRILL_DIMS = ["channel", "region", "device", "objective"]
DIAG = re.compile(r"\b(why|drop|dropped|decline|declined|fell|fall|fallen|decrease|decreased|spike|spiked|anomal\w*|unusual|weird|odd|"
                  r"investigate|explain|diagnose|what happened|went wrong|root cause|outlier\w*)\b")
UP = re.compile(r"\b(spike|spiked|increase|increased|jump|jumped|surge|surged|rose|rise|grew|up|higher)\b")
DOWN = re.compile(r"\b(drop|dropped|decline|declined|fell|fall|fallen|decrease|decreased|down|worse|dip|dipped|lower|tank\w*)\b")
TOTAL = re.compile(r"\b(total|overall|how much|how many|in total|altogether|sum of|kpis?|summary|summari[sz]e|overview|headline numbers)\b")
YOY = re.compile(r"\b(year over year|year-over-year|yoy|vs\.? last year|versus last year|compared to last year|this year vs)\b")
FOLLOWUP = re.compile(r"^\s*(and|what about|how about|now|same|then|also|only|just|instead|ok|okay|what if|but)\b"
                      r"|\b(that|this|those|these|it|them|same)\b")
GRAINS = [
    (r"\b(weekly|by week|per week|week over week|wow)\b", "week"),
    (r"\b(quarterly|by quarter|per quarter|quarter over quarter|qoq)\b", "quarter"),
    (r"\b(yearly|by year|per year|annual\w*|year over year|year-over-year|yoy)\b", "year"),
    (r"\b(daily|by day|per day|day by day)\b", "date"),
    (r"\b(trend|trends|over time|monthly|by month|per month|month over month|mom|growth|evolution|history|timeline|seasonality)\b", "month"),
]
GRAIN_ADJ = {"date": "Daily", "week": "Weekly", "month": "Monthly", "quarter": "Quarterly", "year": "Yearly"}


@dataclass
class Spec:
    metric: str = "revenue"
    mode: str = "rank"                      # rank | trend | total | compare | diagnose
    dim: str | None = None
    grain: str | None = None
    filters: dict = field(default_factory=dict)   # column -> value
    year: int | None = None
    quarter: int | None = None
    month: str | None = None                # 'YYYY-MM-01'
    years: tuple | None = None              # (older, newer) for compare mode
    top: int = 10
    asc: bool = False
    direction: str = "down"                 # down | up | any (diagnose)
    understood: bool = False
    given: set = field(default_factory=set)  # which fields the question stated explicitly


def parse(question: str, profile: dict) -> Spec:
    """Parse one question on its own (no conversation context)."""
    q = question.lower()
    s = Spec()
    for m in MATCH_ORDER:
        if any(re.search(rf"\b{re.escape(a)}", q) for a in METRICS[m]["aliases"]):
            s.metric = m
            s.given.add("metric")
            break
    # dimension: prefer the word after by/per/across/each
    m = re.search(r"\b(?:by|per|across|each|split by|broken down by|between)\s+(\w+)", q)
    if m and m.group(1) in DIM_WORDS:
        s.dim = DIM_WORDS[m.group(1)]
    else:
        for w, col in DIM_WORDS.items():
            if re.search(rf"\b{w}\b", q):
                s.dim = col
                break
    if s.dim:
        s.given.add("dim")
    # filters from known category values
    for col, values in profile["categories"].items():
        for v in values:
            if re.search(rf"\b{re.escape(v.lower())}\b", q):
                s.filters[col] = v
    if s.dim in s.filters and re.search(r"\b(by|per|across|each)\b", q):
        s.filters.pop(s.dim)
    if s.filters:
        s.given.add("filters")
    # time
    years = sorted({int(y) for y in re.findall(r"\b(20\d\d)\b", q)})
    if years:
        s.year = years[-1] if len(years) == 1 else None
        if len(years) >= 2:
            s.years = (years[0], years[-1])
    qq = re.search(r"\bq([1-4])\b", q)
    if qq:
        s.quarter = int(qq.group(1))
    mm = re.search(r"\b(" + "|".join(MONTHS) + r")[a-z]*\.?\s+(20\d\d)\b", q)
    if mm:
        s.month = f"{mm.group(2)}-{MONTHS.index(mm.group(1)) + 1:02d}-01"
        s.year = s.quarter = s.years = None
    if years or qq or mm:
        s.given.add("time")
    # ranking options
    t = re.search(r"\b(?:top|bottom|worst|best|lowest|highest)\s+(\d+)\b", q)
    if t:
        s.top = max(1, min(int(t.group(1)), 50))
        s.given.add("top")
    s.asc = bool(re.search(r"\b(bottom|worst|lowest|least|weakest|smallest)\b", q))
    if METRICS[s.metric].get("bad_when_up"):          # for CAC/CPC, "worst" = highest, "best" = lowest
        if re.search(r"\b(worst|highest|most expensive)\b", q):
            s.asc = False
        elif re.search(r"\b(best|cheapest|efficient|lowest)\b", q):
            s.asc = True
    if re.search(r"\b(top|bottom|worst|best|lowest|highest|least|weakest|smallest|cheapest)\b", q):
        s.given.add("order")
    for rx, g in GRAINS:
        if re.search(rx, q):
            s.grain = g
            s.given.add("grain")
            break
    # direction for diagnosis
    up, down = bool(UP.search(q)), bool(DOWN.search(q))
    s.direction = "up" if up and not down else "down" if down else "any"
    # mode
    yoy = bool(YOY.search(q))
    if DIAG.search(q):
        s.mode, s.grain = "diagnose", "month"
        s.given.add("diag")
    elif s.years or (yoy and (s.dim or s.year)):
        s.mode = "compare"
        if not s.years:
            last = s.year or int(str(profile["date_max"])[:4])
            s.years, s.year = (last - 1, last), None
        s.dim = s.dim or "channel"
        s.given.add("compare")
    elif s.grain:
        s.mode = "trend"
    elif TOTAL.search(q) and "dim" not in s.given:
        s.mode = "total"
        s.given.add("total")
    else:
        s.mode, s.dim = "rank", s.dim or "channel"
    s.understood = bool(s.given)
    return s


def is_followup(question: str, s: Spec) -> bool:
    return bool(FOLLOWUP.search(question.lower())) or ("metric" not in s.given and s.understood)


def merge(prev: Spec, cur: Spec) -> Spec:
    """Apply a follow-up question on top of the previous question's spec."""
    s = replace(prev, filters=dict(prev.filters), given=set(prev.given) | cur.given, understood=True)
    if "metric" in cur.given:
        s.metric = cur.metric
        # "why did ROAS drop?" -> "what about CAC?": keep looking for a *worsening*, which for cost metrics is a rise
        if prev.direction != "any" and bool(METRICS[prev.metric].get("bad_when_up")) != bool(METRICS[cur.metric].get("bad_when_up")):
            s.direction = "up" if prev.direction == "down" else "down"
    if "dim" in cur.given:
        s.dim = cur.dim
        s.filters.pop(cur.dim, None)               # "by region" after "...in Europe" means all regions
    for col, v in cur.filters.items():
        s.filters[col] = v
        if s.dim == col and "dim" not in cur.given:
            s.dim = None
    if "time" in cur.given:
        s.year, s.quarter, s.month, s.years = cur.year, cur.quarter, cur.month, cur.years
    if "grain" in cur.given:
        s.grain = cur.grain
    if "top" in cur.given:
        s.top = cur.top
    if "order" in cur.given:
        s.asc = cur.asc
    if {"diag", "compare", "total"} & cur.given:
        s.mode, s.direction = cur.mode, cur.direction
        s.grain = cur.grain if cur.mode == "diagnose" else s.grain
        if cur.mode == "compare":
            s.years, s.year, s.dim = cur.years, None, cur.dim
    elif "grain" in cur.given:
        s.mode = "trend"
    elif "dim" in cur.given:
        s.mode = "trend" if prev.mode == "trend" else "rank"
    elif "time" in cur.given and prev.mode == "compare" and not cur.years:
        s.mode = "rank"
    if s.mode == "rank":
        s.dim = s.dim or next((d for d in DRILL_DIMS if d not in s.filters), "campaign_name")
    if s.mode == "compare" and not s.years:
        s.mode = "rank"
    if s.mode == "trend" and not s.grain:
        s.grain = "month"
    return s


def resolve(question: str, history: list[str], profile: dict) -> Spec:
    """Spec for `question`, folding in earlier questions when it is a follow-up."""
    spec = None
    for q in [*history, question]:
        p = parse(q, profile)
        if spec is not None and spec.understood and p.understood and is_followup(q, p):
            spec = merge(spec, p)
        elif spec is not None and spec.understood and not p.understood and q is not question:
            continue             # earlier small talk ("thanks") does not reset the context
        else:
            spec = p
    return spec


def describe(s: Spec) -> str:
    label = METRICS[s.metric]["label"]
    bits = {
        "rank": f"rank {_dimname(s.dim)} by {label}",
        "trend": f"{label} by {s.grain}" + (f" split by {_dimname(s.dim)}" if s.dim else ""),
        "total": f"total {label} with KPI summary",
        "compare": f"{label} {s.years[0]} vs {s.years[1]} by {_dimname(s.dim)}" if s.years else "",
        "diagnose": f"investigate {'a rise' if s.direction == 'up' else 'a drop' if s.direction == 'down' else 'the largest change'} in {label}",
    }
    return bits[s.mode] + _scope(s)


# ------------------------------------------------------------------ SQL builders
def _lit(v: str) -> str:
    return "'" + v.replace("'", "''") + "'"


def _where(s: Spec, time: bool = True, skip: tuple = ()) -> str:
    parts = [f"{c} = {_lit(v)}" for c, v in s.filters.items() if c not in skip]
    if time:
        if s.month:
            parts.append(f"month = DATE '{s.month}'")
        if s.year and s.quarter:
            parts.append(f"quarter = DATE '{s.year}-{(s.quarter - 1) * 3 + 1:02d}-01'")
        elif s.year:
            parts.append(f"year = {s.year}")
        elif s.quarter:
            parts.append(f"EXTRACT(quarter FROM date) = {s.quarter}")
        if s.years:
            parts.append(f"year IN ({s.years[0]}, {s.years[1]})")
    return ("WHERE " + " AND ".join(parts)) if parts else ""


def _rank_sql(s: Spec) -> str:
    a, e = alias(s.metric), expr(s.metric)
    share = f",\n  ROUND(100 * {e} / SUM({e}) OVER (), 1) AS share_pct" if METRICS[s.metric]["additive"] else ""
    return (
        f"SELECT {s.dim}, {e} AS {a}{share}\nFROM marketing_performance\n{_where(s)}\n"
        f"GROUP BY {s.dim}\nORDER BY {a} {'ASC' if s.asc else 'DESC'}\nLIMIT {s.top}"
    )


def _trend_sql(s: Spec, time: bool = True, by: str | None = None) -> str:
    g, a, e = s.grain or "month", alias(s.metric), expr(s.metric)
    cols = f"{g}, {by}, " if by else f"{g}, "
    return (
        f"SELECT {cols}{e} AS {a}\nFROM marketing_performance\n{_where(s, time=time)}\n"
        f"GROUP BY {g}{', ' + by if by else ''}\nORDER BY {g}{', ' + by if by else ''}"
    )


def _total_sql(s: Spec) -> str:
    keys = [s.metric] + [k for k in ("revenue", "spend", "conversions", "roas", "cac", "ctr", "cvr") if k != s.metric]
    cols = ",\n  ".join(f"{expr(k)} AS {alias(k)}" for k in keys)
    return f"SELECT\n  {cols}\nFROM marketing_performance\n{_where(s)}"


def _compare_sql(s: Spec) -> str:
    y0, y1 = s.years
    p, c = expr(s.metric, f"year = {y0}"), expr(s.metric, f"year = {y1}")
    return (
        f"SELECT {s.dim},\n  {p} AS value_{y0},\n  {c} AS value_{y1},\n  {c} - {p} AS change,\n"
        f"  ROUND(100 * ({c} - {p}) / NULLIF({p}, 0), 1) AS change_pct\n"
        f"FROM marketing_performance\n{_where(s)}\nGROUP BY {s.dim}\nORDER BY change DESC\nLIMIT 25"
    )


def _compare_total_sql(s: Spec) -> str:
    return (
        f"SELECT year, {expr(s.metric)} AS {alias(s.metric)}\nFROM marketing_performance\n{_where(s)}\n"
        f"GROUP BY year\nORDER BY year"
    )


def _delta_sql(s: Spec, dim: str, cur: str, prev: str, direction: str, extra: dict | None = None) -> str:
    """Attribute a month-over-month change to the segments of `dim`.

    Additive metrics: each segment's change is its contribution.
    Ratio metrics: impact = change in the segment's ratio x its share of the denominator, i.e. the
    'rate effect' - so a tiny segment with a noisy ratio does not look like the main driver.
    """
    m = METRICS[s.metric]
    c, p = f"month = DATE '{cur}'", f"month = DATE '{prev}'"
    filt = dict(s.filters)
    filt.update(extra or {})
    where = " AND ".join([f"month IN (DATE '{prev}', DATE '{cur}')"] + [f"{k} = {_lit(v)}" for k, v in filt.items() if k != dim])
    order = "DESC" if direction == "up" else "ASC"
    if m["additive"]:
        return (
            f"SELECT {dim},\n  {expr(s.metric, p)} AS previous,\n  {expr(s.metric, c)} AS current,\n"
            f"  {expr(s.metric, c)} - {expr(s.metric, p)} AS change,\n"
            f"  ROUND(100 * ({expr(s.metric, c)} - {expr(s.metric, p)}) / NULLIF({expr(s.metric, p)}, 0), 1) AS change_pct,\n"
            f"  ROUND(100 * ({expr(s.metric, c)} - {expr(s.metric, p)}) / NULLIF(SUM({expr(s.metric, c)} - {expr(s.metric, p)}) OVER (), 0), 1) AS share_of_change_pct\n"
            f"FROM marketing_performance\nWHERE {where}\nGROUP BY {dim}\n"
            f"ORDER BY change {order}\nLIMIT 10"
        )
    return (
        f"WITH seg AS (\n  SELECT {dim},\n    {expr(s.metric, p)} AS previous,\n    {expr(s.metric, c)} AS current,\n"
        f"    SUM({m['den']}) FILTER (WHERE {c}) AS volume\n"
        f"  FROM marketing_performance\n  WHERE {where}\n  GROUP BY {dim}\n), scored AS (\n"
        f"  SELECT {dim}, previous, current, current - previous AS change,\n"
        f"    ROUND(100 * (current - previous) / NULLIF(previous, 0), 1) AS change_pct,\n"
        f"    ROUND(100 * volume / SUM(volume) OVER (), 1) AS volume_share_pct,\n"
        f"    (current - previous) * volume / SUM(volume) OVER () AS impact\n  FROM seg\n)\n"
        f"SELECT {dim}, previous, current, change, change_pct, volume_share_pct, impact,\n"
        f"  ROUND(100 * impact / NULLIF(SUM(impact) OVER (), 0), 1) AS share_of_change_pct\n"
        f"FROM scored\nORDER BY impact {order}\nLIMIT 10"
    )


# ------------------------------------------------------------------ analysis helpers
def cap(label: str) -> str:
    return label if label.isupper() else label[:1].upper() + label[1:]


def _dimname(dim: str | None) -> str:
    return (dim or "").replace("_name", "")


def _mon(x) -> str:
    return pd.Timestamp(x).strftime("%b %Y")


def _lbl(x, grain: str | None) -> str:
    """Human label for a time bucket. Uses forms the groundedness checker ignores (ISO dates, 'Jul 2025', 'Q3 2025')."""
    if grain == "year":
        return str(int(x))
    ts = pd.Timestamp(x)
    if grain == "quarter":
        return f"Q{ts.quarter} {ts.year}"
    if grain == "week":
        return f"the week of {ts:%Y-%m-%d}"
    if grain == "date":
        return f"{ts:%Y-%m-%d}"
    return ts.strftime("%b %Y")


def pick_anomaly(df: pd.DataFrame, tcol: str, vcol: str, direction: str, spec: Spec) -> dict | None:
    s = df.sort_values(tcol).set_index(tcol)[vcol].astype(float)
    if len(s) < 3:
        return None
    pct = s.pct_change()
    yoy = pct.shift(12)
    adj = pct - yoy
    score = adj if adj.notna().any() else pct
    idx = pd.Series(score.index, index=score.index)
    ok = score.notna()
    if spec.month:
        ok &= idx == pd.Timestamp(spec.month)
    if spec.year:
        ok &= idx.dt.year == spec.year
    if spec.quarter:
        ok &= idx.dt.quarter == spec.quarter
    cand = score[ok]
    if cand.empty:
        return None
    if direction == "up":
        m = cand.idxmax()
    elif direction == "down":
        m = cand.idxmin()
    else:
        m = cand.abs().idxmax()
    i = list(s.index).index(m)
    return dict(month=m, prev=s.index[i - 1], cur_val=float(s[m]), prev_val=float(s.iloc[i - 1]),
                pct=float(pct[m]), yoy=None if pd.isna(yoy[m]) else float(yoy[m]),
                adj=None if pd.isna(adj[m]) else float(adj[m]), seasonally_adjusted=bool(adj.notna().any()))


def _scope(s: Spec) -> str:
    bits = [v for v in s.filters.values()]
    if s.month:
        bits.append(_mon(s.month))
    elif s.year and s.quarter:
        bits.append(f"Q{s.quarter} {s.year}")
    elif s.year:
        bits.append(str(s.year))
    elif s.quarter:
        bits.append(f"Q{s.quarter}")
    return (" (" + ", ".join(bits) + ")") if bits else ""


def _segcol(df: pd.DataFrame) -> str:
    return df.columns[0]


def _broad(df: pd.DataFrame) -> bool:
    """True when every segment moved the same way by a comparable amount (smallest move >= half the largest).

    Then a big segment's large share of the change only reflects its size, not a localised problem."""
    pcts = df["change_pct"].dropna().astype(float) if "change_pct" in df else pd.Series(dtype=float)
    if len(pcts) < 2 or not ((pcts < 0).all() or (pcts > 0).all()):
        return False
    return pcts.abs().min() >= 0.5 * pcts.abs().max()


def _concentrated(df: pd.DataFrame) -> bool:
    """True when the change is localised and one segment explains at least half of it."""
    return (len(df) > 1 and "share_of_change_pct" in df and not _broad(df)
            and float(df["share_of_change_pct"].iloc[0] or 0) >= 50)


def _drivers(df: pd.DataFrame, limit: int = 3, target: float = 75.0) -> int:
    """How many leading segments are needed to explain `target`% of the change."""
    cum = 0.0
    for i, v in enumerate(df["share_of_change_pct"].fillna(0).astype(float).head(limit)):
        cum += v
        if cum >= target:
            return i + 1
    return 1


def _second_dim(s: Spec, dim1: str) -> str | None:
    return next((d for d in DRILL_DIMS if d != dim1 and d not in s.filters), None)


# ------------------------------------------------------------------ the brain
class OfflineBrain:
    name = "offline-rules"

    def spec(self, ctx: PlanContext) -> Spec:
        return resolve(ctx.question, ctx.history, ctx.profile)

    def describe(self, ctx: PlanContext) -> str | None:
        s = self.spec(ctx)
        if not s.understood:
            return None
        follow = " (follow-up to the previous question)" if ctx.history and s != parse(ctx.question, ctx.profile) else ""
        return describe(s) + follow

    def next_query(self, ctx: PlanContext) -> Proposal | None:
        s = self.spec(ctx)
        ev = ctx.evidence
        step = len(ev)
        if not s.understood:
            return None
        if step and ev[0].df.empty:
            return None
        label = METRICS[s.metric]["label"]
        if s.mode == "rank":
            if step == 0:
                return Proposal(_rank_sql(s), f"{'Lowest' if s.asc else 'Top'} {_dimname(s.dim)} by {label}{_scope(s)}")
            if step == 1:
                s2 = replace(s, grain="month")
                return Proposal(_trend_sql(s2), f"Monthly {label} trend for context{_scope(s)}")
            return None
        if s.mode == "trend":
            if step == 0:
                return Proposal(_trend_sql(s), f"{cap(label)} by {s.grain}{_scope(s)}")
            if step == 1 and s.dim:
                return Proposal(_trend_sql(s, by=s.dim), f"{cap(label)} by {s.grain} and {_dimname(s.dim)}{_scope(s)}")
            return None
        if s.mode == "total":
            if step == 0:
                return Proposal(_total_sql(s), f"KPI summary{_scope(s)}")
            if step == 1:
                dim = next((d for d in DRILL_DIMS if d not in s.filters), "campaign_name")
                return Proposal(_rank_sql(replace(s, dim=dim, asc=False)), f"{cap(label)} by {dim}{_scope(s)}")
            return None
        if s.mode == "compare":
            y0, y1 = s.years
            if step == 0:
                return Proposal(_compare_sql(s), f"{cap(label)} {y0} vs {y1} by {_dimname(s.dim)}{_scope(s)}")
            if step == 1:
                return Proposal(_compare_total_sql(s), f"Overall {label}: {y0} vs {y1}{_scope(s)}")
            return None
        # diagnose
        if step == 0:
            return Proposal(_trend_sql(s, time=False), f"Monthly {label} history (to locate the anomaly){_scope(Spec(filters=s.filters))}")
        a = pick_anomaly(ev[0].df, "month", alias(s.metric), s.direction, s)
        if a is None:
            return None
        cur, prev = a["month"].strftime("%Y-%m-%d"), a["prev"].strftime("%Y-%m-%d")
        direction = "up" if a["pct"] > 0 else "down"
        dim1 = s.dim or next((d for d in DRILL_DIMS if d not in s.filters), "campaign_name")
        if step == 1:
            return Proposal(_delta_sql(s, dim1, cur, prev, direction),
                            f"{cap(label)} change by {_dimname(dim1)}: {_mon(prev)} to {_mon(cur)}")
        if step == 2 and len(ev[1].df):
            dim2 = _second_dim(s, dim1)
            if dim2 is None:
                return None
            if _concentrated(ev[1].df):
                driver = str(ev[1].df.iloc[0][dim1])
                return Proposal(_delta_sql(s, dim2, cur, prev, direction, {dim1: driver}),
                                f"{cap(label)} change by {_dimname(dim2)} within {driver}: {_mon(prev)} to {_mon(cur)}")
            return Proposal(_delta_sql(s, dim2, cur, prev, direction),
                            f"{cap(label)} change by {_dimname(dim2)} (is it broad-based?): {_mon(prev)} to {_mon(cur)}")
        return None

    # ---------------------------------------------------------------- narration
    def narrate(self, ctx: PlanContext) -> Narrative:
        s = self.spec(ctx)
        ev = ctx.evidence
        if not ev or ev[0].df.empty:
            return Narrative("No matching data", "The query ran but returned no rows for those filters.", [], ["Check the filters and date range."])
        return getattr(self, f"_narrate_{s.mode}")(s, ev)

    # rank -------------------------------------------------------------
    def _narrate_rank(self, s: Spec, ev) -> Narrative:
        m = METRICS[s.metric]
        kind, label, a = m["kind"], m["label"], alias(s.metric)
        df, dim = ev[0].df, s.dim
        F: list[dict] = []
        top, n = df.iloc[0], len(df)
        word = "lowest" if s.asc else "highest"
        line = f"{top[dim]} has the {word} {label} at {fmt(top[a], kind)}"
        if "share_pct" in df:
            line += f", {top['share_pct']:.1f}% of the total"
        F.append({"text": line + ".", "evidence": [ev[0].id]})
        if n >= 2:
            sec = df.iloc[1]
            if top[a] and sec[a]:
                gap = abs(top[a] - sec[a]) / abs(sec[a]) * 100
                if gap < 0.5:
                    F.append({"text": f"{top[dim]} and {sec[dim]} are essentially tied ({fmt(top[a], kind)} vs {fmt(sec[a], kind)}).", "evidence": [ev[0].id]})
                else:
                    F.append({"text": f"{top[dim]} is {gap:.1f}% {'below' if s.asc else 'above'} {sec[dim]} ({fmt(sec[a], kind)}).", "evidence": [ev[0].id]})
        if n >= 3 and "share_pct" in df and not s.asc:
            F.append({"text": f"The top 3 together account for {df['share_pct'].head(3).sum():.1f}% of {label}.", "evidence": [ev[0].id]})
        elif n >= 3:
            first, last = df.iloc[0], df.iloc[-1]
            if first[a] and last[a]:
                ratio = max(first[a], last[a]) / min(first[a], last[a])
                hi, lo = (first, last) if first[a] >= last[a] else (last, first)
                if ratio >= 1.2:
                    F.append({"text": f"{hi[dim]} is {ratio:.1f}x {lo[dim]} ({fmt(lo[a], kind)}), a wide spread.", "evidence": [ev[0].id]})
                else:
                    spread = (ratio - 1) * 100
                    F.append({"text": f"The spread is narrow: {hi[dim]} is only {spread:.1f}% above {lo[dim]} ({fmt(lo[a], kind)}).", "evidence": [ev[0].id]})
        if len(ev) > 1 and len(ev[1].df):
            F.append(self._trend_finding(ev[1], a, kind, label, "month"))
        caveats = [] if m["additive"] else [f"{label} is computed from summed numerators and denominators, not an average of daily ratios."]
        headline = f"{top[dim]} has the {word} {label}{_scope(s)}"
        summary = f"Ranked {_dimname(dim)} by {label}{_scope(s)}; {top[dim]} has the {word} value."
        return Narrative(headline, summary, F, caveats)

    # trend ------------------------------------------------------------
    def _narrate_trend(self, s: Spec, ev) -> Narrative:
        m = METRICS[s.metric]
        kind, label, a = m["kind"], m["label"], alias(s.metric)
        df, g = ev[0].df, s.grain
        F = [self._trend_finding(ev[0], a, kind, label, g)]
        if len(df) > 2:
            ser = df.set_index(g)[a].astype(float)
            ch = ser.pct_change().dropna()
            if len(ch):
                lo, hi = ch.idxmin(), ch.idxmax()
                period = "day" if g == "date" else g
                parts = []
                if ch[lo] < 0:
                    parts.append(f"largest drop in {_lbl(lo, g)} ({ch[lo] * 100:.1f}% vs prior {period})")
                if ch[hi] > 0:
                    parts.append(f"largest rise in {_lbl(hi, g)} (+{ch[hi] * 100:.1f}%)")
                if parts:
                    F.append({"text": cap("; ".join(parts)) + ".", "evidence": [ev[0].id]})
        if len(ev) > 1 and len(ev[1].df):
            d2 = ev[1].df
            last = d2[g].max()
            dim = [c for c in d2.columns if c not in (g, a)][0]
            rows = d2[d2[g] == last].sort_values(a, ascending=False)
            if len(rows):
                F.append({"text": f"In {_lbl(last, g)}, {rows.iloc[0][dim]} was highest at {fmt(rows.iloc[0][a], kind)}"
                                  + (f" and {rows.iloc[-1][dim]} lowest at {fmt(rows.iloc[-1][a], kind)}." if len(rows) > 1 else "."),
                          "evidence": [ev[1].id]})
        caveats = []
        if g in ("week", "date"):
            caveats.append("The first and last buckets can be partial periods; treat their changes with care.")
        return Narrative(f"{cap(label)} by {g}{_scope(s)}", f"{GRAIN_ADJ[g]} {label} over time{_scope(s)}.", F, caveats)

    # total ------------------------------------------------------------
    def _narrate_total(self, s: Spec, ev) -> Narrative:
        m = METRICS[s.metric]
        label, a = m["label"], alias(s.metric)
        r = ev[0].df.iloc[0]
        val = fmt(r[a], m["kind"])
        lead = "Total" if m["additive"] else "Overall"
        F = [{"text": f"{lead} {label}{_scope(s)} was {val}.", "evidence": [ev[0].id]}]
        F.append({"text": f"Spend of {fmt(r['total_spend'], 'money')} generated {fmt(r['total_revenue'], 'money')} in revenue, a ROAS of {fmt(r['roas'], 'ratio')}.",
                  "evidence": [ev[0].id]})
        F.append({"text": f"{fmt(r['total_conversions'], 'count')} conversions at a CAC of {fmt(r['cac'], 'money')}; CTR {fmt(r['ctr'], 'pct')}, conversion rate {fmt(r['cvr'], 'pct')}.",
                  "evidence": [ev[0].id]})
        if len(ev) > 1 and len(ev[1].df):
            d = ev[1].df
            dim = d.columns[0]
            t = d.iloc[0]
            extra = f" ({t['share_pct']:.1f}% of the total)" if "share_pct" in d else ""
            F.append({"text": f"By {dim}, {t[dim]} leads with {fmt(t[a], m['kind'])}{extra}.", "evidence": [ev[1].id]})
        return Narrative(f"{lead} {label}{_scope(s)}: {val}", f"Headline KPIs{_scope(s)}, with a breakdown of {label}.", F, [])

    # compare ----------------------------------------------------------
    def _narrate_compare(self, s: Spec, ev) -> Narrative:
        m = METRICS[s.metric]
        kind, label, a = m["kind"], m["label"], alias(s.metric)
        y0, y1 = s.years
        d = ev[0].df.dropna(subset=["change"])
        dim = s.dim
        F: list[dict] = []
        headline = f"{cap(label)}: {y0} vs {y1}{_scope(s)}"
        if len(ev) > 1 and len(ev[1].df) == 2:
            t = ev[1].df.set_index("year")[a].astype(float)
            if y0 in t.index and y1 in t.index and t[y0]:
                pct = (t[y1] / t[y0] - 1) * 100
                verb = "grew" if pct >= 0 else "fell"
                headline = f"{cap(label)} {verb} {abs(pct):.1f}% from {y0} to {y1}{_scope(s)}"
                F.append({"text": f"Overall {label} went from {fmt(t[y0], kind)} in {y0} to {fmt(t[y1], kind)} in {y1} ({pct:+.1f}%).", "evidence": [ev[1].id]})
        if len(d):
            best, worst = d.iloc[0], d.iloc[-1]
            F.append({"text": f"{best[dim]} rose most: {fmt(best[f'value_{y0}'], kind)} to {fmt(best[f'value_{y1}'], kind)} ({best['change_pct']:+.1f}%).", "evidence": [ev[0].id]})
            if len(d) > 1:
                verb = "fell most" if worst["change"] < 0 else "grew least"
                F.append({"text": f"{worst[dim]} {verb}: {fmt(worst[f'value_{y0}'], kind)} to {fmt(worst[f'value_{y1}'], kind)} ({worst['change_pct']:+.1f}%).", "evidence": [ev[0].id]})
            ups = int((d["change"] > 0).sum())
            F.append({"text": f"{ups} of {len(d)} {_dimname(dim)}s increased {label}.", "evidence": [ev[0].id]})
        caveats = [] if m["additive"] else [f"{label} is computed from summed numerators and denominators."]
        if m.get("bad_when_up"):
            caveats.append(f"For {label}, an increase is a deterioration.")
        return Narrative(headline, f"Compared {label} between {y0} and {y1} by {_dimname(dim)}{_scope(s)}.", F, caveats)

    # diagnose ---------------------------------------------------------
    def _narrate_diagnose(self, s: Spec, ev) -> Narrative:
        m = METRICS[s.metric]
        kind, label, a = m["kind"], m["label"], alias(s.metric)
        F: list[dict] = []
        caveats: list[str] = []
        a1 = pick_anomaly(ev[0].df, "month", a, s.direction, s)
        if a1 is None:
            return Narrative("No clear anomaly", "Not enough history or no month matched the requested window.", [], [])
        word = "rose" if a1["pct"] > 0 else "fell"
        headline = f"{cap(label)} {word} {abs(a1['pct']) * 100:.1f}% in {_mon(a1['month'])}{_scope(replace(s, month=None, year=None, quarter=None))}"
        highlights = {ev[0].id: a1["month"].strftime("%Y-%m-%d")}
        F.append({"text": f"{cap(label)} {word} from {fmt(a1['prev_val'], kind)} in {_mon(a1['prev'])} to {fmt(a1['cur_val'], kind)} in {_mon(a1['month'])} ({a1['pct'] * 100:+.1f}% month over month).", "evidence": [ev[0].id]})
        if a1["yoy"] is not None:
            same = f"{a1['yoy'] * 100:+.1f}%"
            adj = f"{abs(a1['adj']) * 100:.1f}"
            if abs(a1["adj"]) < 0.05:
                verdict = "so this is mostly seasonal."
            elif (a1["adj"] < 0) == (a1["pct"] < 0):
                verdict = f"so the move is {adj} percentage points {'weaker' if a1['pct'] < 0 else 'stronger'} than the usual seasonal pattern."
            else:
                verdict = f"so, net of seasonality, the underlying move is {adj} points in the opposite direction."
            F.append({"text": f"The same month-over-month move a year earlier was {same}, {verdict}", "evidence": [ev[0].id]})
        else:
            caveats.append("Less than 13 months of history: could not adjust for seasonality.")
        broad_dims: list[str] = []
        if len(ev) > 1 and len(ev[1].df):
            F.append(self._attribution_finding(ev[1], m, broad_dims))
        if len(ev) > 2 and len(ev[2].df):
            d1 = ev[1].df
            within = str(d1.iloc[0][_segcol(d1)]) if _concentrated(d1) else None
            F.append(self._attribution_finding(ev[2], m, broad_dims, within))
        summary = f"Located the largest {'increase' if a1['pct'] > 0 else 'decline'} in {label} and traced it to its main drivers."
        if abs(a1["pct"]) < 0.02:
            caveats.append("The flagged move is small (under 2%), so the segment attribution below is mostly noise.")
        seasonal = a1["adj"] is not None and abs(a1["adj"]) < 0.05
        if seasonal:
            summary = (f"The largest {'increase' if a1['pct'] > 0 else 'decline'} in {label} ({_mon(a1['month'])}) matches the move "
                       f"seen a year earlier, so it is mostly seasonal rather than an incident.")
        elif broad_dims and len(broad_dims) == len(ev) - 1:
            summary = (f"Located the largest {'increase' if a1['pct'] > 0 else 'decline'} in {label}; it is broad-based across "
                       f"{' and '.join(broad_dims)}, which suggests a cause that affects the whole scope at once.")
            caveats.append("Hypothesis only: broad-based moves usually come from tracking, pricing, deliverability or platform-wide changes. The data cannot confirm which.")
        caveats.append("This shows where the change happened, not why. Check budget, creative, tracking and pricing changes for the flagged segment before acting.")
        return Narrative(headline, summary, F, caveats, highlights)

    @staticmethod
    def _attribution_finding(e, m: dict, broad_dims: list, within: str | None = None) -> dict:
        d = e.df
        dim = _segcol(d)
        kind, label = m["kind"], m["label"]
        r = d.iloc[0]
        prefix = f"Within {within}, " if within else ""
        if _broad(d):
            broad_dims.append(_dimname(dim) + "s")
            pcts = d["change_pct"].dropna().astype(float)
            verb = "fell" if pcts.iloc[0] < 0 else "rose"
            lo, hi = pcts.abs().min(), pcts.abs().max()
            text = (f"{prefix}the change is broad-based: every {_dimname(dim)} {verb}, by {lo:.1f}% to {hi:.1f}%, "
                    f"so {r[dim]}'s {r['share_of_change_pct']:.1f}% share of the change mostly reflects its size.")
            return {"text": cap(text), "evidence": [e.id]}
        if _concentrated(d) or len(d) == 1:
            k = _drivers(d)
            if k >= 2:
                names = " and ".join(str(x) for x in d[dim].head(k))
                cum = d["share_of_change_pct"].head(k).sum()
                text = (f"{prefix}the change is concentrated in {names} (together {cum:.1f}% of it); "
                        f"in {r[dim]}, {label} went from {fmt(r['previous'], kind)} to {fmt(r['current'], kind)}.")
            elif m["additive"]:
                text = f"{prefix}{r[dim]} drove the biggest move: {fmt(r['change'], kind)} ({r['share_of_change_pct']:.1f}% of the net change)."
            elif r["share_of_change_pct"] > 100:
                text = (f"{prefix}{r[dim]} accounts for more than the whole net change ({r['share_of_change_pct']:.1f}%; other "
                        f"{_dimname(dim)}s partly offset it): {label} went from {fmt(r['previous'], kind)} to {fmt(r['current'], kind)}.")
            else:
                text = (f"{prefix}{r[dim]} explains {r['share_of_change_pct']:.1f}% of the change: "
                        f"{label} went from {fmt(r['previous'], kind)} to {fmt(r['current'], kind)}.")
            return {"text": cap(text), "evidence": [e.id]}
        text = (f"{prefix}no single {_dimname(dim)} dominates; {r[dim]} contributes most "
                f"({r['share_of_change_pct']:.1f}% of the change, {label} {fmt(r['previous'], kind)} to {fmt(r['current'], kind)}).")
        return {"text": cap(text), "evidence": [e.id]}

    @staticmethod
    def _trend_finding(ev, a, kind, label, grain: str) -> dict:
        df = ev.df
        tcol = df.columns[0]
        ser = df.set_index(tcol)[a].astype(float)
        first, last = ser.iloc[0], ser.iloc[-1]
        pk = ser.idxmax()
        chg = (last / first - 1) * 100 if first else 0
        return {"text": f"{GRAIN_ADJ.get(grain, 'Monthly')} {label} went from {fmt(first, kind)} in {_lbl(ser.index[0], grain)} to "
                        f"{fmt(last, kind)} in {_lbl(ser.index[-1], grain)} ({chg:+.1f}%), peaking at {fmt(ser[pk], kind)} in {_lbl(pk, grain)}.",
                "evidence": [ev.id]}
