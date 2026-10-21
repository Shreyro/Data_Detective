"""Offline 'brain': a deterministic, rule-based planner + narrator.

It lets the whole project run (and be tested in CI) with no API key. It turns a natural-language
question into a QuerySpec, generates SQL from the semantic layer, and runs a multi-step
investigation for "why did X drop?" questions: find the anomalous month (seasonally adjusted),
break it down by channel, then drill into the biggest driver.
The Claude brain (claude.py) plugs into the same interface.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from .metrics import METRICS, MATCH_ORDER, alias, expr, fmt
from .models import Narrative, PlanContext, Proposal

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
DIM_WORDS = {
    "channel": "channel", "channels": "channel",
    "region": "region", "regions": "region", "country": "region", "market": "region", "markets": "region", "geo": "region",
    "device": "device", "devices": "device",
    "campaign": "campaign_name", "campaigns": "campaign_name",
    "objective": "objective", "objectives": "objective",
}
DIAG = re.compile(r"\b(why|drop|dropped|decline|declined|fell|fall|fallen|decrease|decreased|spike|spiked|anomal\w*|unusual|weird|investigate|explain|diagnose|what happened|went wrong)\b")
UP = re.compile(r"\b(spike|spiked|increase|increased|jump|jumped|surge|surged|rose|rise|grew|up)\b")
DOWN = re.compile(r"\b(drop|dropped|decline|declined|fell|fall|fallen|decrease|decreased|down|worse|dip|dipped)\b")
GRAINS = [
    (r"\b(weekly|by week|per week|week over week)\b", "week"),
    (r"\b(quarterly|by quarter|per quarter)\b", "quarter"),
    (r"\b(yearly|by year|per year|annual\w*|year over year|yoy)\b", "year"),
    (r"\b(daily|by day|per day)\b", "date"),
    (r"\b(trend|trends|over time|monthly|by month|per month|month over month|mom|growth|evolution|history|timeline)\b", "month"),
]


@dataclass
class Spec:
    metric: str = "revenue"
    mode: str = "rank"                      # rank | trend | diagnose
    dim: str | None = None
    grain: str | None = None
    filters: dict = field(default_factory=dict)   # column -> value
    year: int | None = None
    quarter: int | None = None
    month: str | None = None                # 'YYYY-MM-01'
    top: int = 10
    asc: bool = False
    direction: str = "down"                 # for diagnose
    understood: bool = False


def parse(question: str, profile: dict) -> Spec:
    q = question.lower()
    s = Spec()
    for m in MATCH_ORDER:
        if any(re.search(rf"\b{re.escape(a)}", q) for a in METRICS[m]["aliases"]):
            s.metric, s.understood = m, True
            break
    # dimension: prefer the word after by/per/across/each
    m = re.search(r"\b(?:by|per|across|each|split by|broken down by)\s+(\w+)", q)
    if m and m.group(1) in DIM_WORDS:
        s.dim = DIM_WORDS[m.group(1)]
    else:
        for w, col in DIM_WORDS.items():
            if re.search(rf"\b{w}\b", q):
                s.dim = col
                break
    # filters from known category values
    for col, values in profile["categories"].items():
        for v in values:
            if re.search(rf"\b{re.escape(v.lower())}\b", q):
                s.filters[col] = v
    if s.dim in s.filters and re.search(r"\b(by|per|across|each)\b", q):
        s.filters.pop(s.dim)
    y = re.search(r"\b(20\d\d)\b", q)
    if y:
        s.year = int(y.group(1))
    qq = re.search(r"\bq([1-4])\b", q)
    if qq:
        s.quarter = int(qq.group(1))
    mm = re.search(r"\b(" + "|".join(MONTHS) + r")[a-z]*\.?\s+(20\d\d)\b", q)
    if mm:
        s.month = f"{mm.group(2)}-{MONTHS.index(mm.group(1)) + 1:02d}-01"
        s.year = s.quarter = None
    t = re.search(r"\b(?:top|bottom|worst|best|lowest|highest)\s+(\d+)\b", q)
    if t:
        s.top = max(1, min(int(t.group(1)), 50))
    s.asc = bool(re.search(r"\b(bottom|worst|lowest|least|weakest|smallest)\b", q))
    if METRICS[s.metric].get("bad_when_up"):          # for CAC/CPC, "worst" = highest, "best" = lowest
        if re.search(r"\b(worst|highest|most expensive)\b", q):
            s.asc = False
        elif re.search(r"\b(best|cheapest|efficient)\b", q):
            s.asc = True
    for rx, g in GRAINS:
        if re.search(rx, q):
            s.grain = g
            break
    s.understood = s.understood or bool(s.dim or s.filters or s.grain or DIAG.search(q))
    if DIAG.search(q):
        s.mode, s.grain = "diagnose", "month"
    elif s.grain:
        s.mode = "trend"
    else:
        s.mode, s.dim = "rank", s.dim or "channel"
    bad_up = METRICS[s.metric].get("bad_when_up", False)
    s.direction = "up" if UP.search(q) and not DOWN.search(q) else "down" if DOWN.search(q) else ("up" if bad_up else "down")
    return s


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


def _delta_sql(s: Spec, dim: str, cur: str, prev: str, extra: dict | None = None) -> str:
    c, p = f"month = DATE '{cur}'", f"month = DATE '{prev}'"
    filt = dict(s.filters)
    filt.update(extra or {})
    where = " AND ".join([f"month IN (DATE '{prev}', DATE '{cur}')"] + [f"{k} = {_lit(v)}" for k, v in filt.items() if k != dim])
    return (
        f"SELECT {dim},\n  {expr(s.metric, p)} AS previous,\n  {expr(s.metric, c)} AS current,\n"
        f"  {expr(s.metric, c)} - {expr(s.metric, p)} AS change\n"
        f"FROM marketing_performance\nWHERE {where}\nGROUP BY {dim}\n"
        f"ORDER BY change {'DESC' if s.direction == 'up' else 'ASC'}\nLIMIT 10"
    )


# ------------------------------------------------------------------ analysis helpers
def cap(label: str) -> str:
    return label if label.isupper() else label[:1].upper() + label[1:]


def _mon(x) -> str:
    return pd.Timestamp(x).strftime("%b %Y")


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
    m = cand.idxmax() if direction == "up" else cand.idxmin()
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


# ------------------------------------------------------------------ the brain
class OfflineBrain:
    name = "offline-rules"

    def next_query(self, ctx: PlanContext) -> Proposal | None:
        s = parse(ctx.question, ctx.profile)
        ev = ctx.evidence
        step = len(ev)
        if not s.understood:
            return None
        if step and ev[0].df.empty:
            return None
        label = METRICS[s.metric]["label"]
        if s.mode == "rank":
            if step == 0:
                return Proposal(_rank_sql(s), f"{'Lowest' if s.asc else 'Top'} {s.dim.replace('_name','')} by {label}{_scope(s)}")
            if step == 1:
                s2 = Spec(**{**s.__dict__, "grain": "month"})
                return Proposal(_trend_sql(s2), f"Monthly {label} trend for context{_scope(s)}")
            return None
        if s.mode == "trend":
            if step == 0:
                return Proposal(_trend_sql(s), f"{label} by {s.grain}{_scope(s)}")
            if step == 1 and s.dim:
                return Proposal(_trend_sql(s, by=s.dim), f"{label} by {s.grain} and {s.dim.replace('_name','')}{_scope(s)}")
            return None
        # diagnose
        if step == 0:
            return Proposal(_trend_sql(s, time=False), f"Monthly {label} history (to locate the anomaly){_scope(Spec(filters=s.filters))}")
        a = pick_anomaly(ev[0].df, "month", alias(s.metric), s.direction, s)
        if a is None:
            return None
        cur, prev = a["month"].strftime("%Y-%m-%d"), a["prev"].strftime("%Y-%m-%d")
        dim1 = s.dim or ("region" if "channel" in s.filters else "channel")
        if step == 1:
            return Proposal(_delta_sql(s, dim1, cur, prev), f"{label} change by {dim1.replace('_name','')}: {_mon(prev)} to {_mon(cur)}")
        if step == 2 and len(ev[1].df):
            dim2 = "region" if dim1 != "region" else ("device" if "channel" in s.filters else "channel")
            driver = str(ev[1].df.iloc[0][dim1])
            return Proposal(_delta_sql(s, dim2, cur, prev, {dim1: driver}),
                            f"{label} change by {dim2.replace('_name','')} within {driver}: {_mon(prev)} to {_mon(cur)}")
        return None

    # ---------------------------------------------------------------- narration
    def narrate(self, ctx: PlanContext) -> Narrative:
        s = parse(ctx.question, ctx.profile)
        ev = ctx.evidence
        m = METRICS[s.metric]
        kind, label, a = m["kind"], m["label"], alias(s.metric)
        if not ev or ev[0].df.empty:
            return Narrative("No matching data", "The query ran but returned no rows for those filters.", [], ["Check the filters and date range."])
        F: list[dict] = []
        caveats: list[str] = []
        highlights: dict = {}

        if s.mode == "rank":
            df = ev[0].df
            dim = s.dim
            top, n = df.iloc[0], len(df)
            word = "lowest" if s.asc else "highest"
            headline = f"{top[dim]} has the {word} {label}{_scope(s)}"
            line = f"{top[dim]} has the {word} {label} at {fmt(top[a], kind)}"
            if "share_pct" in df:
                line += f", {top['share_pct']:.1f}% of the total"
            F.append({"text": line + ".", "evidence": [ev[0].id]})
            if n >= 2:
                sec = df.iloc[1]
                if top[a] and sec[a]:
                    gap = abs(top[a] - sec[a]) / abs(sec[a]) * 100
                    F.append({"text": f"{top[dim]} is {gap:.1f}% {'below' if s.asc else 'above'} {sec[dim]} ({fmt(sec[a], kind)}).", "evidence": [ev[0].id]})
            if n >= 3 and "share_pct" in df:
                F.append({"text": f"The top 3 together account for {df['share_pct'].head(3).sum():.1f}% of {label}.", "evidence": [ev[0].id]})
            elif n >= 3 and not m["additive"]:
                lo = df.iloc[df[a].astype(float).idxmin()]
                hi = df.iloc[df[a].astype(float).idxmax()]
                if lo[a]:
                    F.append({"text": f"{hi[dim]} is {hi[a] / lo[a]:.1f}x higher than {lo[dim]} ({fmt(lo[a], kind)}).", "evidence": [ev[0].id]})
            if len(ev) > 1:
                F.append(self._trend_finding(ev[1], a, kind, label))
            if not m["additive"]:
                caveats.append(f"{label} is computed from summed numerators and denominators, not an average of daily ratios.")
            summary = f"Ranked {dim.replace('_name','')} by {label}{_scope(s)}; {top[dim]} has the {word} value."

        elif s.mode == "trend":
            df = ev[0].df
            tcol = s.grain
            headline = f"{cap(label)} by {tcol}{_scope(s)}"
            F.append(self._trend_finding(ev[0], a, kind, label))
            if len(df) > 2:
                ser = df.set_index(tcol)[a].astype(float)
                ch = ser.pct_change().dropna()
                if len(ch):
                    lo, hi = ch.idxmin(), ch.idxmax()
                    F.append({"text": f"Largest drop: {_mon(lo)} ({ch[lo] * 100:.1f}% vs prior {tcol}); largest rise: {_mon(hi)} (+{ch[hi] * 100:.1f}%).", "evidence": [ev[0].id]})
            if len(ev) > 1:
                d2 = ev[1].df
                last = d2[tcol].max()
                dim = [c for c in d2.columns if c not in (tcol, a)][0]
                rows = d2[d2[tcol] == last].sort_values(a, ascending=False)
                if len(rows):
                    F.append({"text": f"In {_mon(last)}, {rows.iloc[0][dim]} was highest at {fmt(rows.iloc[0][a], kind)}.", "evidence": [ev[1].id]})
            summary = f"{cap(label)} over time{_scope(s)}."

        else:  # diagnose
            a1 = pick_anomaly(ev[0].df, "month", a, s.direction, s)
            if a1 is None:
                return Narrative("No clear anomaly", "Not enough history or no month matched the requested window.", [], [])
            word = "rose" if a1["pct"] > 0 else "fell"
            headline = f"{cap(label)} {word} {abs(a1['pct']) * 100:.1f}% in {_mon(a1['month'])}"
            highlights[ev[0].id] = a1["month"].strftime("%Y-%m-%d")
            F.append({"text": f"{cap(label)} {word} from {fmt(a1['prev_val'], kind)} in {_mon(a1['prev'])} to {fmt(a1['cur_val'], kind)} in {_mon(a1['month'])} ({a1['pct'] * 100:+.1f}% month over month).", "evidence": [ev[0].id]})
            if a1["yoy"] is not None:
                same = f"{a1['yoy'] * 100:+.1f}%"
                adj = f"{abs(a1['adj']) * 100:.1f}"
                verdict = "so this is mostly seasonal." if abs(a1["adj"]) < 0.05 else f"so the change is {adj} percentage points {'worse' if (a1['adj'] < 0) == (a1['pct'] < 0) else 'better'} than normal seasonality."
                F.append({"text": f"The same month-over-month move a year earlier was {same}, {verdict}", "evidence": [ev[0].id]})
            else:
                caveats.append("Less than 13 months of history: could not adjust for seasonality.")
            if len(ev) > 1 and len(ev[1].df):
                d = ev[1].df
                dim1 = [c for c in d.columns if c not in ("previous", "current", "change")][0]
                r = d.iloc[0]
                if m["additive"]:
                    tot = d["change"].sum()
                    share = r["change"] / tot * 100 if tot else 0
                    F.append({"text": f"{r[dim1]} drove the biggest move: {fmt(r['change'], kind)} of the {fmt(tot, kind)} net change ({share:.0f}%).", "evidence": [ev[1].id]})
                else:
                    F.append({"text": f"{r[dim1]} moved most: {label} went from {fmt(r['previous'], kind)} to {fmt(r['current'], kind)}.", "evidence": [ev[1].id]})
            if len(ev) > 2 and len(ev[2].df):
                d = ev[2].df
                dim2 = [c for c in d.columns if c not in ("previous", "current", "change")][0]
                r = d.iloc[0]
                driver = ev[1].df.iloc[0][[c for c in ev[1].df.columns if c not in ("previous", "current", "change")][0]]
                F.append({"text": f"Within {driver}, {r[dim2]} changed most: {label} went from {fmt(r['previous'], kind)} to {fmt(r['current'], kind)}.", "evidence": [ev[2].id]})
            caveats.append("This shows where the change happened, not why. Check budget, creative, tracking and pricing changes for the flagged segment before acting.")
            summary = f"Located the largest {'increase' if s.direction == 'up' else 'decline'} in {label} and traced it to its main drivers."
        return Narrative(headline, summary, F, caveats, highlights)

    @staticmethod
    def _trend_finding(ev, a, kind, label) -> dict:
        df = ev.df
        tcol = df.columns[0]
        ser = df.set_index(tcol)[a].astype(float)
        first, last = ser.iloc[0], ser.iloc[-1]
        pk = ser.idxmax()
        chg = (last / first - 1) * 100 if first else 0
        return {"text": f"Monthly {label} went from {fmt(first, kind)} in {_mon(ser.index[0])} to {fmt(last, kind)} in {_mon(ser.index[-1])} ({chg:+.1f}%), peaking at {fmt(ser[pk], kind)} in {_mon(pk)}.", "evidence": [ev.id]}
