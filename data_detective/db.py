"""Read-only DuckDB warehouse: schema inspection, profiling and guarded execution."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import duckdb
import pandas as pd

from .config import DB_PATH, QUERY_TIMEOUT_S

TABLE = "marketing_performance"

COLUMN_DOCS = {
    "date": "calendar day",
    "week": "first day of the week (DATE) - use for weekly trends",
    "month": "first day of the month (DATE) - use for monthly trends",
    "quarter": "first day of the quarter (DATE), e.g. DATE '2025-07-01' = Q3 2025",
    "year": "calendar year (INTEGER)",
    "channel": "marketing channel",
    "campaign_name": "campaign name",
    "objective": "campaign objective",
    "region": "sales region",
    "device": "device type",
    "impressions": "ad impressions / email sends (additive)",
    "clicks": "clicks (additive)",
    "spend": "marketing cost in USD (additive)",
    "conversions": "orders (additive)",
    "revenue": "revenue in USD (additive)",
}
CATEGORICAL = ["channel", "region", "device", "objective"]

METRIC_DOCS = """Derived metrics - always compute from SUMs, never AVG of per-row ratios:
  ROAS = SUM(revenue) / NULLIF(SUM(spend), 0)
  CTR  = SUM(clicks) / NULLIF(SUM(impressions), 0)
  CVR  = SUM(conversions) / NULLIF(SUM(clicks), 0)
  CPC  = SUM(spend) / NULLIF(SUM(clicks), 0)
  CAC  = SUM(spend) / NULLIF(SUM(conversions), 0)"""


class Warehouse:
    def __init__(self, path: Path = DB_PATH):
        if not Path(path).exists():
            from . import datagen

            datagen.build(Path(path))
        self.path = Path(path)
        self.con = duckdb.connect(str(self.path), read_only=True)
        # Defence in depth: no file/network access from SQL, config frozen.
        self.con.execute("SET enable_external_access=false")
        self.con.execute("SET lock_configuration=true")
        self._profile: dict | None = None

    # ---------- inspection ----------
    def schema(self) -> list[dict]:
        df = self.con.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = ? ORDER BY ordinal_position", [TABLE]
        ).df()
        return [
            {"name": r.column_name, "type": r.data_type, "description": COLUMN_DOCS.get(r.column_name, "")}
            for r in df.itertuples()
        ]

    def profile(self) -> dict:
        if self._profile is None:
            n, d0, d1 = self.con.execute(f"SELECT COUNT(*), MIN(date), MAX(date) FROM {TABLE}").fetchone()
            cats = {
                c: [r[0] for r in self.con.execute(f"SELECT DISTINCT {c} FROM {TABLE} ORDER BY 1").fetchall()]
                for c in CATEGORICAL
            }
            nulls = self.con.execute(
                "SELECT " + ", ".join(f"COUNT(*) - COUNT({c['name']}) AS {c['name']}" for c in self.schema())
                + f" FROM {TABLE}"
            ).df().iloc[0].to_dict()
            self._profile = {
                "table": TABLE, "rows": int(n), "date_min": str(d0), "date_max": str(d1),
                "categories": cats, "null_counts": {k: int(v) for k, v in nulls.items()},
            }
        return self._profile

    def schema_prompt(self) -> str:
        p = self.profile()
        cols = "\n".join(f"  {c['name']} {c['type']} - {c['description']}" for c in self.schema())
        cats = "\n".join(f"  {k}: {', '.join(v)}" for k, v in p["categories"].items())
        return (
            f"DuckDB table `{TABLE}` ({p['rows']:,} rows, {p['date_min']} to {p['date_max']}, no NULLs).\n"
            f"Columns:\n{cols}\nCategorical values:\n{cats}\n{METRIC_DOCS}"
        )

    # ---------- execution ----------
    def explain(self, sql: str) -> None:
        """Raises duckdb.Error when SQL does not bind (unknown column, type error...)."""
        self.con.cursor().execute("EXPLAIN " + sql)

    def run(self, sql: str, timeout: float = QUERY_TIMEOUT_S) -> tuple[pd.DataFrame, float]:
        cur = self.con.cursor()
        timer = threading.Timer(timeout, cur.interrupt)
        timer.start()
        t0 = time.perf_counter()
        try:
            df = cur.execute(sql).df()
        finally:
            timer.cancel()
        return df, (time.perf_counter() - t0) * 1000
