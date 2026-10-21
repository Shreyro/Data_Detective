"""Synthetic marketing data generator (100% fake, safe to publish).

Writes data/marketing.duckdb with two tables and one analyst-facing view:
  campaigns            - one row per campaign
  daily_performance    - campaign x region x device x day
  marketing_performance (view) - the denormalised view the agent is allowed to query

Two anomalies are planted so the agent has something to detect:
  1. Paid Social in APAC + LATAM, Jul-Sep 2025: CPM doubles and conversion rate drops ~45%
  2. Email, Mar 2024: click-through rate collapses (deliverability problem)

Run:  python -m data_detective.datagen
"""
from __future__ import annotations

import sys

import duckdb
import numpy as np
import pandas as pd

from .config import DB_PATH

SEED = 42
START, END = "2023-01-01", "2025-12-31"

# channel: (cpm $, ctr, cvr, avg order value $, daily impressions scale)
CHANNELS = {
    "Paid Search": (40, 0.035, 0.045, 85, 9000),
    "Paid Social": (9, 0.012, 0.018, 70, 60000),
    "Email": (25, 0.060, 0.070, 60, 12000),
    "Display": (4, 0.006, 0.012, 65, 90000),
    "Affiliate": (20, 0.020, 0.050, 75, 10000),
    "Influencer": (15, 0.015, 0.030, 90, 14000),
}
REGIONS = {"North America": 1.0, "Europe": 0.8, "APAC": 0.7, "LATAM": 0.4, "Middle East": 0.25}
DEVICES = {"Mobile": 0.55, "Desktop": 0.35, "Tablet": 0.10}
OBJECTIVES = ["Acquisition", "Retention", "Brand Awareness", "Seasonal Promo"]


def build_campaigns(rng: np.random.Generator) -> pd.DataFrame:
    rows, cid = [], 1
    for channel in CHANNELS:
        for k in range(4):
            start, end = pd.Timestamp(START), pd.Timestamp(END)  # always-on campaigns
            rows.append(
                dict(
                    campaign_id=cid,
                    campaign_name=f"{channel} - {OBJECTIVES[k]} {k + 1}",
                    channel=channel,
                    objective=OBJECTIVES[k],
                    start_date=start.date(),
                    end_date=end.date(),
                    scale=float(rng.lognormal(0, 0.35)),
                )
            )
            cid += 1
    return pd.DataFrame(rows)


def build_daily(campaigns: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    days = pd.date_range(START, END, freq="D")
    frames = []
    for c in campaigns.itertuples():
        cpm, ctr, cvr, aov, scale = CHANNELS[c.channel]
        d = days[(days >= pd.Timestamp(c.start_date)) & (days <= pd.Timestamp(c.end_date))]
        for region, rw in REGIONS.items():
            for device, dw in DEVICES.items():
                n = len(d)
                growth = 1.0 + 0.3 * (d - pd.Timestamp(START)).days.values / 1095
                weekend = np.where(d.dayofweek >= 5, 1.12, 1.0)
                season = np.select([d.month == 11, d.month == 12, d.month.isin([1, 2])], [1.45, 1.35, 0.85], 1.0)
                lam = scale * c.scale * rw * dw * growth * weekend * season
                imps = rng.poisson(lam).astype(np.int64)

                cpm_d = np.full(n, float(cpm))
                ctr_d = np.full(n, float(ctr))
                cvr_d = np.full(n, float(cvr))

                # Planted anomaly 1: Paid Social creative fatigue in APAC + LATAM, Q3 2025
                if c.channel == "Paid Social" and region in ("APAC", "LATAM"):
                    m = (d >= "2025-07-01") & (d <= "2025-09-30")
                    cpm_d[m] *= 2.0
                    cvr_d[m] *= 0.55
                # Planted anomaly 2: Email deliverability problem, Mar 2024
                if c.channel == "Email":
                    m = (d >= "2024-03-01") & (d <= "2024-03-31")
                    ctr_d[m] *= 0.4

                clicks = rng.binomial(imps, np.clip(ctr_d * rng.normal(1, 0.05, n), 0, 1))
                conv = rng.binomial(clicks, np.clip(cvr_d * rng.normal(1, 0.06, n), 0, 1))
                revenue = conv * aov * rng.lognormal(0, 0.08, n)
                spend = imps * cpm_d / 1000 * rng.normal(1, 0.04, n)
                frames.append(
                    pd.DataFrame(
                        dict(
                            date=d.date, campaign_id=c.campaign_id, region=region, device=device,
                            impressions=imps, clicks=clicks, spend=spend.round(2),
                            conversions=conv, revenue=revenue.round(2),
                        )
                    )
                )
    return pd.concat(frames, ignore_index=True)


def build(path=DB_PATH) -> None:
    rng = np.random.default_rng(SEED)
    campaigns = build_campaigns(rng)
    daily = build_daily(campaigns, rng)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = duckdb.connect(str(path))
    con.register("c_df", campaigns.drop(columns=["scale"]))
    con.register("d_df", daily)
    con.execute("CREATE TABLE campaigns AS SELECT * FROM c_df")
    con.execute("CREATE TABLE daily_performance AS SELECT * FROM d_df")
    con.execute(
        """
        CREATE VIEW marketing_performance AS
        SELECT
            CAST(p.date AS DATE)                         AS date,
            CAST(DATE_TRUNC('week', p.date) AS DATE)     AS week,
            CAST(DATE_TRUNC('month', p.date) AS DATE)    AS month,
            CAST(DATE_TRUNC('quarter', p.date) AS DATE)  AS quarter,
            CAST(EXTRACT(year FROM p.date) AS INTEGER)   AS year,
            c.channel, c.campaign_name, c.objective,
            p.region, p.device,
            p.impressions, p.clicks, p.spend, p.conversions, p.revenue
        FROM daily_performance p
        JOIN campaigns c USING (campaign_id)
        """
    )
    n = con.execute("SELECT COUNT(*) FROM marketing_performance").fetchone()[0]
    con.close()
    print(f"Built {path} with {n:,} rows.")


if __name__ == "__main__":
    build()
    sys.exit(0)
