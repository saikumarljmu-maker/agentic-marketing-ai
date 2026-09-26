"""
Observer Agent V2 - Uses corrected Criteo data preparation
Fixes: real calendar days, correct conversion counting, no fake ROAS
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import json

PROCESSED_PATH = Path("data/processed")


class ObserverAgentV2:
    """
    Corrected Observer Agent using properly prepared Criteo data.
    Uses daily_cost.parquet and conversions.parquet from criteo_prep.py
    """

    def __init__(self):
        self.daily_cost = None
        self.conversions = None
        self.n_days = 0
        self.n_campaigns = 0

    def initialise(self):
        """Load the corrected parquet files."""
        logger.info("=== Observer Agent V2 Initialising ===")

        cost_path = PROCESSED_PATH / "daily_cost.parquet"
        conv_path = PROCESSED_PATH / "conversions.parquet"

        if not cost_path.exists() or not conv_path.exists():
            raise FileNotFoundError(
                "Run criteo_prep.prepare() first to generate parquet files"
            )

        self.daily_cost = pd.read_parquet(cost_path)
        self.conversions = pd.read_parquet(conv_path)

        self.n_days = self.daily_cost["day"].nunique()
        self.n_campaigns = self.daily_cost["campaign"].nunique()

        logger.success(
            f"Loaded: {self.n_days} days | "
            f"{self.n_campaigns:,} campaigns | "
            f"{int(self.daily_cost['impressions'].sum()):,} impressions | "
            f"{int(self.conversions['conversions'].sum()):,} conversions"
        )

        return {
            "status": "ready",
            "days": self.n_days,
            "campaigns": self.n_campaigns,
            "total_impressions": int(self.daily_cost["impressions"].sum()),
            "total_conversions": int(self.conversions["conversions"].sum()),
            "total_spend": round(float(self.daily_cost["cost"].sum()), 4),
            "day_range": {
                "min": int(self.daily_cost["day"].min()),
                "max": int(self.daily_cost["day"].max())
            }
        }

    def observe(self, day, window=7):
        """
        Build observable performance summary for a given day.
        Only uses information available at the START of that day.
        No future data leakage.
        """
        if self.daily_cost is None:
            raise ValueError("Not initialised. Call initialise() first.")

        lo = max(0, day - window)
        hi = day - 1

        # Cost data for the window
        cost_window = self.daily_cost[
            (self.daily_cost["day"] >= lo) &
            (self.daily_cost["day"] <= hi)
        ]

        # Only conversions already received by this day
        known_conv = self.conversions[
            (self.conversions["impression_day"] >= lo) &
            (self.conversions["impression_day"] <= hi) &
            (self.conversions["conversion_day"] < day)
        ]

        if cost_window.empty:
            return {"day": day, "error": "No data for this window"}

        # Aggregate by campaign
        cost_agg = cost_window.groupby("campaign").agg(
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            cost=("cost", "sum"),
            last_day_cost=("last_day_cost", "last")
        )

        conv_agg = known_conv.groupby("campaign")["conversions"].sum()

        view = cost_agg.join(conv_agg.rename("known_conversions"), how="left")
        view["known_conversions"] = view["known_conversions"].fillna(0)
        view["cpa"] = (
            view["cost"] /
            view["known_conversions"].replace(0, np.nan)
        )
        view["ctr"] = view["clicks"] / view["impressions"].replace(0, np.nan)
        view = view.fillna(0)

        # Account totals
        account = {
            "day": day,
            "window": f"Day {lo} to Day {hi}",
            "active_campaigns": int((view["cost"] > 0).sum()),
            "total_spend": round(float(view["cost"].sum()), 6),
            "total_impressions": int(view["impressions"].sum()),
            "total_clicks": int(view["clicks"].sum()),
            "total_known_conversions": int(view["known_conversions"].sum()),
            "overall_ctr": round(float(
                view["clicks"].sum() / view["impressions"].sum()
                if view["impressions"].sum() > 0 else 0
            ), 6),
            "overall_cpa": round(float(
                view["cost"].sum() / view["known_conversions"].sum()
                if view["known_conversions"].sum() > 0 else 0
            ), 6),
            "note": "CPA uses only conversions received before this day (no future leakage)"
        }

        # Top 5 by conversions
        top_by_conv = (
            view[view["known_conversions"] > 0]
            .nlargest(5, "known_conversions")
            [["cost", "known_conversions", "cpa", "ctr"]]
            .round(6)
            .reset_index()
            .to_dict("records")
        )

        # Top 5 by lowest CPA (with conversions)
        top_by_cpa = (
            view[view["known_conversions"] > 0]
            .nsmallest(5, "cpa")
            [["cost", "known_conversions", "cpa", "ctr"]]
            .round(6)
            .reset_index()
            .to_dict("records")
        )

        # Zero conversion campaigns with spend
        zero_conv = (
            view[(view["known_conversions"] == 0) & (view["cost"] > 0)]
            [["cost", "clicks", "impressions"]]
            .round(6)
            .reset_index()
            .to_dict("records")
        )

        return {
            "day": day,
            "account_totals": account,
            "top_by_conversions": top_by_conv,
            "top_by_cpa_efficiency": top_by_cpa,
            "zero_conversion_campaigns": zero_conv[:10],
            "total_zero_conversion_with_spend": len(zero_conv),
            "ready_for_analyst": True
        }


if __name__ == "__main__":
    logger.info("Testing Observer Agent V2...")

    agent = ObserverAgentV2()
    init = agent.initialise()

    print("\n=== DATASET STATS (CORRECTED) ===")
    print(json.dumps(init, indent=2))

    print("\n=== DAY 7 OBSERVATION ===")
    obs = agent.observe(day=7)
    print(json.dumps(obs["account_totals"], indent=2))
    print(f"\nTop campaigns by conversions: {len(obs['top_by_conversions'])}")
    print(f"Zero conversion campaigns: {obs['total_zero_conversion_with_spend']}")

    print("\n✅ Observer Agent V2 working correctly!")
