"""
Observer Agent - Phase 2
Reads the Criteo dataset, validates it, aggregates to daily
campaign-level summaries, and flags anomalies.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from datetime import datetime
import json

DATA_PATH = Path("data/raw/criteo_attribution_dataset/criteo_attribution_dataset.tsv")
PROCESSED_PATH = Path("data/processed")
REQUIRED_COLUMNS = [
    "timestamp", "uid", "campaign", "conversion", "conversion_timestamp",
    "conversion_id", "attribution", "click", "click_pos", "click_nb",
    "cost", "cpo", "time_since_last_click",
    "cat1", "cat2", "cat3", "cat4", "cat5", "cat6", "cat7", "cat8", "cat9"
]


class CriteoDataLoader:
    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path
        self.df = None

    def load(self, sample_size=None):
        logger.info(f"Loading Criteo dataset from {self.data_path}")
        try:
            if sample_size:
                logger.info(f"Loading sample of {sample_size:,} rows")
                self.df = pd.read_csv(
                    self.data_path, sep="\t", nrows=sample_size,
                    dtype={"timestamp": "int64", "uid": "int64", "campaign": "int64",
                           "conversion": "int8", "attribution": "int8", "click": "int8",
                           "cost": "float64", "cpo": "float64"}
                )
            else:
                logger.info("Loading full dataset - this may take 2-3 minutes")
                self.df = pd.read_csv(
                    self.data_path, sep="\t",
                    dtype={"timestamp": "int64", "uid": "int64", "campaign": "int64",
                           "conversion": "int8", "attribution": "int8", "click": "int8",
                           "cost": "float64", "cpo": "float64"}
                )
            logger.success(f"Loaded {len(self.df):,} rows successfully")
            return self.df
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def validate(self):
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        logger.info("Running Level 1 validation checks...")
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_records": len(self.df),
            "checks": {}
        }
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in self.df.columns]
        report["checks"]["required_columns"] = {
            "passed": len(missing_cols) == 0,
            "missing": missing_cols
        }
        null_counts = self.df.isnull().sum()
        high_null = null_counts[null_counts > len(self.df) * 0.5].to_dict()
        report["checks"]["null_values"] = {
            "passed": len(high_null) == 0,
            "high_null_columns": {k: int(v) for k, v in high_null.items()}
        }
        binary_cols = ["conversion", "attribution", "click"]
        binary_issues = {}
        for col in binary_cols:
            if col in self.df.columns:
                unique_vals = self.df[col].unique().tolist()
                valid = all(v in [0, 1] for v in unique_vals)
                if not valid:
                    binary_issues[col] = unique_vals
        report["checks"]["binary_columns"] = {
            "passed": len(binary_issues) == 0,
            "issues": binary_issues
        }
        if "cost" in self.df.columns:
            negative_cost = (self.df["cost"] < 0).sum()
            report["checks"]["cost_values"] = {
                "passed": int(negative_cost) == 0,
                "negative_cost_records": int(negative_cost)
            }
        expected_min = 100_000
        expected_max = 20_000_000
        report["checks"]["record_count"] = {
            "passed": expected_min <= len(self.df) <= expected_max,
            "count": len(self.df),
            "expected_range": f"{expected_min:,} - {expected_max:,}"
        }
        all_passed = all(v["passed"] for v in report["checks"].values())
        report["overall_passed"] = all_passed
        if all_passed:
            logger.success("All validation checks PASSED")
        else:
            failed = [k for k, v in report["checks"].items() if not v["passed"]]
            logger.warning(f"Validation checks FAILED: {failed}")
        return report


class DailyAggregator:
    def __init__(self, df):
        self.df = df
        self.daily_df = None

    def create_day_column(self):
        max_ts = self.df["timestamp"].max()
        day_size = max_ts / 30
        self.df["day"] = (self.df["timestamp"] / day_size).astype(int).clip(0, 29)
        logger.info(f"Created day column - {self.df['day'].nunique()} unique days")

    def aggregate(self):
        logger.info("Aggregating to daily campaign-level summaries...")
        if "day" not in self.df.columns:
            self.create_day_column()
        agg = self.df.groupby(["day", "campaign"]).agg(
            impressions=("timestamp", "count"),
            clicks=("click", "sum"),
            conversions=("conversion", "sum"),
            attributed_conversions=("attribution", "sum"),
            total_spend=("cost", "sum"),
            total_cpo=("cpo", lambda x: x[x > 0].sum()),
        ).reset_index()
        agg["ctr"] = (agg["clicks"] / agg["impressions"]).round(6)
        agg["cvr"] = (agg["conversions"] / agg["clicks"].replace(0, np.nan)).round(6)
        agg["cpa"] = (agg["total_spend"] / agg["conversions"].replace(0, np.nan)).round(6)
        agg["roas"] = (agg["total_cpo"] / agg["total_spend"].replace(0, np.nan)).round(6)
        agg = agg.fillna(0)
        self.daily_df = agg
        logger.success(f"Aggregated to {len(agg):,} daily campaign records")
        logger.info(f"Campaigns: {agg['campaign'].nunique()} | Days: {agg['day'].nunique()}")
        return agg

    def save(self, path=PROCESSED_PATH / "daily_campaign_summary.parquet"):
        if self.daily_df is None:
            raise ValueError("No aggregated data. Call aggregate() first.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.daily_df.to_parquet(path, index=False)
        logger.success(f"Saved daily summaries to {path}")


class AnomalyDetector:
    def __init__(self, daily_df):
        self.daily_df = daily_df
        self.anomalies = []

    def detect(self, day, window=7):
        self.anomalies = []
        current = self.daily_df[self.daily_df["day"] == day]
        historical = self.daily_df[
            (self.daily_df["day"] >= max(0, day - window)) &
            (self.daily_df["day"] < day)
        ]
        if historical.empty or current.empty:
            return []
        hist_avg = historical.groupby("campaign").agg(
            avg_spend=("total_spend", "mean"),
            avg_ctr=("ctr", "mean"),
            avg_cvr=("cvr", "mean"),
            avg_roas=("roas", "mean")
        ).reset_index()
        merged = current.merge(hist_avg, on="campaign", how="left")
        for _, row in merged.iterrows():
            campaign = int(row["campaign"])
            if row["avg_spend"] > 0:
                spend_change = (row["total_spend"] - row["avg_spend"]) / row["avg_spend"]
                if spend_change < -0.5:
                    self.anomalies.append({
                        "day": day, "campaign": campaign,
                        "type": "SPEND_DROP", "severity": "HIGH",
                        "detail": f"Spend dropped {abs(spend_change)*100:.1f}% vs {window}-day avg",
                        "current_value": round(row["total_spend"], 4),
                        "historical_avg": round(row["avg_spend"], 4)
                    })
            if row["avg_ctr"] > 0:
                ctr_change = (row["ctr"] - row["avg_ctr"]) / row["avg_ctr"]
                if ctr_change < -0.4:
                    self.anomalies.append({
                        "day": day, "campaign": campaign,
                        "type": "CTR_DROP", "severity": "MEDIUM",
                        "detail": f"CTR dropped {abs(ctr_change)*100:.1f}% vs {window}-day avg",
                        "current_value": round(row["ctr"], 6),
                        "historical_avg": round(row["avg_ctr"], 6)
                    })
            if row["avg_roas"] > 0 and row["roas"] > 0:
                roas_change = (row["roas"] - row["avg_roas"]) / row["avg_roas"]
                if roas_change < -0.3:
                    self.anomalies.append({
                        "day": day, "campaign": campaign,
                        "type": "ROAS_DROP", "severity": "HIGH",
                        "detail": f"ROAS dropped {abs(roas_change)*100:.1f}% vs {window}-day avg",
                        "current_value": round(row["roas"], 4),
                        "historical_avg": round(row["avg_roas"], 4)
                    })
            if row["avg_cvr"] > 0.001 and row["cvr"] == 0 and row["clicks"] > 10:
                self.anomalies.append({
                    "day": day, "campaign": campaign,
                    "type": "CONVERSION_COLLAPSE", "severity": "CRITICAL",
                    "detail": f"Zero conversions today vs {row['avg_cvr']*100:.2f}% avg CVR",
                    "current_value": 0,
                    "historical_avg": round(row["avg_cvr"], 6)
                })
        if self.anomalies:
            logger.warning(f"Day {day}: {len(self.anomalies)} anomalies detected")
        else:
            logger.info(f"Day {day}: No anomalies detected")
        return self.anomalies


class PerformanceSummaryBuilder:
    def __init__(self, daily_df):
        self.daily_df = daily_df

    def build(self, current_day, window=7):
        start_day = max(0, current_day - window + 1)
        window_data = self.daily_df[
            (self.daily_df["day"] >= start_day) &
            (self.daily_df["day"] <= current_day)
        ]
        if window_data.empty:
            return {"error": "No data available for this period"}
        account_totals = {
            "period": f"Day {start_day} to Day {current_day}",
            "total_impressions": int(window_data["impressions"].sum()),
            "total_clicks": int(window_data["clicks"].sum()),
            "total_conversions": int(window_data["conversions"].sum()),
            "total_spend": round(float(window_data["total_spend"].sum()), 4),
            "overall_ctr": round(float(
                window_data["clicks"].sum() / window_data["impressions"].sum()
                if window_data["impressions"].sum() > 0 else 0
            ), 6),
            "overall_cvr": round(float(
                window_data["conversions"].sum() / window_data["clicks"].sum()
                if window_data["clicks"].sum() > 0 else 0
            ), 6),
            "overall_cpa": round(float(
                window_data["total_spend"].sum() / window_data["conversions"].sum()
                if window_data["conversions"].sum() > 0 else 0
            ), 4),
        }
        campaign_summary = window_data.groupby("campaign").agg(
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            conversions=("conversions", "sum"),
            spend=("total_spend", "sum"),
            avg_ctr=("ctr", "mean"),
            avg_cvr=("cvr", "mean"),
            avg_roas=("roas", "mean"),
            avg_cpa=("cpa", "mean")
        ).reset_index()
        top_campaigns = campaign_summary.nlargest(5, "avg_roas")[
            ["campaign", "spend", "conversions", "avg_roas", "avg_cpa"]
        ].round(4).to_dict("records")
        spending_campaigns = campaign_summary[campaign_summary["spend"] > 0]
        bottom_campaigns = spending_campaigns.nsmallest(5, "avg_roas")[
            ["campaign", "spend", "conversions", "avg_roas", "avg_cpa"]
        ].round(4).to_dict("records")
        zero_conv = campaign_summary[
            (campaign_summary["conversions"] == 0) &
            (campaign_summary["spend"] > 0)
        ][["campaign", "spend", "clicks"]].to_dict("records")
        summary = {
            "simulation_day": current_day,
            "analysis_window_days": window,
            "account_totals": account_totals,
            "top_performing_campaigns": top_campaigns,
            "bottom_performing_campaigns": bottom_campaigns,
            "zero_conversion_campaigns_with_spend": zero_conv[:10],
            "total_active_campaigns": int(
                campaign_summary[campaign_summary["spend"] > 0].shape[0]
            ),
            "total_campaigns_with_conversions": int(
                campaign_summary[campaign_summary["conversions"] > 0].shape[0]
            )
        }
        return summary


class ObserverAgent:
    def __init__(self, sample_size=None):
        self.loader = CriteoDataLoader()
        self.df = None
        self.daily_df = None
        self.sample_size = sample_size

    def initialise(self):
        logger.info("=== Observer Agent Initialising ===")
        self.df = self.loader.load(sample_size=self.sample_size)
        validator = CriteoDataLoader(DATA_PATH)
        validator.df = self.df
        validation_report = validator.validate()
        if not validation_report["overall_passed"]:
            logger.error("Dataset validation failed")
            return validation_report
        aggregator = DailyAggregator(self.df)
        self.daily_df = aggregator.aggregate()
        aggregator.save()
        logger.success("Observer Agent initialised successfully")
        return {
            "status": "ready",
            "validation": validation_report,
            "dataset_stats": {
                "total_impressions": len(self.df),
                "total_campaigns": int(self.df["campaign"].nunique()),
                "total_days": int(self.daily_df["day"].nunique()),
                "total_conversions": int(self.df["conversion"].sum()),
                "total_spend": round(float(self.df["cost"].sum()), 4)
            }
        }

    def observe(self, day):
        logger.info(f"=== Observer Agent - Day {day} ===")
        if self.daily_df is None:
            raise ValueError("Observer not initialised. Call initialise() first.")
        summary_builder = PerformanceSummaryBuilder(self.daily_df)
        performance_summary = summary_builder.build(current_day=day)
        detector = AnomalyDetector(self.daily_df)
        anomalies = detector.detect(day=day)
        return {
            "day": day,
            "performance_summary": performance_summary,
            "anomalies": anomalies,
            "anomaly_count": len(anomalies),
            "ready_for_analyst": True
        }


if __name__ == "__main__":
    logger.info("Running Observer Agent test with 500K sample rows...")
    agent = ObserverAgent(sample_size=500_000)
    init_result = agent.initialise()
    print("\n=== INITIALISATION RESULT ===")
    print(json.dumps(init_result["dataset_stats"], indent=2))
    print("\nValidation passed:", init_result["validation"]["overall_passed"])
    print("\n=== DAY 7 OBSERVATION ===")
    day_result = agent.observe(day=7)
    print(json.dumps(day_result["performance_summary"]["account_totals"], indent=2))
    print(f"\nAnomalies detected: {day_result['anomaly_count']}")
    if day_result["anomalies"]:
        print("\nFirst anomaly:")
        print(json.dumps(day_result["anomalies"][0], indent=2))
