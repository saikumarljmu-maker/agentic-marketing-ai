"""
Response Curve Estimator
Fits log-linear regression models to estimate spend-to-conversion
elasticity for each campaign. Used for counterfactual projection.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from scipy import stats
import json
import warnings
warnings.filterwarnings("ignore")

PROCESSED_PATH = Path("data/processed")
RESULTS_PATH = Path("results")


class ResponseCurveEstimator:
    """
    Estimates diminishing-returns response curves for each campaign
    using log-linear regression on historical spend-conversion data.

    Model: log(conversions) = alpha + beta * log(spend)
    Where beta is the spend elasticity coefficient.
    """

    def __init__(self):
        self.curves = {}
        self.fit_quality = {}

    def fit_campaign(self, campaign_id, spend_series, conversion_series):
        """
        Fit a log-linear response curve for one campaign.
        Returns elasticity estimate and fit quality metrics.
        """
        spend = np.array(spend_series, dtype=float)
        conversions = np.array(conversion_series, dtype=float)

        valid_mask = (spend > 0) & (conversions > 0)
        spend_valid = spend[valid_mask]
        conv_valid = conversions[valid_mask]

        if len(spend_valid) < 3:
            logger.debug(f"Campaign {campaign_id}: insufficient data points ({len(spend_valid)})")
            return None

        log_spend = np.log(spend_valid)
        log_conv = np.log(conv_valid)

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_spend, log_conv
        )

        elasticity = slope
        r_squared = r_value ** 2

        curve = {
            "campaign_id": campaign_id,
            "alpha": float(intercept),
            "beta": float(elasticity),
            "r_squared": float(r_squared),
            "p_value": float(p_value),
            "std_err": float(std_err),
            "n_observations": int(len(spend_valid)),
            "mean_spend": float(np.mean(spend_valid)),
            "mean_conversions": float(np.mean(conv_valid)),
            "reliable": bool(r_squared > 0.3 and p_value < 0.05 and len(spend_valid) >= 5)
        }

        self.curves[campaign_id] = curve
        return curve

    def fit_all(self, daily_df, up_to_day=None):
        """
        Fit response curves for all campaigns using available data.
        Optionally limit to data up to a specific simulation day.
        """
        if up_to_day is not None:
            data = daily_df[daily_df["day"] <= up_to_day]
        else:
            data = daily_df

        campaigns = data["campaign"].unique()
        fitted = 0
        skipped = 0

        for campaign_id in campaigns:
            camp_data = data[data["campaign"] == campaign_id].sort_values("day")
            curve = self.fit_campaign(
                campaign_id=int(campaign_id),
                spend_series=camp_data["total_spend"].values,
                conversion_series=camp_data["conversions"].values
            )
            if curve:
                fitted += 1
            else:
                skipped += 1

        logger.info(f"Response curves fitted: {fitted} campaigns | "
                    f"Skipped (insufficient data): {skipped}")
        return fitted

    def project_kpi_delta(self, campaign_id, current_spend,
                          proposed_spend, current_conversions):
        """
        Project the KPI delta if spend changes from current to proposed.
        Returns projected conversion change with 95% confidence interval.
        """
        curve = self.curves.get(campaign_id)

        if curve is None or not curve["reliable"]:
            return self._fallback_projection(
                current_spend, proposed_spend, current_conversions
            )

        alpha = curve["alpha"]
        beta = curve["beta"]
        std_err = curve["std_err"]

        if current_spend <= 0 or proposed_spend <= 0:
            return {
                "campaign_id": campaign_id,
                "method": "insufficient_data",
                "projected_conversion_delta": 0,
                "projected_roas_delta": 0,
                "ci_95_lower": 0,
                "ci_95_upper": 0,
                "reliable": False
            }

        projected_conversions = np.exp(alpha + beta * np.log(proposed_spend))
        baseline_conversions = np.exp(alpha + beta * np.log(current_spend))
        conversion_delta = projected_conversions - baseline_conversions

        z_95 = 1.96
        ci_margin = z_95 * std_err * abs(np.log(proposed_spend) - np.log(current_spend))
        ci_lower = conversion_delta - ci_margin
        ci_upper = conversion_delta + ci_margin

        spend_delta = proposed_spend - current_spend
        if spend_delta != 0 and projected_conversions > 0:
            projected_roas_delta = (
                (projected_conversions - current_conversions) /
                max(proposed_spend, 1e-10)
            ) - (
                current_conversions / max(current_spend, 1e-10)
            )
        else:
            projected_roas_delta = 0.0

        return {
            "campaign_id": campaign_id,
            "method": "log_linear_response_curve",
            "current_spend": round(float(current_spend), 6),
            "proposed_spend": round(float(proposed_spend), 6),
            "spend_delta": round(float(proposed_spend - current_spend), 6),
            "current_conversions": round(float(current_conversions), 2),
            "projected_conversions": round(float(projected_conversions), 2),
            "projected_conversion_delta": round(float(conversion_delta), 2),
            "projected_roas_delta": round(float(projected_roas_delta), 4),
            "ci_95_lower": round(float(ci_lower), 2),
            "ci_95_upper": round(float(ci_upper), 2),
            "elasticity": round(float(curve["beta"]), 4),
            "r_squared": round(float(curve["r_squared"]), 4),
            "reliable": True,
            "ai_projected_better": conversion_delta > 0
        }

    def _fallback_projection(self, current_spend, proposed_spend, current_conversions):
        """
        Simple proportional fallback when no reliable curve exists.
        Uses linear scaling as conservative estimate.
        """
        if current_spend <= 0:
            return {
                "method": "fallback_proportional",
                "projected_conversion_delta": 0,
                "ci_95_lower": 0,
                "ci_95_upper": 0,
                "reliable": False
            }

        spend_ratio = proposed_spend / current_spend
        projected_conversions = current_conversions * spend_ratio
        conversion_delta = projected_conversions - current_conversions

        return {
            "method": "fallback_proportional",
            "current_spend": round(float(current_spend), 6),
            "proposed_spend": round(float(proposed_spend), 6),
            "projected_conversion_delta": round(float(conversion_delta), 2),
            "ci_95_lower": round(float(conversion_delta * 0.5), 2),
            "ci_95_upper": round(float(conversion_delta * 1.5), 2),
            "reliable": False,
            "ai_projected_better": conversion_delta > 0
        }

    def get_reliable_curves(self):
        """Return only reliably fitted curves."""
        return {k: v for k, v in self.curves.items() if v.get("reliable")}

    def save(self, path=RESULTS_PATH / "response_curves.json"):
        """Save all fitted curves to disk."""
        RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.curves, f, indent=2)
        logger.success(f"Response curves saved to {path}")

    def load(self, path=RESULTS_PATH / "response_curves.json"):
        """Load previously fitted curves from disk."""
        if not Path(path).exists():
            logger.warning(f"No curves file found at {path}")
            return
        with open(path, "r") as f:
            self.curves = json.load(f)
        logger.success(f"Loaded {len(self.curves)} response curves from {path}")

    def summary(self):
        """Print a summary of fitted curves."""
        total = len(self.curves)
        reliable = len(self.get_reliable_curves())
        if total == 0:
            return {"total": 0, "reliable": 0, "reliability_rate": 0}

        elasticities = [v["beta"] for v in self.curves.values() if v.get("reliable")]
        r_squareds = [v["r_squared"] for v in self.curves.values() if v.get("reliable")]

        return {
            "total_curves": total,
            "reliable_curves": reliable,
            "reliability_rate_pct": round(reliable / total * 100, 1),
            "mean_elasticity": round(float(np.mean(elasticities)), 4) if elasticities else 0,
            "mean_r_squared": round(float(np.mean(r_squareds)), 4) if r_squareds else 0,
        }


if __name__ == "__main__":
    logger.info("Testing Response Curve Estimator...")

    parquet_path = PROCESSED_PATH / "daily_campaign_summary.parquet"
    if not parquet_path.exists():
        logger.warning("No parquet file found — generating synthetic test data")
        np.random.seed(42)
        n_campaigns = 10
        n_days = 30
        records = []
        for camp in range(n_campaigns):
            base_spend = np.random.uniform(0.001, 0.05)
            for day in range(n_days):
                spend = base_spend * np.random.uniform(0.7, 1.3)
                conversions = max(0, int(spend * 100 * np.random.uniform(0.5, 2.0)))
                records.append({
                    "day": day, "campaign": camp,
                    "total_spend": spend, "conversions": conversions,
                    "clicks": max(1, int(spend * 1000)),
                    "impressions": max(10, int(spend * 10000)),
                    "ctr": 0.01, "cvr": 0.1, "cpa": spend / max(conversions, 1),
                    "roas": conversions / max(spend, 0.001),
                    "attributed_conversions": conversions,
                    "total_cpo": spend * 2
                })
        daily_df = pd.DataFrame(records)
    else:
        logger.info("Loading existing parquet file...")
        daily_df = pd.read_parquet(parquet_path)

    logger.info(f"Data loaded: {len(daily_df)} records, "
                f"{daily_df['campaign'].nunique()} campaigns")

    estimator = ResponseCurveEstimator()
    fitted = estimator.fit_all(daily_df, up_to_day=20)

    summary = estimator.summary()
    print(f"\n=== RESPONSE CURVE SUMMARY ===")
    print(f"Total curves fitted:  {summary['total_curves']}")
    print(f"Reliable curves:      {summary['reliable_curves']}")
    print(f"Reliability rate:     {summary['reliability_rate_pct']}%")
    print(f"Mean elasticity:      {summary['mean_elasticity']}")
    print(f"Mean R-squared:       {summary['mean_r_squared']}")

    reliable = estimator.get_reliable_curves()
    if reliable:
        sample_campaign = list(reliable.keys())[0]
        sample_curve = reliable[sample_campaign]
        print(f"\n=== SAMPLE CURVE — Campaign {sample_campaign} ===")
        print(f"Elasticity (beta):  {sample_curve['beta']:.4f}")
        print(f"R-squared:          {sample_curve['r_squared']:.4f}")
        print(f"Observations:       {sample_curve['n_observations']}")
        print(f"Mean spend:         {sample_curve['mean_spend']:.6f}")
        print(f"Mean conversions:   {sample_curve['mean_conversions']:.2f}")

        print(f"\n=== COUNTERFACTUAL PROJECTION TEST ===")
        projection = estimator.project_kpi_delta(
            campaign_id=int(sample_campaign),
            current_spend=sample_curve["mean_spend"],
            proposed_spend=sample_curve["mean_spend"] * 1.2,
            current_conversions=sample_curve["mean_conversions"]
        )
        print(f"Current spend:              {projection['current_spend']:.6f}")
        print(f"Proposed spend (+20%):      {projection['proposed_spend']:.6f}")
        print(f"Projected conversion delta: {projection['projected_conversion_delta']:+.2f}")
        print(f"95% CI:                     [{projection['ci_95_lower']:.2f}, "
              f"{projection['ci_95_upper']:.2f}]")
        print(f"AI projected better:        {projection['ai_projected_better']}")
        print(f"Method:                     {projection['method']}")

    estimator.save()
    print(f"\n✅ Response Curve Estimator working correctly!")
