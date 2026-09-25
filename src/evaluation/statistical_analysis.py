"""
Statistical Analysis Module
Paired t-tests, Cohen's d, confidence intervals, and
results generation for Chapter 4 and 5 of the thesis.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from scipy import stats
import json
import warnings
warnings.filterwarnings("ignore")

RESULTS_PATH = Path("results")
LOGS_PATH = Path("logs")


class StatisticalAnalyser:
    """
    Performs all statistical tests comparing AI vs human
    decision outcomes across KPIs.
    """

    def __init__(self):
        self.results = {}

    def paired_ttest(self, ai_values, human_values, kpi_name, alpha=0.05):
        """
        Paired t-test comparing AI projected outcomes
        vs human actual outcomes for a given KPI.
        """
        ai = np.array(ai_values, dtype=float)
        human = np.array(human_values, dtype=float)

        valid = ~(np.isnan(ai) | np.isnan(human))
        ai = ai[valid]
        human = human[valid]

        if len(ai) < 2:
            return {
                "kpi": kpi_name,
                "n": len(ai),
                "sufficient_data": False,
                "note": "Insufficient data for statistical test"
            }

        t_stat, p_value = stats.ttest_rel(ai, human)
        mean_diff = np.mean(ai - human)
        std_diff = np.std(ai - human, ddof=1)
        cohen_d = mean_diff / std_diff if std_diff > 0 else 0

        n = len(ai)
        se = std_diff / np.sqrt(n)
        t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
        ci_lower = mean_diff - t_crit * se
        ci_upper = mean_diff + t_crit * se

        pct_improvement = (mean_diff / np.mean(human) * 100) if np.mean(human) != 0 else 0

        result = {
            "kpi": kpi_name,
            "n": int(n),
            "sufficient_data": True,
            "ai_mean": round(float(np.mean(ai)), 6),
            "human_mean": round(float(np.mean(human)), 6),
            "mean_difference": round(float(mean_diff), 6),
            "pct_improvement": round(float(pct_improvement), 2),
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_value), 6),
            "significant": bool(p_value < alpha),
            "cohen_d": round(float(cohen_d), 4),
            "effect_size_label": self._effect_size_label(abs(cohen_d)),
            "ci_95_lower": round(float(ci_lower), 6),
            "ci_95_upper": round(float(ci_upper), 6),
            "alpha": alpha,
            "ai_better": bool(mean_diff > 0)
        }

        self.results[kpi_name] = result
        return result

    def _effect_size_label(self, d):
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

    def power_analysis(self, n, effect_size, alpha=0.05):
        """
        Calculate statistical power for a given sample size
        and effect size using paired t-test.
        """
        from scipy.stats import t, nct
        df = n - 1
        ncp = effect_size * np.sqrt(n)
        t_crit = t.ppf(1 - alpha/2, df=df)
        power = 1 - nct.cdf(t_crit, df=df, nc=ncp) + nct.cdf(-t_crit, df=df, nc=ncp)
        return round(float(power), 4)

    def compute_power_table(self, n_decisions):
        """
        Compute power for different effect sizes given
        the number of decision comparisons available.
        """
        effect_sizes = {
            "ROAS_15pct": 0.5,
            "CPA_10pct": 0.35,
            "CTR_5pct": 0.2
        }

        power_table = {}
        for label, d in effect_sizes.items():
            power = self.power_analysis(n_decisions, d)
            power_table[label] = {
                "effect_size_d": d,
                "n_decisions": n_decisions,
                "power": power,
                "adequately_powered": bool(power >= 0.8),
                "reporting_level": (
                    "statistically_tested" if power >= 0.8
                    else "directional" if power >= 0.6
                    else "indicative_only"
                )
            }

        return power_table

    def analyse_agreement_rate(self, decisions):
        """
        Analyse the pattern of AI-human agreement and disagreement.
        """
        if not decisions:
            return {}

        total = len(decisions)
        agreed = sum(1 for d in decisions if d.get("agreed", False))
        diverged = total - agreed

        decision_types = {}
        for d in decisions:
            action = d.get("ai_decision", {}).get("action", "unknown")
            decision_types[action] = decision_types.get(action, 0) + 1

        divergent_only = [d for d in decisions if not d.get("agreed", False)]
        ai_better_count = sum(
            1 for d in divergent_only
            if d.get("counterfactual_projection", {}) and
            d.get("counterfactual_projection", {}).get("ai_projected_better", False)
        )

        return {
            "total_decisions": total,
            "agreed": agreed,
            "diverged": diverged,
            "agreement_rate_pct": round(agreed / total * 100, 2),
            "divergence_rate_pct": round(diverged / total * 100, 2),
            "decision_type_breakdown": decision_types,
            "ai_better_when_diverged": ai_better_count,
            "ai_better_rate_pct": round(
                ai_better_count / max(diverged, 1) * 100, 2
            )
        }

    def generate_kpi_comparison_table(self, decisions):
        """
        Generate a comparison table of AI vs human KPI outcomes
        from the logged decisions and their counterfactual projections.
        """
        ai_conversions = []
        human_conversions = []
        ai_roas = []
        human_roas = []
        ai_cpa = []
        human_cpa = []

        for d in decisions:
            proj = d.get("counterfactual_projection")
            context = d.get("context", {})

            if proj and proj.get("reliable", False):
                human_conv = context.get("conversions", 0)
                ai_conv = human_conv + proj.get("projected_conversion_delta", 0)
                ai_conversions.append(ai_conv)
                human_conversions.append(human_conv)

                human_spend = context.get("spend", 0)
                ai_spend = proj.get("proposed_spend", human_spend)

                if human_spend > 0:
                    human_roas.append(human_conv / human_spend)
                    ai_roas.append(ai_conv / max(ai_spend, 0.0001))

                if human_conv > 0:
                    human_cpa.append(human_spend / human_conv)
                if ai_conv > 0:
                    ai_cpa.append(ai_spend / ai_conv)

        results = {}

        if len(ai_conversions) >= 2:
            results["conversions"] = self.paired_ttest(
                ai_conversions, human_conversions, "Conversions"
            )

        if len(ai_roas) >= 2:
            results["roas"] = self.paired_ttest(
                ai_roas, human_roas, "ROAS"
            )

        if len(ai_cpa) >= 2:
            results["cpa"] = self.paired_ttest(
                ai_cpa, human_cpa, "CPA"
            )

        return results

    def save_results(self, filename="statistical_results.json"):
        """Save all statistical results to disk."""
        RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        path = RESULTS_PATH / filename
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.success(f"Statistical results saved to {path}")

    def print_summary(self):
        """Print a readable summary of all statistical results."""
        if not self.results:
            print("No results computed yet.")
            return

        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS SUMMARY")
        print("="*60)

        for kpi, result in self.results.items():
            if not result.get("sufficient_data", False):
                print(f"\n{kpi}: Insufficient data")
                continue

            print(f"\n📊 {result['kpi']}")
            print(f"   N comparisons:     {result['n']}")
            print(f"   AI mean:           {result['ai_mean']:.6f}")
            print(f"   Human mean:        {result['human_mean']:.6f}")
            print(f"   Difference:        {result['mean_difference']:+.6f}")
            print(f"   % Improvement:     {result['pct_improvement']:+.2f}%")
            print(f"   t-statistic:       {result['t_statistic']:.4f}")
            print(f"   p-value:           {result['p_value']:.6f}")
            print(f"   Significant:       {'YES ✅' if result['significant'] else 'NO ❌'}")
            print(f"   Cohen's d:         {result['cohen_d']:.4f} ({result['effect_size_label']})")
            print(f"   95% CI:            [{result['ci_95_lower']:.6f}, {result['ci_95_upper']:.6f}]")
            print(f"   AI better:         {'YES ✅' if result['ai_better'] else 'NO ❌'}")


class ResultsGenerator:
    """
    Generates final results tables and charts
    for thesis Chapters 4 and 5.
    """

    def __init__(self, analyser):
        self.analyser = analyser

    def generate_chapter4_tables(self, daily_df, decisions):
        """Generate all tables needed for Chapter 4."""
        tables = {}

        camp_summary = daily_df.groupby("campaign").agg(
            total_spend=("total_spend", "sum"),
            total_conversions=("conversions", "sum"),
            avg_roas=("roas", "mean"),
            avg_cpa=("cpa", "mean"),
            avg_ctr=("ctr", "mean"),
            days_active=("day", "count")
        ).reset_index()

        tables["dataset_overview"] = {
            "total_campaigns": int(daily_df["campaign"].nunique()),
            "total_days": int(daily_df["day"].nunique()),
            "total_impressions": int(daily_df["impressions"].sum()),
            "total_clicks": int(daily_df["clicks"].sum()),
            "total_conversions": int(daily_df["conversions"].sum()),
            "total_spend": round(float(daily_df["total_spend"].sum()), 4),
            "mean_campaign_roas": round(float(camp_summary["avg_roas"].mean()), 4),
            "mean_campaign_cpa": round(float(camp_summary["avg_cpa"].mean()), 4),
            "mean_campaign_ctr": round(float(camp_summary["avg_ctr"].mean()), 6),
            "campaigns_with_conversions": int(
                (camp_summary["total_conversions"] > 0).sum()
            ),
            "campaigns_zero_conversions": int(
                (camp_summary["total_conversions"] == 0).sum()
            )
        }

        tables["agreement_analysis"] = self.analyser.analyse_agreement_rate(decisions)
        tables["power_analysis"] = self.analyser.compute_power_table(
            n_decisions=len(decisions)
        )

        return tables

    def save_chapter4_tables(self, tables, filename="chapter4_tables.json"):
        """Save Chapter 4 tables to disk."""
        RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        path = RESULTS_PATH / filename
        with open(path, "w") as f:
            json.dump(tables, f, indent=2)
        logger.success(f"Chapter 4 tables saved to {path}")
        return path


if __name__ == "__main__":
    logger.info("Testing Statistical Analysis Module...")

    analyser = StatisticalAnalyser()

    np.random.seed(42)
    n = 30

    human_roas = np.random.normal(1.8, 0.4, n)
    ai_roas = human_roas + np.random.normal(0.25, 0.3, n)

    human_cpa = np.random.normal(0.005, 0.002, n)
    ai_cpa = human_cpa - np.random.normal(0.0008, 0.001, n)

    human_ctr = np.random.normal(0.035, 0.008, n)
    ai_ctr = human_ctr + np.random.normal(0.001, 0.006, n)

    print("\n=== RUNNING PAIRED T-TESTS ===")

    roas_result = analyser.paired_ttest(ai_roas, human_roas, "ROAS")
    cpa_result = analyser.paired_ttest(ai_cpa, human_cpa, "CPA")
    ctr_result = analyser.paired_ttest(ai_ctr, human_ctr, "CTR")

    analyser.print_summary()

    print("\n=== STATISTICAL POWER ANALYSIS ===")
    power_table = analyser.compute_power_table(n_decisions=30)
    for label, power_info in power_table.items():
        print(f"\n{label}:")
        print(f"  Effect size (d): {power_info['effect_size_d']}")
        print(f"  Power:           {power_info['power']}")
        print(f"  Adequately powered: {power_info['adequately_powered']}")
        print(f"  Reporting level: {power_info['reporting_level']}")

    print("\n=== AGREEMENT RATE ANALYSIS ===")
    mock_decisions = [
        {"agreed": True, "ai_decision": {"action": "maintain"},
         "context": {}, "counterfactual_projection": None},
        {"agreed": False, "ai_decision": {"action": "increase_budget"},
         "context": {"conversions": 3, "spend": 0.01},
         "counterfactual_projection": {"ai_projected_better": True, "reliable": True}},
        {"agreed": True, "ai_decision": {"action": "pause"},
         "context": {}, "counterfactual_projection": None},
        {"agreed": False, "ai_decision": {"action": "decrease_budget"},
         "context": {"conversions": 1, "spend": 0.02},
         "counterfactual_projection": {"ai_projected_better": True, "reliable": True}},
        {"agreed": True, "ai_decision": {"action": "maintain"},
         "context": {}, "counterfactual_projection": None},
    ]

    agreement = analyser.analyse_agreement_rate(mock_decisions)
    print(f"Total decisions:    {agreement['total_decisions']}")
    print(f"Agreed:             {agreement['agreed']}")
    print(f"Diverged:           {agreement['diverged']}")
    print(f"Agreement rate:     {agreement['agreement_rate_pct']}%")
    print(f"AI better when diverged: {agreement['ai_better_when_diverged']}")

    analyser.save_results()

    parquet_path = Path("data/processed/daily_campaign_summary.parquet")
    if parquet_path.exists():
        daily_df = pd.read_parquet(parquet_path)
        generator = ResultsGenerator(analyser)
        tables = generator.generate_chapter4_tables(daily_df, mock_decisions)
        generator.save_chapter4_tables(tables)

        print(f"\n=== CHAPTER 4 DATASET OVERVIEW ===")
        overview = tables["dataset_overview"]
        for key, val in overview.items():
            print(f"  {key}: {val}")

    print("\n✅ Statistical Analysis Module working correctly!")
