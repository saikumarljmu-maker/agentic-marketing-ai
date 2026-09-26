"""
Full Replay Simulation V2 — Portfolio Manager with Policy Comparison
Compares: logged_mix, uniform, rule_based, thompson_sampling, llm_portfolio
Uses corrected data with no leakage, real calendar days, correct conversions.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent))

from data.criteo_prep import observable_view, realised_outcomes
from agents.observer_v2 import ObserverAgentV2
from agents.portfolio_strategy import PortfolioStrategyAgent, PortfolioAllocator

RESULTS_PATH = Path("results")
LOGS_PATH = Path("logs")
RESULTS_PATH.mkdir(exist_ok=True)
LOGS_PATH.mkdir(exist_ok=True)

# Conversion lag note for LLM
LAG_NOTE = (
    "Conversion lag warning: conversions from the last 3 days are "
    "underreported by approximately 30-50% as many have not yet been "
    "attributed. Do not penalise recently-started campaigns for low "
    "conversion counts due to this lag effect."
)


class ReplaySimulationV2:
    """
    Full replay simulation comparing all policies on the same data.
    Budget fractions tested: 1/2, 1/4, 1/8 of logged spend.
    """

    def __init__(self, n_seeds=3, use_llm=True):
        self.n_seeds = n_seeds
        self.use_llm = use_llm
        self.observer = ObserverAgentV2()
        self.allocator = PortfolioAllocator()
        self.llm_agent = PortfolioStrategyAgent() if use_llm else None
        self.daily_cost = None
        self.conversions = None
        self.results = []
        self.llm_decisions = []

    def setup(self):
        """Load corrected data."""
        logger.info("=== Replay Simulation V2 Setup ===")
        init = self.observer.initialise()
        self.daily_cost = self.observer.daily_cost
        self.conversions = self.observer.conversions

        # Use days 0-23 (exclude last 7 for conversion lag)
        max_day = self.daily_cost["day"].max()
        self.eval_days = list(range(7, max_day - 7))
        logger.info(
            f"Evaluation days: {self.eval_days[0]} to {self.eval_days[-1]} "
            f"({len(self.eval_days)} days)"
        )
        return init

    def run_day(self, day, budget_fraction, seed):
        """Run one day of replay for all policies."""
        # Get observable view (no future leakage)
        view = observable_view(
            self.daily_cost, self.conversions,
            decision_day=day, window=7
        )

        if view.empty:
            return None

        # Today's logged spend as budget
        today_logged = self.daily_cost[
            self.daily_cost["day"] == day
        ]["cost"].sum()

        if today_logged <= 0:
            return None

        budget = today_logged * budget_fraction

        # Ground truth for evaluation
        truth = realised_outcomes(self.daily_cost, self.conversions, day)

        if truth.empty:
            return None

        # Hindsight oracle allocation
        oracle_rate = (
            truth["conversions"] /
            truth["logged_cost"].replace(0, np.nan)
        ).fillna(0)

        rng = np.random.default_rng(seed)
        cap = view.set_index("campaign")["last_day_cost"]

        def oracle_alloc():
            alloc = pd.Series(0.0, index=cap.index)
            remaining = budget
            for c in oracle_rate.sort_values(ascending=False).index:
                if c not in cap.index or remaining <= 0:
                    continue
                take = min(float(cap.get(c, 0)), remaining)
                alloc[c] = take
                remaining -= take
            return alloc

        # Get all policy allocations
        baselines = self.allocator.get_baseline_allocations(view, budget, seed)
        baselines["hindsight_oracle"] = oracle_alloc()

        # LLM portfolio allocation
        if self.use_llm and self.llm_agent:
            strategy = self.llm_agent.get_strategy(
                day=day, view_df=view,
                budget=budget, lag_note=LAG_NOTE
            )
            llm_alloc = self.allocator.allocate(view, budget, strategy, seed)
            baselines["llm_portfolio"] = llm_alloc
            strategy["budget_fraction"] = budget_fraction
            strategy["seed"] = seed
            self.llm_decisions.append(strategy)
        else:
            baselines["llm_portfolio"] = baselines["thompson_sampling"].copy()

        # Evaluate each policy
        day_results = {}
        for policy_name, alloc in baselines.items():
            # Replay: scale conversions proportionally to spend
            conv_sum = 0
            spend_sum = 0
            unspent = 0

            for camp in alloc.index:
                if camp not in truth.index:
                    continue
                logged_spend = float(truth.loc[camp, "logged_cost"])
                logged_conv = float(truth.loc[camp, "conversions"])
                policy_spend = float(alloc.get(camp, 0))

                if logged_spend <= 0:
                    continue

                if policy_spend <= logged_spend:
                    # Scale conversions proportionally
                    scale = policy_spend / logged_spend
                    conv_sum += logged_conv * scale
                    spend_sum += policy_spend
                else:
                    # Cannot spend above logged — cap at logged
                    conv_sum += logged_conv
                    spend_sum += logged_spend
                    unspent += policy_spend - logged_spend

            day_results[policy_name] = {
                "conversions": round(conv_sum, 2),
                "spend": round(spend_sum, 4),
                "cpa": round(spend_sum / conv_sum, 4) if conv_sum > 0 else None,
                "unspent": round(unspent, 4)
            }

        # Oracle for % comparison
        oracle_conv = day_results.get("hindsight_oracle", {}).get("conversions", 1)

        return {
            "day": day,
            "budget_fraction": budget_fraction,
            "seed": seed,
            "logged_spend": round(float(today_logged), 4),
            "budget": round(float(budget), 4),
            "policies": day_results,
            "oracle_conversions": round(float(oracle_conv), 2)
        }

    def run(self, budget_fractions=None):
        """Run full simulation across all days, budgets, and seeds."""
        if budget_fractions is None:
            budget_fractions = [0.5, 0.25, 0.125]

        logger.info("=== Starting Replay Simulation V2 ===")
        start_time = datetime.now()
        self.setup()
        logger.info(
            f"Days: {len(self.eval_days)} | "
            f"Budget fractions: {budget_fractions} | "
            f"Seeds: {self.n_seeds} | "
            f"LLM: {self.use_llm}"
        )

        for day in self.eval_days:
            for bf in budget_fractions:
                for seed in range(self.n_seeds):
                    try:
                        result = self.run_day(day, bf, seed)
                        if result:
                            self.results.append(result)
                    except Exception as e:
                        logger.error(f"Day {day} bf={bf} seed={seed} failed: {e}")

            if day % 5 == 0:
                logger.info(
                    f"Progress: Day {day}/{self.eval_days[-1]} | "
                    f"Results: {len(self.results)}"
                )

        return self._finalise(start_time)

    def _finalise(self, start_time):
        """Aggregate results and save."""
        logger.info("=== Finalising Results ===")

        df = pd.DataFrame([
            {
                "day": r["day"],
                "budget_fraction": r["budget_fraction"],
                "seed": r["seed"],
                "policy": policy,
                "conversions": metrics["conversions"],
                "spend": metrics["spend"],
                "cpa": metrics["cpa"],
                "oracle_conversions": r["oracle_conversions"]
            }
            for r in self.results
            for policy, metrics in r["policies"].items()
        ])

        df["pct_of_oracle"] = (
            df["conversions"] / df["oracle_conversions"] * 100
        ).round(2)

        # Summary by policy and budget fraction
        summary = df.groupby(["policy", "budget_fraction"]).agg(
            mean_conversions=("conversions", "mean"),
            mean_cpa=("cpa", "mean"),
            mean_pct_oracle=("pct_of_oracle", "mean"),
            total_conversions=("conversions", "sum"),
            n_days=("day", "count")
        ).round(4)

        print("\n" + "="*80)
        print("REPLAY SIMULATION V2 — RESULTS SUMMARY")
        print("="*80)
        print("\nMean % of Hindsight Oracle by Policy and Budget Fraction:")
        print("-"*80)

        pivot = df.groupby(["policy", "budget_fraction"])[
            "pct_of_oracle"
        ].mean().round(2).unstack()
        print(pivot.to_string())

        print("\nMean Conversions per Day:")
        print("-"*80)
        conv_pivot = df.groupby(["policy", "budget_fraction"])[
            "conversions"
        ].mean().round(2).unstack()
        print(conv_pivot.to_string())

        # Save results
        df.to_parquet(RESULTS_PATH / "replay_results_v2.parquet", index=False)

        summary_dict = summary.reset_index().to_dict("records")
        with open(RESULTS_PATH / "replay_summary_v2.json", "w") as f:
            json.dump(summary_dict, f, indent=2)

        if self.llm_decisions:
            with open(LOGS_PATH / "llm_portfolio_decisions.jsonl", "w") as f:
                for d in self.llm_decisions:
                    f.write(json.dumps(d) + "\n")

            total_cost = sum(
                d.get("cost_usd", 0) for d in self.llm_decisions
            )
            logger.info(f"Total LLM cost: ${round(total_cost, 4)}")

        duration = (datetime.now() - start_time).total_seconds()
        logger.success(
            f"Simulation complete in {duration:.0f}s | "
            f"{len(self.results)} day-budget-seed combinations"
        )

        return {
            "summary": summary_dict,
            "total_results": len(self.results),
            "policies_compared": list(df["policy"].unique()),
            "budget_fractions": list(df["budget_fraction"].unique()),
            "duration_seconds": round(duration, 1)
        }


if __name__ == "__main__":
    logger.info("Starting Full Replay Simulation V2...")
    logger.info("This will take approximately 15-20 minutes")
    logger.info("Comparing: logged_mix, uniform, rule_based, thompson, llm_portfolio, oracle")

    sim = ReplaySimulationV2(n_seeds=3, use_llm=True)
    results = sim.run(budget_fractions=[0.5, 0.25, 0.125])

    print("\n=== POLICIES COMPARED ===")
    for p in results["policies_compared"]:
        print(f"  - {p}")

    print(f"\nTotal results: {results['total_results']}")
    print(f"Duration: {results['duration_seconds']}s")
    print("\n✅ Replay simulation complete!")
    print("Results saved to results/replay_results_v2.parquet")
