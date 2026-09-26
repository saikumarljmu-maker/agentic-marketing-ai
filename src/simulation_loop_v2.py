"""
Replay Simulation V2 — Round 3 fixes.
budget_fraction stored with LLM decisions.
Fallbacks cached correctly.
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

LAG_NOTE = (
    "\nConversion lag warning: conversions from the last 3 days are "
    "underreported as many have not yet been attributed. "
    "Do not penalise recently-started campaigns for low conversion counts."
)


class ReplaySimulationV2:

    def __init__(self, n_seeds=3, use_llm=True,
                 lag_note=LAG_NOTE, run_name="v2"):
        self.n_seeds = n_seeds
        self.use_llm = use_llm
        self.lag_note = lag_note
        self.run_name = run_name
        self.observer = ObserverAgentV2()
        self.allocator = PortfolioAllocator()
        self.llm_agent = PortfolioStrategyAgent() if use_llm else None
        self.daily_cost = None
        self.conversions = None
        self.results = []
        self.run_meta = []
        self._llm_cache = {}

    def setup(self):
        logger.info("=== Replay Simulation V2 Setup ===")
        self.observer.initialise()
        self.daily_cost = self.observer.daily_cost
        self.conversions = self.observer.conversions
        max_day = self.daily_cost["day"].max()
        self.eval_days = list(range(7, min(24, max_day - 6)))
        logger.info(f"Evaluation days: {self.eval_days[0]} to "
                    f"{self.eval_days[-1]} ({len(self.eval_days)} days)")

    def run_day(self, day, budget_fraction, seed):
        view = observable_view(
            self.daily_cost, self.conversions,
            decision_day=day, window=7
        )
        if view.empty:
            return None

        today_logged = self.daily_cost[
            self.daily_cost["day"] == day]["cost"].sum()
        if today_logged <= 0:
            return None

        budget = today_logged * budget_fraction
        truth = realised_outcomes(self.daily_cost, self.conversions, day)
        if truth.empty:
            return None

        oracle_rate = (
            truth["conversions"] /
            truth["logged_cost"].replace(0, np.nan)
        ).fillna(0)

        def oracle_alloc():
            cap = truth["logged_cost"]
            alloc = pd.Series(0.0, index=cap.index)
            remaining = budget
            for c in oracle_rate.sort_values(ascending=False).index:
                if remaining <= 0:
                    break
                take = min(float(cap.get(c, 0)), remaining)
                alloc[c] = take
                remaining -= take
            return alloc

        baselines = self.allocator.get_baselines(view, budget, seed)
        baselines["hindsight_oracle"] = oracle_alloc()

        meta = {"day": day, "budget_fraction": budget_fraction,
                "seed": seed, "fallback": False, "success": True}

        if self.use_llm and self.llm_agent:
            cache_key = f"{day}_{budget_fraction}"
            strategy = self.llm_agent.get_strategy(
                day=day, view_df=view, budget=budget,
                lag_note=self.lag_note,
                cache=self._llm_cache, cache_key=cache_key,
                budget_fraction=budget_fraction
            )
            try:
                llm_alloc = self.allocator.allocate(view, budget, strategy, seed)
                baselines["llm_portfolio"] = llm_alloc
                meta["fallback"] = bool(strategy.get("fallback", False))
                meta["strategy"] = strategy.get("strategy")
                meta["escalate"] = bool(strategy.get("escalate_to_human", False))
                meta["confidence"] = strategy.get("confidence")
            except Exception as e:
                logger.error(f"Allocator failed Day {day}: {e}")
                baselines["llm_portfolio"] = baselines["thompson_sampling"].copy()
                meta["success"] = False
                meta["error"] = str(e)
        else:
            baselines["llm_portfolio"] = baselines["thompson_sampling"].copy()

        self.run_meta.append(meta)

        day_results = {}
        for policy_name, alloc in baselines.items():
            conv_sum = spend_sum = unspent = 0
            for camp in alloc.index:
                if camp not in truth.index:
                    unspent += float(alloc.get(camp, 0))
                    continue
                logged_spend = float(truth.loc[camp, "logged_cost"])
                logged_conv = float(truth.loc[camp, "conversions"])
                policy_spend = float(alloc.get(camp, 0))
                if logged_spend <= 0:
                    unspent += policy_spend
                    continue
                if policy_spend <= logged_spend:
                    conv_sum += logged_conv * (policy_spend / logged_spend)
                    spend_sum += policy_spend
                else:
                    conv_sum += logged_conv
                    spend_sum += logged_spend
                    unspent += policy_spend - logged_spend

            day_results[policy_name] = {
                "conversions": round(conv_sum, 2),
                "spend": round(spend_sum, 4),
                "cpa": round(spend_sum / conv_sum, 4) if conv_sum > 0 else None,
                "unspent": round(unspent, 4)
            }

        oracle_conv = day_results.get("hindsight_oracle", {}).get("conversions", 1)
        return {
            "day": day, "budget_fraction": budget_fraction, "seed": seed,
            "logged_spend": round(float(today_logged), 4),
            "budget": round(float(budget), 4),
            "policies": day_results,
            "oracle_conversions": round(float(oracle_conv), 2)
        }

    def run(self, budget_fractions=None):
        if budget_fractions is None:
            budget_fractions = [0.5, 0.25, 0.125]

        self.setup()
        logger.info(f"=== Starting Replay Simulation — run_name={self.run_name} ===")
        logger.info(f"Days: {len(self.eval_days)} | "
                    f"Budgets: {budget_fractions} | Seeds: {self.n_seeds}")

        start_time = datetime.now()

        for day in self.eval_days:
            for bf in budget_fractions:
                for seed in range(self.n_seeds):
                    try:
                        result = self.run_day(day, bf, seed)
                        if result:
                            self.results.append(result)
                    except Exception as e:
                        logger.error(f"Day {day} bf={bf} seed={seed}: {e}")
            if day % 5 == 0:
                logger.info(f"Progress: Day {day}/{self.eval_days[-1]}")

        return self._finalise(start_time)

    def _finalise(self, start_time):
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

        run = self.run_name
        df.to_parquet(RESULTS_PATH / f"replay_results_{run}.parquet", index=False)

        with open(RESULTS_PATH / f"run_meta_{run}.json", "w") as f:
            json.dump(self.run_meta, f, indent=2)

        if self.llm_agent and self.llm_agent.decisions:
            self.llm_agent.save(run_name=run)
            total_cost = sum(d.get("cost_usd", 0) for d in self.llm_agent.decisions)
            logger.info(f"Total LLM cost: ${round(total_cost, 4)}")

        print("\n" + "="*80)
        print(f"REPLAY SIMULATION V2 — {run.upper()} RESULTS")
        print("="*80)
        print("\nMean % of Hindsight Oracle:")
        pivot = df.groupby(["policy", "budget_fraction"])[
            "pct_of_oracle"].mean().round(2).unstack()
        print(pivot.to_string())
        print("\nMean Conversions per Day:")
        conv_pivot = df.groupby(["policy", "budget_fraction"])[
            "conversions"].mean().round(2).unstack()
        print(conv_pivot.to_string())

        duration = (datetime.now() - start_time).total_seconds()
        logger.success(f"Done in {duration:.0f}s | {len(self.results)} results | run={run}")

        return {"total_results": len(self.results),
                "policies": list(df["policy"].unique()),
                "run_name": run, "duration_seconds": round(duration, 1)}


if __name__ == "__main__":
    logger.info("Starting Replay Simulation V2 (Round 3 fixes)...")
    sim = ReplaySimulationV2(n_seeds=3, use_llm=True,
                              lag_note=LAG_NOTE, run_name="v2")
    sim.run(budget_fractions=[0.5, 0.25, 0.125])
