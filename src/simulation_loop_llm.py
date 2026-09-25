"""
Full LLM Simulation Loop
Runs the complete 30-day retrospective simulation
using Claude for Analyst + Strategy agents.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent))

from agents.observer import ObserverAgent
from agents.analyst import AnalystAgent
from agents.strategy import StrategyAgent
from agents.memory import MemoryLayer
from agents.execution import ExecutionAgent
from evaluation.response_curves import ResponseCurveEstimator
from evaluation.statistical_analysis import StatisticalAnalyser, ResultsGenerator

RESULTS_PATH = Path("results")
LOGS_PATH = Path("logs")


class LLMSimulationLoop:
    """
    Full LLM-powered 30-day retrospective simulation.
    Uses Claude for Analyst and Strategy agents.
    """

    def __init__(self, simulation_name="criteo_llm_sim", sample_size=500_000):
        self.simulation_name = simulation_name
        self.sample_size = sample_size
        self.observer = None
        self.analyst = None
        self.strategy = None
        self.memory = None
        self.execution = None
        self.curve_estimator = None
        self.analyser = StatisticalAnalyser()
        self.daily_df = None
        self.all_decisions = []
        self.total_tokens_used = {"input": 0, "output": 0}
        logger.info(f"LLM Simulation Loop initialised: {simulation_name}")

    def setup(self):
        """Initialise all agents."""
        logger.info("=== LLM SIMULATION SETUP ===")

        self.observer = ObserverAgent(sample_size=self.sample_size)
        init_result = self.observer.initialise()
        if init_result.get("status") != "ready":
            raise RuntimeError("Observer failed to initialise")
        self.daily_df = self.observer.daily_df
        logger.success(f"Data: {len(self.daily_df):,} daily records")

        self.analyst = AnalystAgent()
        self.strategy = StrategyAgent()
        self.memory = MemoryLayer(
            collection_name=f"{self.simulation_name}_memory"
        )
        self.execution = ExecutionAgent(
            simulation_name=self.simulation_name
        )
        self.curve_estimator = ResponseCurveEstimator()

        checkpoint = self.execution.load_checkpoint()
        if checkpoint:
            resume_day = checkpoint.get("last_completed_day", -1)
            logger.info(f"Resuming from Day {resume_day + 1}")
            return resume_day
        return -1

    def _infer_human_decision(self, campaign_id, day, metrics):
        """Infer human decision from spend changes day-over-day."""
        if day == 0:
            return {"action": "maintain", "reason": "First day baseline"}

        prev = self.daily_df[
            (self.daily_df["campaign"] == campaign_id) &
            (self.daily_df["day"] == day - 1)
        ]
        if prev.empty:
            return {"action": "maintain", "reason": "No previous data"}

        prev_spend = float(prev["total_spend"].iloc[0])
        curr_spend = metrics.get("spend", 0)

        if prev_spend > 0:
            change = (curr_spend - prev_spend) / prev_spend
            if change > 0.15:
                return {"action": "increase_budget",
                        "amount": round(curr_spend - prev_spend, 6),
                        "reason": f"Human increased spend {change*100:.1f}%"}
            elif change < -0.15:
                return {"action": "decrease_budget",
                        "amount": round(prev_spend - curr_spend, 6),
                        "reason": f"Human decreased spend {abs(change)*100:.1f}%"}
            elif curr_spend == 0 and prev_spend > 0:
                return {"action": "pause", "reason": "Human stopped spending"}

        return {"action": "maintain",
                "reason": f"Spend stable at {curr_spend:.6f}"}

    def run_day(self, day):
        """Run one complete simulation day with full LLM pipeline."""
        logger.info(f"\n{'='*50}")
        logger.info(f"DAY {day} — LLM PIPELINE")
        logger.info(f"{'='*50}")

        # Step 1 — Observe
        observation = self.observer.observe(day=day)
        performance_summary = observation["performance_summary"]
        anomalies = observation["anomalies"]

        account = performance_summary.get("account_totals", {})
        logger.info(
            f"Day {day} data: "
            f"{account.get('total_conversions', 0)} conversions | "
            f"Spend: {account.get('total_spend', 0):.4f}"
        )

        # Step 2 — Memory retrieval
        memory_context = self.memory.retrieve(
            query=f"day {day} campaign performance ROAS CPA conversions spend",
            n_results=3
        )

        # Step 3 — Analyst
        analysis_result = self.analyst.analyse(
            day=day,
            performance_summary=performance_summary,
            anomalies=anomalies[:10],
            memory_context=memory_context
        )

        self.total_tokens_used["input"] += analysis_result.get("input_tokens", 0)
        self.total_tokens_used["output"] += analysis_result.get("output_tokens", 0)

        if not analysis_result.get("analysis"):
            logger.warning(f"Day {day} — Analyst failed, using fallback")
            return []

        # Step 4 — Strategy
        strategy_result = self.strategy.recommend(
            day=day,
            analysis=analysis_result["analysis"],
            performance_summary=performance_summary,
            memory_context=memory_context
        )

        self.total_tokens_used["input"] += strategy_result.get("input_tokens", 0)
        self.total_tokens_used["output"] += strategy_result.get("output_tokens", 0)

        # Step 5 — Execute and log decisions
        day_data = self.daily_df[self.daily_df["day"] == day]
        campaign_metrics = day_data.groupby("campaign").agg(
            spend=("total_spend", "sum"),
            conversions=("conversions", "sum"),
            avg_roas=("roas", "mean"),
            avg_cpa=("cpa", "mean"),
            avg_ctr=("ctr", "mean")
        ).reset_index()

        self.curve_estimator.fit_all(self.daily_df, up_to_day=day)

        day_decisions = []
        recommendations = strategy_result.get("recommendations", [])

        for rec in recommendations:
            campaign_id = rec.get("campaign_id")
            if campaign_id is None:
                continue

            camp_row = campaign_metrics[
                campaign_metrics["campaign"] == campaign_id
            ]
            if camp_row.empty:
                continue

            metrics = camp_row.iloc[0].to_dict()
            current_spend = float(metrics.get("spend", 0))

            ai_decision = {
                "action": rec.get("action"),
                "reason": rec.get("reason"),
                "confidence": rec.get("confidence"),
                "decision_type": rec.get("decision_type"),
                "expected_kpi_impact": rec.get("expected_kpi_impact"),
                "priority": rec.get("priority"),
                "source": "llm_strategy_agent"
            }

            human_decision = self._infer_human_decision(
                campaign_id, day, metrics
            )

            context = {
                "day": day,
                "campaign": campaign_id,
                "spend": round(current_spend, 6),
                "conversions": int(metrics.get("conversions", 0)),
                "roas": round(float(metrics.get("avg_roas", 0)), 4),
                "cpa": round(float(metrics.get("avg_cpa", 0)), 6),
                "ctr": round(float(metrics.get("avg_ctr", 0)), 6),
                "human_review_required": strategy_result.get(
                    "human_review_required", False
                ),
                "analyst_summary": analysis_result["analysis"][:300]
            }

            decision_id = self.execution.execute(
                day=day,
                campaign=campaign_id,
                ai_decision=ai_decision,
                human_decision=human_decision,
                context=context,
                analyst_output=analysis_result["analysis"]
            )

            # Counterfactual projection for divergent decisions
            if not self.execution.logger.decisions[-1]["agreed"]:
                action = rec.get("action", "")
                if action == "increase_budget":
                    proposed = current_spend * 1.20
                elif action == "decrease_budget":
                    proposed = current_spend * 0.70
                elif action == "pause":
                    proposed = 0.0
                else:
                    proposed = current_spend

                if proposed != current_spend and current_spend > 0:
                    projection = self.curve_estimator.project_kpi_delta(
                        campaign_id=int(campaign_id),
                        current_spend=current_spend,
                        proposed_spend=proposed,
                        current_conversions=float(
                            metrics.get("conversions", 0)
                        )
                    )
                    self.execution.add_counterfactual(decision_id, projection)

            # Store in memory
            self.memory.store(
                day=day,
                campaign=int(campaign_id),
                decision_type=rec.get("decision_type", "UNKNOWN"),
                decision=ai_decision,
                context=context
            )

            day_decisions.append({
                "id": decision_id,
                "day": day,
                "campaign": campaign_id,
                "ai_action": ai_decision["action"],
                "human_action": human_decision["action"],
                "agreed": self.execution.logger.decisions[-1]["agreed"],
                "confidence": rec.get("confidence"),
                "human_review": strategy_result.get("human_review_required")
            })

        logger.info(
            f"Day {day} complete — "
            f"{len(day_decisions)} LLM decisions | "
            f"Agreement: {self.execution.get_agreement_rate():.1f}% | "
            f"Tokens: {self.total_tokens_used['input']}in "
            f"{self.total_tokens_used['output']}out"
        )

        return day_decisions

    def run(self, start_day=0, end_day=29):
        """Run full LLM simulation."""
        logger.info("=== STARTING FULL LLM SIMULATION ===")
        logger.info("Using Claude Sonnet 4.6 for Analyst + Strategy agents")
        start_time = datetime.now()

        resume_from = self.setup()
        actual_start = max(start_day, resume_from + 1)

        if actual_start > end_day:
            logger.info("Simulation already complete!")
            return self._finalise()

        for day in range(actual_start, end_day + 1):
            try:
                day_decisions = self.run_day(day)
                self.all_decisions.extend(day_decisions)
                self.execution.save_checkpoint(day)

                elapsed = (datetime.now() - start_time).total_seconds()
                est_remaining = (elapsed / (day - actual_start + 1)) * (end_day - day)
                logger.info(
                    f"Progress: Day {day}/{end_day} | "
                    f"Elapsed: {elapsed:.0f}s | "
                    f"Est remaining: {est_remaining:.0f}s"
                )

            except Exception as e:
                logger.error(f"Day {day} failed: {e} — continuing to next day")
                self.execution.save_checkpoint(day)
                continue

        return self._finalise(start_time)

    def _finalise(self, start_time=None):
        """Generate and save all final results."""
        logger.info("=== FINALISING LLM SIMULATION ===")

        self.analyst.save_analyses()
        self.strategy.save_recommendations()

        summary = self.execution.get_summary()
        all_logged = self.execution.logger.decisions
        generator = ResultsGenerator(self.analyser)
        tables = generator.generate_chapter4_tables(self.daily_df, all_logged)
        generator.save_chapter4_tables(tables, "llm_chapter4_tables.json")
        self.analyser.save_results("llm_statistical_results.json")

        final = {
            "simulation": self.simulation_name,
            "total_decisions": summary["total_decisions"],
            "agreement_rate": summary["agreement_rate_pct"],
            "divergent_decisions": summary["divergent_decisions"],
            "total_tokens": self.total_tokens_used,
            "chapter4_tables": tables
        }

        results_path = RESULTS_PATH / "llm_simulation_final.json"
        with open(results_path, "w") as f:
            json.dump(final, f, indent=2)

        if start_time:
            duration = (datetime.now() - start_time).total_seconds()
            logger.success(
                f"LLM Simulation complete in {duration:.0f}s | "
                f"Decisions: {summary['total_decisions']} | "
                f"Agreement: {summary['agreement_rate_pct']}% | "
                f"Total tokens: {self.total_tokens_used}"
            )

        return final


if __name__ == "__main__":
    logger.info("Starting Full LLM-Powered Simulation...")
    logger.info("Estimated time: 30-40 minutes")
    logger.info("Estimated cost: $2-4")
    logger.info("Checkpointing enabled — safe to interrupt and resume")

    sim = LLMSimulationLoop(
        simulation_name="criteo_llm_v1",
        sample_size=500_000
    )

    # Run Days 0-29 (full 30-day simulation)
    results = sim.run(start_day=0, end_day=29)

    print("\n" + "="*60)
    print("LLM SIMULATION COMPLETE")
    print("="*60)
    print(f"Total decisions:    {results['total_decisions']}")
    print(f"Agreement rate:     {results['agreement_rate']}%")
    print(f"Divergent:          {results['divergent_decisions']}")
    print(f"Total tokens used:  {results['total_tokens']}")

    print("\n=== CHAPTER 4 OVERVIEW ===")
    overview = results["chapter4_tables"]["dataset_overview"]
    for key, val in overview.items():
        print(f"  {key}: {val}")

    print("\n=== AGREEMENT ANALYSIS ===")
    agreement = results["chapter4_tables"]["agreement_analysis"]
    print(f"  Total:          {agreement.get('total_decisions', 0)}")
    print(f"  Agreed:         {agreement.get('agreed', 0)}")
    print(f"  Diverged:       {agreement.get('diverged', 0)}")
    print(f"  Agreement rate: {agreement.get('agreement_rate_pct', 0)}%")

    print("\n✅ All results saved to results/")
    print("Next step: Write Chapter 4 and 5 using these results!")
