"""
Simulation Loop
Ties all agents together into a complete 30-day
retrospective simulation pipeline.
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
from agents.memory import MemoryLayer
from agents.execution import ExecutionAgent
from evaluation.response_curves import ResponseCurveEstimator
from evaluation.statistical_analysis import StatisticalAnalyser, ResultsGenerator

RESULTS_PATH = Path("results")
LOGS_PATH = Path("logs")


class SimulationLoop:
    def __init__(self, simulation_name="criteo_simulation", sample_size=500_000):
        self.simulation_name = simulation_name
        self.sample_size = sample_size
        self.observer = None
        self.memory = None
        self.execution = None
        self.curve_estimator = None
        self.analyser = StatisticalAnalyser()
        self.daily_df = None
        self.all_decisions = []
        logger.info(f"Simulation Loop initialised: {simulation_name}")

    def setup(self):
        logger.info("=== SIMULATION SETUP ===")
        self.observer = ObserverAgent(sample_size=self.sample_size)
        init_result = self.observer.initialise()
        if init_result.get("status") != "ready":
            raise RuntimeError("Observer Agent failed to initialise")
        self.daily_df = self.observer.daily_df
        logger.success(f"Data loaded: {len(self.daily_df):,} daily records")
        self.memory = MemoryLayer(collection_name=f"{self.simulation_name}_memory")
        self.execution = ExecutionAgent(simulation_name=self.simulation_name)
        self.curve_estimator = ResponseCurveEstimator()
        checkpoint = self.execution.load_checkpoint()
        if checkpoint:
            logger.info(f"Resuming from Day {checkpoint['last_completed_day'] + 1}")
            return checkpoint.get("last_completed_day", -1)
        return -1

    def _generate_rule_based_decision(self, campaign_id, metrics, memory_context):
        roas = metrics.get("avg_roas", 0)
        cpa = metrics.get("avg_cpa", 0)
        spend = metrics.get("spend", 0)
        conversions = metrics.get("conversions", 0)
        ctr = metrics.get("avg_ctr", 0)

        if spend == 0:
            return {"action": "maintain", "reason": "No spend detected",
                    "confidence": "low", "decision_type": "MAINTAIN"}

        if conversions == 0 and spend > 0:
            return {"action": "pause",
                    "reason": f"Zero conversions with spend {spend:.6f}",
                    "confidence": "high", "decision_type": "PAUSE_RECOMMENDATION"}

        if roas > 2.0:
            return {"action": "increase_budget",
                    "amount": round(spend * 0.20, 6),
                    "reason": f"ROAS {roas:.2f} above 2.0 threshold — scale up",
                    "confidence": "high", "decision_type": "BUDGET_INCREASE"}

        if 0 < roas < 0.5 and spend > 0.001:
            return {"action": "decrease_budget",
                    "amount": round(spend * 0.30, 6),
                    "reason": f"ROAS {roas:.2f} below 0.5 — reduce spend",
                    "confidence": "high", "decision_type": "BUDGET_DECREASE"}

        if ctr < 0.01 and spend > 0.001:
            return {"action": "expand_audience",
                    "reason": f"CTR {ctr:.4f} very low — broaden targeting",
                    "confidence": "medium", "decision_type": "AUDIENCE_EXPANSION"}

        return {"action": "maintain",
                "reason": f"Performance acceptable — ROAS {roas:.2f}",
                "confidence": "medium", "decision_type": "MAINTAIN"}

    def _infer_human_decision(self, campaign_id, day, metrics):
        if day == 0:
            return {"action": "maintain", "reason": "First day — baseline"}

        prev_day = self.daily_df[
            (self.daily_df["campaign"] == campaign_id) &
            (self.daily_df["day"] == day - 1)
        ]

        if prev_day.empty:
            return {"action": "maintain", "reason": "No previous day data"}

        prev_spend = float(prev_day["total_spend"].iloc[0])
        curr_spend = metrics.get("spend", 0)

        if prev_spend > 0:
            spend_change = (curr_spend - prev_spend) / prev_spend
            if spend_change > 0.15:
                return {"action": "increase_budget",
                        "amount": round(curr_spend - prev_spend, 6),
                        "reason": f"Human increased spend {spend_change*100:.1f}%"}
            elif spend_change < -0.15:
                return {"action": "decrease_budget",
                        "amount": round(prev_spend - curr_spend, 6),
                        "reason": f"Human decreased spend {abs(spend_change)*100:.1f}%"}
            elif curr_spend == 0 and prev_spend > 0:
                return {"action": "pause", "reason": "Human stopped spending"}

        return {"action": "maintain", "reason": f"Spend stable at {curr_spend:.6f}"}

    def run_day(self, day):
        logger.info(f"--- Day {day} ---")
        observation = self.observer.observe(day=day)
        anomalies = observation["anomalies"]
        day_data = self.daily_df[self.daily_df["day"] == day]

        if day_data.empty:
            logger.warning(f"No data for Day {day} — skipping")
            return []

        campaign_summary = day_data.groupby("campaign").agg(
            spend=("total_spend", "sum"),
            conversions=("conversions", "sum"),
            avg_roas=("roas", "mean"),
            avg_cpa=("cpa", "mean"),
            avg_ctr=("ctr", "mean")
        ).reset_index()

        active_campaigns = campaign_summary[campaign_summary["spend"] > 0]
        day_decisions = []
        self.curve_estimator.fit_all(self.daily_df, up_to_day=day)

        for _, row in active_campaigns.iterrows():
            campaign_id = int(row["campaign"])
            metrics = row.to_dict()

            memory_context = self.memory.retrieve(
                query=f"campaign {campaign_id} ROAS {row['avg_roas']:.2f} spend {row['spend']:.6f}",
                n_results=3,
                campaign=campaign_id
            )

            ai_decision = self._generate_rule_based_decision(
                campaign_id, metrics, memory_context
            )
            human_decision = self._infer_human_decision(campaign_id, day, metrics)

            context = {
                "day": day,
                "campaign": campaign_id,
                "spend": round(float(row["spend"]), 6),
                "conversions": int(row["conversions"]),
                "roas": round(float(row["avg_roas"]), 4),
                "cpa": round(float(row["avg_cpa"]), 6),
                "ctr": round(float(row["avg_ctr"]), 6),
                "anomalies_detected": sum(
                    1 for a in anomalies if a["campaign"] == campaign_id
                )
            }

            decision_id = self.execution.execute(
                day=day,
                campaign=campaign_id,
                ai_decision=ai_decision,
                human_decision=human_decision,
                context=context
            )

            if not self.execution.logger.decisions[-1]["agreed"]:
                current_spend = float(row["spend"])
                ai_action = ai_decision.get("action", "")
                if ai_action == "increase_budget":
                    proposed_spend = current_spend * 1.20
                elif ai_action == "decrease_budget":
                    proposed_spend = current_spend * 0.70
                elif ai_action == "pause":
                    proposed_spend = 0.0
                else:
                    proposed_spend = current_spend

                if proposed_spend != current_spend:
                    projection = self.curve_estimator.project_kpi_delta(
                        campaign_id=campaign_id,
                        current_spend=current_spend,
                        proposed_spend=proposed_spend,
                        current_conversions=float(row["conversions"])
                    )
                    self.execution.add_counterfactual(decision_id, projection)

            self.memory.store(
                day=day,
                campaign=campaign_id,
                decision_type=ai_decision.get("decision_type", "UNKNOWN"),
                decision=ai_decision,
                context=context
            )

            day_decisions.append({
                "id": decision_id,
                "day": day,
                "campaign": campaign_id,
                "ai_action": ai_decision.get("action"),
                "human_action": human_decision.get("action"),
                "agreed": self.execution.logger.decisions[-1]["agreed"]
            })

        logger.info(
            f"Day {day} complete — {len(day_decisions)} campaigns | "
            f"Agreement: {self.execution.get_agreement_rate():.1f}%"
        )
        return day_decisions

    def run(self, start_day=0, end_day=29):
        logger.info("=== STARTING FULL SIMULATION ===")
        start_time = datetime.now()
        resume_from = self.setup()
        actual_start = max(start_day, resume_from + 1)

        if actual_start > end_day:
            logger.info("Simulation already complete!")
            return self._finalise()

        for day in range(actual_start, end_day + 1):
            day_decisions = self.run_day(day)
            self.all_decisions.extend(day_decisions)
            self.execution.save_checkpoint(day)

            if day % 5 == 0 or day == end_day:
                logger.info(
                    f"Progress: Day {day}/{end_day} | "
                    f"Total: {len(self.execution.logger.decisions)} decisions | "
                    f"Agreement: {self.execution.get_agreement_rate():.1f}%"
                )

        return self._finalise(start_time)

    def _finalise(self, start_time=None):
        logger.info("=== FINALISING RESULTS ===")
        summary = self.execution.get_summary()
        all_logged = self.execution.logger.decisions
        generator = ResultsGenerator(self.analyser)
        tables = generator.generate_chapter4_tables(self.daily_df, all_logged)
        generator.save_chapter4_tables(tables)
        self.analyser.save_results("final_statistical_results.json")

        if start_time:
            duration = (datetime.now() - start_time).total_seconds()
            logger.success(
                f"Done in {duration:.1f}s | "
                f"Decisions: {summary['total_decisions']} | "
                f"Agreement: {summary['agreement_rate_pct']}%"
            )

        return {
            "summary": summary,
            "chapter4_tables": tables,
            "total_decisions": summary["total_decisions"],
            "agreement_rate": summary["agreement_rate_pct"],
            "divergent_decisions": summary["divergent_decisions"]
        }


if __name__ == "__main__":
    logger.info("Starting Criteo Retrospective Simulation...")
    logger.info("NOTE: Rule-based decisions now — LLM replaces this when API ready")

    sim = SimulationLoop(
        simulation_name="criteo_sim_v1",
        sample_size=500_000
    )

    results = sim.run(start_day=0, end_day=29)

    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print(f"Total decisions:    {results['total_decisions']}")
    print(f"Agreement rate:     {results['agreement_rate']}%")
    print(f"Divergent:          {results['divergent_decisions']}")

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
