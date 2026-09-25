"""
Execution Agent
Records AI recommendations, maps them against human decisions
in the dataset, flags divergences for counterfactual projection.
"""

import json
import csv
from pathlib import Path
from loguru import logger
from datetime import datetime

LOGS_PATH = Path("logs")
RESULTS_PATH = Path("results")


class DecisionLogger:
    """
    Logs every AI decision alongside the corresponding
    human decision from the dataset.
    """

    def __init__(self, simulation_name="criteo_simulation"):
        self.simulation_name = simulation_name
        self.log_path = LOGS_PATH / f"{simulation_name}_decisions.jsonl"
        self.summary_path = RESULTS_PATH / f"{simulation_name}_summary.json"
        self.decisions = []
        LOGS_PATH.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        logger.success(f"Decision Logger initialised — logging to {self.log_path}")

    def log(self, day, campaign, ai_decision, human_decision,
            context, analyst_output=None):
        """
        Log one AI decision vs one human decision.
        Automatically detects if they agree or diverge.
        """
        agreed = self._decisions_agree(ai_decision, human_decision)

        record = {
            "id": f"{self.simulation_name}_day{day}_camp{campaign}_{datetime.now().strftime('%H%M%S%f')}",
            "simulation": self.simulation_name,
            "day": day,
            "campaign": campaign,
            "timestamp": datetime.now().isoformat(),
            "ai_decision": ai_decision,
            "human_decision": human_decision,
            "agreed": agreed,
            "requires_counterfactual": not agreed,
            "context": context,
            "analyst_output": analyst_output,
            "counterfactual_projection": None,
            "evaluation": {
                "level_4_score": None,
                "kpi_delta": None,
                "confidence_interval": None
            }
        }

        self.decisions.append(record)

        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        status = "AGREE" if agreed else "DIVERGE"
        logger.info(f"Day {day} | Campaign {campaign} | {status} | "
                    f"AI: {ai_decision.get('action', 'N/A')} | "
                    f"Human: {human_decision.get('action', 'N/A')}")

        return record["id"]

    def _decisions_agree(self, ai_decision, human_decision):
        """
        Check if AI and human decisions broadly agree.
        Compares the action type rather than exact parameters.
        """
        ai_action = ai_decision.get("action", "").lower()
        human_action = human_decision.get("action", "").lower()

        agree_map = {
            "increase_budget": ["increase_budget", "scale_up"],
            "decrease_budget": ["decrease_budget", "scale_down", "reduce_spend"],
            "pause": ["pause", "stop", "disable"],
            "maintain": ["maintain", "keep", "no_change"],
            "expand_audience": ["expand_audience", "broaden_targeting"],
            "narrow_audience": ["narrow_audience", "tighten_targeting"],
        }

        for canonical, variants in agree_map.items():
            if ai_action in variants and human_action in variants:
                return True

        return ai_action == human_action

    def update_counterfactual(self, decision_id, projection):
        """Add counterfactual projection to a logged decision."""
        for record in self.decisions:
            if record["id"] == decision_id:
                record["counterfactual_projection"] = projection
                record["evaluation"]["kpi_delta"] = projection.get("kpi_delta")
                record["evaluation"]["confidence_interval"] = projection.get("ci_95")
                logger.info(f"Updated counterfactual for {decision_id[:20]}...")
                return
        logger.warning(f"Decision {decision_id[:20]} not found for update")

    def get_divergent_decisions(self):
        """Return all decisions where AI and human disagreed."""
        return [d for d in self.decisions if not d["agreed"]]

    def get_agreement_rate(self):
        """Calculate what percentage of decisions AI and human agreed on."""
        if not self.decisions:
            return 0.0
        agreed = sum(1 for d in self.decisions if d["agreed"])
        return round(agreed / len(self.decisions) * 100, 2)

    def save_summary(self):
        """Save a summary of all decisions to results folder."""
        if not self.decisions:
            logger.warning("No decisions to summarise")
            return

        divergent = self.get_divergent_decisions()
        agreement_rate = self.get_agreement_rate()

        decision_type_counts = {}
        for d in self.decisions:
            dt = d["ai_decision"].get("action", "unknown")
            decision_type_counts[dt] = decision_type_counts.get(dt, 0) + 1

        summary = {
            "simulation": self.simulation_name,
            "generated_at": datetime.now().isoformat(),
            "total_decisions": len(self.decisions),
            "agreed_decisions": len(self.decisions) - len(divergent),
            "divergent_decisions": len(divergent),
            "agreement_rate_pct": agreement_rate,
            "decision_type_breakdown": decision_type_counts,
            "days_covered": sorted(list(set(d["day"] for d in self.decisions))),
            "campaigns_covered": len(set(d["campaign"] for d in self.decisions)),
            "decisions_with_counterfactual": sum(
                1 for d in self.decisions
                if d["counterfactual_projection"] is not None
            )
        }

        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.success(f"Summary saved to {self.summary_path}")
        return summary

    def load_existing(self):
        """Load existing decision log — useful for resuming simulation."""
        if not self.log_path.exists():
            logger.info("No existing log found — starting fresh")
            return []

        loaded = []
        with open(self.log_path, "r") as f:
            for line in f:
                if line.strip():
                    loaded.append(json.loads(line))

        self.decisions = loaded
        logger.success(f"Loaded {len(loaded)} existing decisions from log")
        return loaded


class ExecutionAgent:
    """
    The Execution Agent — records AI vs human decisions,
    tracks divergences, and prepares data for counterfactual analysis.
    In a live system this would also execute recommendations via APIs.
    """

    def __init__(self, simulation_name="criteo_simulation"):
        self.logger = DecisionLogger(simulation_name)
        self.simulation_name = simulation_name
        self.checkpoint_path = LOGS_PATH / f"{simulation_name}_checkpoint.json"

    def execute(self, day, campaign, ai_decision,
                human_decision, context, analyst_output=None):
        """
        Record an AI decision vs the human baseline decision.
        Returns the decision ID for later counterfactual updates.
        """
        decision_id = self.logger.log(
            day=day,
            campaign=campaign,
            ai_decision=ai_decision,
            human_decision=human_decision,
            context=context,
            analyst_output=analyst_output
        )
        return decision_id

    def add_counterfactual(self, decision_id, projection):
        """Add a counterfactual projection to a recorded decision."""
        self.logger.update_counterfactual(decision_id, projection)

    def save_checkpoint(self, day):
        """Save simulation checkpoint so it can be resumed."""
        checkpoint = {
            "simulation": self.simulation_name,
            "last_completed_day": day,
            "total_decisions": len(self.logger.decisions),
            "agreement_rate": self.logger.get_agreement_rate(),
            "saved_at": datetime.now().isoformat()
        }
        with open(self.checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)
        logger.info(f"Checkpoint saved — Day {day} complete")

    def load_checkpoint(self):
        """Load existing checkpoint to resume simulation."""
        if not self.checkpoint_path.exists():
            return None
        with open(self.checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        logger.info(f"Checkpoint found — resuming from Day {checkpoint['last_completed_day']}")
        return checkpoint

    def get_summary(self):
        """Get and save full simulation summary."""
        return self.logger.save_summary()

    def get_divergent_decisions(self):
        """Get all decisions requiring counterfactual projection."""
        return self.logger.get_divergent_decisions()

    def get_agreement_rate(self):
        return self.logger.get_agreement_rate()


if __name__ == "__main__":
    logger.info("Testing Execution Agent...")

    agent = ExecutionAgent(simulation_name="test_simulation")

    id1 = agent.execute(
        day=5,
        campaign=12345,
        ai_decision={
            "action": "increase_budget",
            "amount": 0.002,
            "reason": "ROAS above 2.0 for 3 days",
            "expected_kpi_impact": {"roas": "+15%", "conversions": "+20%"}
        },
        human_decision={
            "action": "maintain",
            "reason": "No change made by human manager"
        },
        context={
            "roas": 2.5, "cpa": 0.003,
            "spend": 0.01, "conversions": 3
        }
    )

    id2 = agent.execute(
        day=6,
        campaign=67890,
        ai_decision={
            "action": "pause",
            "reason": "Zero conversions for 3 days"
        },
        human_decision={
            "action": "pause",
            "reason": "Human also paused this campaign"
        },
        context={
            "roas": 0.0, "cpa": 0.0,
            "spend": 0.005, "conversions": 0
        }
    )

    id3 = agent.execute(
        day=7,
        campaign=11111,
        ai_decision={
            "action": "decrease_budget",
            "amount": 0.003,
            "reason": "CPA 3x above target"
        },
        human_decision={
            "action": "decrease_budget",
            "reason": "Human reduced budget"
        },
        context={
            "roas": 0.5, "cpa": 0.02,
            "spend": 0.02, "conversions": 1
        }
    )

    print(f"\nTotal decisions logged: {len(agent.logger.decisions)}")
    print(f"Agreement rate: {agent.get_agreement_rate()}%")

    divergent = agent.get_divergent_decisions()
    print(f"Divergent decisions: {len(divergent)}")
    if divergent:
        print(f"  First divergence — Day {divergent[0]['day']}, "
              f"Campaign {divergent[0]['campaign']}")
        print(f"  AI said: {divergent[0]['ai_decision']['action']}")
        print(f"  Human said: {divergent[0]['human_decision']['action']}")

    agent.add_counterfactual(id1, {
        "kpi_delta": {"roas": +0.3, "conversions": +1},
        "ci_95": {"roas": [+0.1, +0.5], "conversions": [0, +2]},
        "projection_method": "log_linear_response_curve",
        "ai_projected_better": True
    })
    print(f"\nCounterfactual added to decision {id1[:8]}...")

    agent.save_checkpoint(day=7)

    summary = agent.get_summary()
    print(f"\n=== SIMULATION SUMMARY ===")
    print(f"Total decisions: {summary['total_decisions']}")
    print(f"Agreed: {summary['agreed_decisions']}")
    print(f"Diverged: {summary['divergent_decisions']}")
    print(f"Agreement rate: {summary['agreement_rate_pct']}%")
    print(f"Campaigns covered: {summary['campaigns_covered']}")

    print("\n✅ Execution Agent working correctly!")
