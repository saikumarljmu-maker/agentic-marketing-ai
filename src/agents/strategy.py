"""
Strategy Agent
Takes the Analyst Agent's analysis and generates
specific, actionable campaign management recommendations
using the ReAct framework with Claude.
"""

import json
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
import os
import anthropic

load_dotenv()

LOGS_PATH = Path("logs")


class StrategyAgent:
    """
    Generates specific campaign management recommendations
    based on the Analyst Agent's analysis.
    Uses ReAct: Reason -> Act -> Observe pattern.
    """

    def __init__(self):
        self.client = anthropic.Anthropic(
            default_headers={
                "anthropic-workspace-id": os.getenv("ANTHROPIC_WORKSPACE_ID")
            }
        )
        self.model = "claude-sonnet-4-6"
        self.recommendations = []
        logger.success("Strategy Agent initialised with Claude Sonnet 4.6")

    def _build_strategy_prompt(self, day, analysis, performance_summary, memory_context):
        """Build ReAct-style prompt for strategy generation."""

        memory_text = ""
        if memory_context:
            memory_text = "\n\nHISTORICAL DECISION CONTEXT:\n"
            for i, mem in enumerate(memory_context[:3]):
                memory_text += f"\n[Past Decision {i+1}]\n{mem['document']}\n"

        top_campaigns = performance_summary.get("top_performing_campaigns", [])
        bottom_campaigns = performance_summary.get("bottom_performing_campaigns", [])
        zero_conv = performance_summary.get("zero_conversion_campaigns_with_spend", [])
        account = performance_summary.get("account_totals", {})

        prompt = f"""You are an expert digital marketing campaign manager making decisions for Day {day}.

ANALYST FINDINGS:
{analysis}

ACCOUNT SUMMARY (Day {day}):
- Total Spend: {account.get('total_spend', 0):.6f}
- Total Conversions: {account.get('total_conversions', 0)}
- Overall ROAS: {account.get('overall_ctr', 0):.4f} CTR
- Overall CPA: {account.get('overall_cpa', 0):.6f}

TOP 5 CAMPAIGNS BY ROAS:
{json.dumps(top_campaigns, indent=2)}

BOTTOM 5 CAMPAIGNS BY ROAS:
{json.dumps(bottom_campaigns, indent=2)}

ZERO CONVERSION CAMPAIGNS WITH SPEND:
{json.dumps(zero_conv[:5], indent=2)}
{memory_text}

Using the ReAct framework, generate specific campaign management decisions:

REASON: Think through what the data and analysis tell you about what actions are needed.
What are the 3 most important issues to address? What is the risk of each action?

ACT: Generate exactly 5 specific recommendations in this JSON format:

{{
  "recommendations": [
    {{
      "priority": 1,
      "campaign_id": <campaign_id as integer>,
      "action": "<one of: increase_budget | decrease_budget | pause | expand_audience | narrow_audience | maintain>",
      "reason": "<specific data-driven reason referencing actual metrics>",
      "expected_kpi_impact": {{
        "metric": "<ROAS|CPA|CTR|Conversions>",
        "direction": "<increase|decrease>",
        "estimated_pct_change": <number>
      }},
      "confidence": "<high|medium|low>",
      "decision_type": "<BUDGET_INCREASE|BUDGET_DECREASE|PAUSE_RECOMMENDATION|AUDIENCE_EXPANSION|AUDIENCE_NARROWING|MAINTAIN>"
    }}
  ],
  "strategy_summary": "<2-3 sentence overall strategy for this day>",
  "key_risks": ["<risk 1>", "<risk 2>"],
  "human_review_required": <true|false>,
  "human_review_reason": "<reason if true, null if false>"
}}

OBSERVE: After generating recommendations, note what outcome metrics you would
monitor in the next 24-48 hours to validate these decisions.

Return ONLY the JSON object — no markdown, no explanation outside the JSON."""

        return prompt

    def recommend(self, day, analysis, performance_summary, memory_context=None):
        """
        Generate strategic recommendations for a simulation day.
        Returns structured recommendations with reasoning.
        """
        logger.info(f"Strategy Agent generating recommendations for Day {day}...")

        if memory_context is None:
            memory_context = []

        prompt = self._build_strategy_prompt(
            day, analysis, performance_summary, memory_context
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text.strip()

            # Clean JSON if needed
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])

            strategy = json.loads(response_text)

            result = {
                "day": day,
                "model": self.model,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "recommendations": strategy.get("recommendations", []),
                "strategy_summary": strategy.get("strategy_summary", ""),
                "key_risks": strategy.get("key_risks", []),
                "human_review_required": strategy.get("human_review_required", False),
                "human_review_reason": strategy.get("human_review_reason"),
                "n_recommendations": len(strategy.get("recommendations", []))
            }

            self.recommendations.append(result)

            logger.success(
                f"Day {day} strategy complete — "
                f"{result['n_recommendations']} recommendations generated | "
                f"Human review: {result['human_review_required']}"
            )

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed — returning raw text: {e}")
            return {
                "day": day,
                "error": "json_parse_failed",
                "raw_response": response.content[0].text,
                "recommendations": []
            }
        except Exception as e:
            logger.error(f"Strategy Agent failed on Day {day}: {e}")
            return {
                "day": day,
                "error": str(e),
                "recommendations": []
            }

    def get_decisions_for_execution(self, strategy_result):
        """
        Convert strategy recommendations into execution-ready
        decisions for the Execution Agent.
        """
        decisions = []
        for rec in strategy_result.get("recommendations", []):
            decisions.append({
                "campaign_id": rec.get("campaign_id"),
                "action": rec.get("action"),
                "reason": rec.get("reason"),
                "decision_type": rec.get("decision_type"),
                "confidence": rec.get("confidence"),
                "expected_kpi_impact": rec.get("expected_kpi_impact"),
                "priority": rec.get("priority"),
                "human_review_required": strategy_result.get("human_review_required", False)
            })
        return decisions

    def save_recommendations(self, path=None):
        """Save all recommendations to disk."""
        if path is None:
            path = LOGS_PATH / "strategy_outputs.jsonl"
        LOGS_PATH.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for r in self.recommendations:
                f.write(json.dumps(r) + "\n")
        logger.success(f"Saved {len(self.recommendations)} strategies to {path}")


if __name__ == "__main__":
    logger.info("Testing Strategy Agent...")

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from agents.observer import ObserverAgent
    from agents.analyst import AnalystAgent

    logger.info("Step 1 — Loading data...")
    observer = ObserverAgent(sample_size=100_000)
    observer.initialise()
    day_result = observer.observe(day=10)
    performance_summary = day_result["performance_summary"]
    anomalies = day_result["anomalies"][:5]

    logger.info("Step 2 — Running Analyst Agent...")
    analyst = AnalystAgent()
    analysis_result = analyst.analyse(
        day=10,
        performance_summary=performance_summary,
        anomalies=anomalies,
        memory_context=[]
    )

    logger.info("Step 3 — Running Strategy Agent...")
    strategy = StrategyAgent()
    strategy_result = strategy.recommend(
        day=10,
        analysis=analysis_result.get("analysis", ""),
        performance_summary=performance_summary,
        memory_context=[]
    )

    print("\n" + "="*60)
    print(f"DAY 10 STRATEGY RECOMMENDATIONS")
    print("="*60)

    if strategy_result.get("strategy_summary"):
        print(f"\nSTRATEGY SUMMARY:")
        print(strategy_result["strategy_summary"])

    print(f"\nRECOMMENDATIONS ({strategy_result.get('n_recommendations', 0)} total):")
    for rec in strategy_result.get("recommendations", []):
        print(f"\n  Priority {rec.get('priority')}: {rec.get('action').upper()}")
        print(f"  Campaign:  {rec.get('campaign_id')}")
        print(f"  Reason:    {rec.get('reason')}")
        print(f"  Expected:  {rec.get('expected_kpi_impact', {}).get('metric')} "
              f"{rec.get('expected_kpi_impact', {}).get('direction')} "
              f"{rec.get('expected_kpi_impact', {}).get('estimated_pct_change')}%")
        print(f"  Confidence: {rec.get('confidence')}")

    if strategy_result.get("key_risks"):
        print(f"\nKEY RISKS:")
        for risk in strategy_result["key_risks"]:
            print(f"  - {risk}")

    print(f"\nHuman review required: {strategy_result.get('human_review_required')}")
    if strategy_result.get("human_review_reason"):
        print(f"Reason: {strategy_result.get('human_review_reason')}")

    decisions = strategy.get_decisions_for_execution(strategy_result)
    print(f"\nExecution-ready decisions: {len(decisions)}")

    print(f"\nTokens used: "
          f"{strategy_result.get('input_tokens', 0)} in / "
          f"{strategy_result.get('output_tokens', 0)} out")

    strategy.save_recommendations()
    print("\n✅ Strategy Agent working correctly!")
