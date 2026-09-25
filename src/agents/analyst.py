"""
Analyst Agent
Uses Claude to interpret campaign performance data
and generate natural language strategic analysis.
"""

import json
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
import os
import anthropic

load_dotenv()

LOGS_PATH = Path("logs")


class AnalystAgent:
    """
    Reads the Observer's performance summary and uses
    Claude to generate strategic campaign analysis
    with chain-of-thought reasoning.
    """

    def __init__(self):
        self.client = anthropic.Anthropic(
            default_headers={
                "anthropic-workspace-id": os.getenv("ANTHROPIC_WORKSPACE_ID")
            }
        )
        self.model = "claude-sonnet-4-6"
        self.analyses = []
        logger.success("Analyst Agent initialised with Claude Sonnet 4.6")

    def _build_prompt(self, performance_summary, anomalies, memory_context):
        """Build the chain-of-thought prompt for Claude."""

        memory_text = ""
        if memory_context:
            memory_text = "\n\nRELEVANT HISTORICAL CONTEXT:\n"
            for i, mem in enumerate(memory_context[:3]):
                memory_text += f"\n[Memory {i+1}]\n{mem['document']}\n"

        anomaly_text = ""
        if anomalies:
            anomaly_text = f"\n\nANOMALIES DETECTED ({len(anomalies)} total):\n"
            for a in anomalies[:5]:
                anomaly_text += (
                    f"- Campaign {a['campaign']}: {a['type']} "
                    f"({a['severity']}) — {a['detail']}\n"
                )

        prompt = f"""You are an expert digital marketing campaign analyst.
Analyse the following campaign performance data and provide strategic insights.

PERFORMANCE SUMMARY:
{json.dumps(performance_summary, indent=2)}
{anomaly_text}
{memory_text}

Think step by step:

STEP 1 — DATA DESCRIPTION:
Describe what the data shows. What are the key metrics? What is the overall account performance?

STEP 2 — TREND ANALYSIS:
What trends are visible? Are performance metrics improving, declining, or stable?
Which campaigns are performing well and which are struggling?

STEP 3 — ROOT CAUSE ANALYSIS:
For any underperforming campaigns or anomalies, what are the most likely explanations?
Consider spend levels, conversion rates, CTR patterns.

STEP 4 — STRATEGIC OPPORTUNITIES:
What specific opportunities exist to improve performance?
Which campaigns should be scaled? Which need intervention?

STEP 5 — RECOMMENDED ACTIONS:
List 3-5 specific, actionable recommendations in priority order.
For each recommendation explain the expected KPI impact.

Be specific and data-driven. Reference actual numbers from the data."""

        return prompt

    def analyse(self, day, performance_summary, anomalies, memory_context=None):
        """
        Generate strategic analysis for a simulation day.
        Returns structured analysis with reasoning chain.
        """
        logger.info(f"Analyst Agent analysing Day {day}...")

        if memory_context is None:
            memory_context = []

        prompt = self._build_prompt(
            performance_summary, anomalies, memory_context
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )

            analysis_text = response.content[0].text

            analysis = {
                "day": day,
                "model": self.model,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "analysis": analysis_text,
                "account_totals": performance_summary.get("account_totals", {}),
                "anomaly_count": len(anomalies),
                "memory_context_used": len(memory_context),
                "top_campaigns_analysed": len(
                    performance_summary.get("top_performing_campaigns", [])
                )
            }

            self.analyses.append(analysis)

            logger.success(
                f"Day {day} analysis complete — "
                f"{response.usage.output_tokens} tokens generated"
            )

            return analysis

        except Exception as e:
            logger.error(f"Analyst Agent failed on Day {day}: {e}")
            return {
                "day": day,
                "error": str(e),
                "analysis": None
            }

    def save_analyses(self, path=None):
        """Save all analyses to disk."""
        if path is None:
            path = LOGS_PATH / "analyst_outputs.jsonl"
        LOGS_PATH.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for a in self.analyses:
                f.write(json.dumps(a) + "\n")
        logger.success(f"Saved {len(self.analyses)} analyses to {path}")


if __name__ == "__main__":
    logger.info("Testing Analyst Agent...")

    from pathlib import Path
    import pandas as pd
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from agents.observer import ObserverAgent

    observer = ObserverAgent(sample_size=100_000)
    init_result = observer.initialise()

    day_result = observer.observe(day=7)
    performance_summary = day_result["performance_summary"]
    anomalies = day_result["anomalies"][:5]

    analyst = AnalystAgent()

    analysis = analyst.analyse(
        day=7,
        performance_summary=performance_summary,
        anomalies=anomalies,
        memory_context=[]
    )

    if analysis.get("analysis"):
        print("\n" + "="*60)
        print(f"DAY 7 CAMPAIGN ANALYSIS")
        print("="*60)
        print(analysis["analysis"])
        print("\n" + "="*60)
        print(f"Tokens used: {analysis['input_tokens']} in / {analysis['output_tokens']} out")
        print(f"Anomalies flagged: {analysis['anomaly_count']}")
        analyst.save_analyses()
        print("\n✅ Analyst Agent working correctly!")
    else:
        print(f"Error: {analysis.get('error')}")
