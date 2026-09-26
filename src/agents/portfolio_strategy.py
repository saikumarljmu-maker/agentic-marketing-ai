"""
Portfolio Strategy Agent — LLM as Portfolio Manager
The LLM sets strategy parameters; a numeric allocator does the math.
This gives a clean ablation: LLM-only vs Bandit-only vs LLM+Bandit hybrid.
"""

import json
import time
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
import os
import anthropic
import pandas as pd
import numpy as np

load_dotenv()

LOGS_PATH = Path("logs")
LOGS_PATH.mkdir(exist_ok=True)


class PortfolioStrategyAgent:
    """
    LLM reads compact portfolio view and sets allocator parameters.
    Numeric allocator then distributes budget accordingly.
    """

    def __init__(self):
        self.client = anthropic.Anthropic(
            default_headers={
                "anthropic-workspace-id": os.getenv("ANTHROPIC_WORKSPACE_ID")
            }
        )
        self.model = "claude-sonnet-4-6"
        self.decisions = []
        logger.success("Portfolio Strategy Agent initialised")

    def _build_portfolio_prompt(self, day, view_df, budget, lag_note):
        """
        Build compact portfolio view prompt for the LLM.
        Shows uncertainty ranges, not just averages.
        """
        # Sort by conversions descending
        v = view_df.copy().sort_values("known_conversions", ascending=False)

        # Build compact table
        rows = []
        for _, row in v.head(20).iterrows():
            conv = int(row["known_conversions"])
            cost = round(float(row["cost"]), 4)
            cpa = round(float(row["cpa"]), 4) if row["cpa"] > 0 else "N/A"
            ctr = round(float(row.get("ctr", 0)), 4)

            # Thompson uncertainty range
            if conv > 0 and cost > 0:
                rate = conv / cost
                ci_low = round(float(np.random.gamma(conv, 1/cost) * 0.5), 4)
                ci_high = round(float(np.random.gamma(conv + 1, 1/cost) * 1.5), 4)
                uncertainty = f"[{ci_low}-{ci_high}]"
            else:
                uncertainty = "[unknown]"

            rows.append(
                f"Camp {int(row['campaign'])}: "
                f"spend={cost}, conv={conv}, "
                f"CPA={cpa}, CTR={ctr}, "
                f"CI={uncertainty}"
            )

        portfolio_text = "\n".join(rows)

        # Stats
        total_spend = round(float(v["cost"].sum()), 4)
        total_conv = int(v["known_conversions"].sum())
        zero_conv = int((v["known_conversions"] == 0).sum())
        active = int((v["cost"] > 0).sum())

        prompt = f"""You are a digital marketing portfolio manager for Day {day}.

PORTFOLIO SUMMARY:
- Total campaigns: {len(v)} ({active} active, {zero_conv} with zero conversions)
- Total spend last 7 days: {total_spend}
- Total known conversions: {total_conv}
- Available budget today: {round(budget, 4)}
- {lag_note}

TOP 20 CAMPAIGNS (by conversions):
campaign: spend, conversions, CPA, CTR, uncertainty_range
{portfolio_text}

Your job is to set STRATEGY PARAMETERS for the budget allocator.
The allocator will distribute the budget mathematically based on your parameters.

Respond ONLY with this JSON (no markdown, no extra text):
{{
  "strategy": "aggressive" | "conservative" | "balanced",
  "protect_campaigns": [list of INTEGER campaign IDs only, e.g. 9100693 not "Camp 9100693"],
  "exclude_campaigns": [list of INTEGER campaign IDs only, e.g. 442617 not "Camp 442617"],
  "focus_on_conversions": true | false,
  "escalate_to_human": true | false,
  "escalation_reason": "reason or null",
  "confidence": "high" | "medium" | "low",
  "reasoning": "one sentence explaining overall strategy",
  "lag_acknowledged": true | false
}}"""

        return prompt

    def get_strategy(self, day, view_df, budget, lag_note=""):
        """Get LLM strategy parameters for the portfolio allocator."""
        logger.info(f"Portfolio Strategy Agent — Day {day}")

        prompt = self._build_portfolio_prompt(day, view_df, budget, lag_note)

        try:
            start = time.time()
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            duration = round(time.time() - start, 2)

            text = response.content[0].text.strip()

            # Clean JSON
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]

            strategy = json.loads(text)
            strategy["day"] = day
            strategy["duration_seconds"] = duration
            strategy["input_tokens"] = response.usage.input_tokens
            strategy["output_tokens"] = response.usage.output_tokens
            strategy["cost_usd"] = round(
                (response.usage.input_tokens * 3 +
                 response.usage.output_tokens * 15) / 1_000_000, 6
            )

            self.decisions.append(strategy)

            logger.success(
                f"Day {day}: strategy={strategy.get('strategy')} | "
                f"protect={len(strategy.get('protect_campaigns', []))} | "
                f"exclude={len(strategy.get('exclude_campaigns', []))} | "
                f"escalate={strategy.get('escalate_to_human')} | "
                f"{duration}s | ${strategy['cost_usd']}"
            )

            return strategy

        except json.JSONDecodeError as e:
            logger.warning(f"Day {day} JSON parse failed: {e}")
            return self._fallback_strategy(day)
        except Exception as e:
            logger.error(f"Day {day} strategy failed: {e}")
            return self._fallback_strategy(day)

    def _fallback_strategy(self, day):
        """Conservative fallback if LLM fails."""
        logger.warning(f"Day {day} using fallback strategy")
        return {
            "day": day,
            "strategy": "balanced",
            "protect_campaigns": [],
            "exclude_campaigns": [],
            "focus_on_conversions": True,
            "escalate_to_human": False,
            "escalation_reason": None,
            "confidence": "low",
            "reasoning": "Fallback: LLM unavailable, using balanced allocation",
            "lag_acknowledged": False,
            "fallback": True,
            "cost_usd": 0.0
        }

    def save(self, path=None):
        if path is None:
            path = LOGS_PATH / "portfolio_strategy_outputs.jsonl"
        with open(path, "w") as f:
            for d in self.decisions:
                f.write(json.dumps(d) + "\n")
        logger.success(f"Saved {len(self.decisions)} strategies to {path}")


class PortfolioAllocator:
    """
    Numeric budget allocator that implements LLM strategy parameters.
    Uses Thompson Sampling as the base allocation mechanism.
    """

    STRATEGY_MULTIPLIERS = {
        "aggressive": 1.3,
        "balanced": 1.0,
        "conservative": 0.7
    }

    def allocate(self, view_df, budget, strategy_params, seed=42):
        """
        Distribute budget across campaigns based on LLM strategy parameters.
        Returns: pd.Series of campaign -> spend allocation
        """
        rng = np.random.default_rng(seed)
        v = view_df.set_index("campaign").copy()

        def parse_camp_id(x):
            s = str(x).strip().replace('Camp ', '').replace('camp ', '')
            try:
                return int(s)
            except:
                return None
        exclude = set(v for v in (parse_camp_id(x) for x in strategy_params.get("exclude_campaigns", [])) if v is not None)
        protect = set(v for v in (parse_camp_id(x) for x in strategy_params.get("protect_campaigns", [])) if v is not None)
        strat = strategy_params.get("strategy", "balanced")
        mult = self.STRATEGY_MULTIPLIERS.get(strat, 1.0)

        # Thompson Sampling scores
        pooled_rate = (
            v["known_conversions"].sum() /
            max(v["cost"].sum(), 1e-12)
        )
        prior_cost = 1.0 / max(pooled_rate, 1e-12)

        a = 1.0 + v["known_conversions"]
        b = prior_cost + v["cost"]
        scores = pd.Series(
            rng.gamma(a.values, 1.0 / b.values),
            index=v.index
        )

        # Apply LLM strategy
        scores[list(exclude)] = -np.inf  # Exclude completely
        scores[list(protect)] = scores.max() * 2  # Prioritise protected

        # Apply strategy multiplier to conversion-positive campaigns
        if strategy_params.get("focus_on_conversions", True):
            converting = v[v["known_conversions"] > 0].index
            scores[converting] *= mult

        # Greedy allocation
        alloc = pd.Series(0.0, index=v.index)
        remaining = budget
        capacity = v["last_day_cost"].copy()
        capacity[list(exclude)] = 0.0

        for camp in scores.sort_values(ascending=False).index:
            if remaining <= 0:
                break
            cap = float(capacity.get(camp, 0))
            if cap <= 0:
                continue
            take = min(cap, remaining)
            alloc[camp] = take
            remaining -= take

        return alloc

    def get_baseline_allocations(self, view_df, budget, seed=42):
        """
        Generate all baseline policy allocations for comparison.
        Returns dict of policy_name -> pd.Series allocation
        """
        rng = np.random.default_rng(seed)
        v = view_df.set_index("campaign").copy()
        cap = v["last_day_cost"]

        allocations = {}

        # 1. Logged mix (status quo)
        w = cap.clip(lower=0)
        allocations["logged_mix"] = w / w.sum() * budget if w.sum() > 0 else w

        # 2. Uniform
        uniform_scores = pd.Series(rng.random(len(cap)), index=cap.index)
        allocations["uniform"] = self._greedy(uniform_scores, cap, budget)

        # 3. Rule-based
        median_cpa = v["cpa"].median()
        dead = (v["known_conversions"] == 0) & (v["cost"] > 3.0 * median_cpa)
        rule_scores = -v["cpa"].fillna(np.inf)
        rule_scores[dead] = -1e18
        rule_cap = cap.where(~dead, 0.0)
        allocations["rule_based"] = self._greedy(rule_scores, rule_cap, budget)

        # 4. Thompson Sampling
        pooled = v["known_conversions"].sum() / max(v["cost"].sum(), 1e-12)
        prior_cost = 1.0 / max(pooled, 1e-12)
        a = 1.0 + v["known_conversions"]
        b = prior_cost + v["cost"]
        thompson_scores = pd.Series(
            rng.gamma(a.values, 1.0 / b.values), index=v.index
        )
        allocations["thompson_sampling"] = self._greedy(thompson_scores, cap, budget)

        return allocations

    def _greedy(self, scores, capacity, budget):
        alloc = pd.Series(0.0, index=capacity.index)
        remaining = budget
        for c in scores.sort_values(ascending=False).index:
            if remaining <= 0:
                break
            take = min(float(capacity.get(c, 0)), remaining)
            if take <= 0:
                continue
            alloc[c] = take
            remaining -= take
        return alloc


if __name__ == "__main__":
    logger.info("Testing Portfolio Strategy Agent...")

    from src.agents.observer_v2 import ObserverAgentV2

    observer = ObserverAgentV2()
    observer.initialise()
    obs = observer.observe(day=7)

    # Build view dataframe
    daily_cost = observer.daily_cost
    conversions = observer.conversions

    from src.data.criteo_prep import observable_view
    view = observable_view(daily_cost, conversions, decision_day=7)

    # Test portfolio strategy
    agent = PortfolioStrategyAgent()
    budget = float(daily_cost[daily_cost["day"] == 7]["cost"].sum())

    lag_note = (
        "Conversion lag warning: conversions from the last 3 days "
        "are underreported by ~40% as many have not yet been attributed. "
        "Do not penalise recently-started campaigns for low conversion counts."
    )

    strategy = agent.get_strategy(
        day=7,
        view_df=view,
        budget=budget,
        lag_note=lag_note
    )

    print("\n=== PORTFOLIO STRATEGY ===")
    print(json.dumps({k: v for k, v in strategy.items()
                      if k not in ['input_tokens', 'output_tokens']}, indent=2))

    # Test allocator
    allocator = PortfolioAllocator()
    llm_alloc = allocator.allocate(view, budget, strategy)
    baselines = allocator.get_baseline_allocations(view, budget)

    print(f"\n=== ALLOCATION COMPARISON ===")
    print(f"Total budget: {round(budget, 4)}")
    print(f"LLM+Bandit allocated: {round(float(llm_alloc.sum()), 4)}")
    for name, alloc in baselines.items():
        print(f"{name}: {round(float(alloc.sum()), 4)}")

    print(f"\nCampaigns excluded by LLM: {len(strategy.get('exclude_campaigns', []))}")
    print(f"Campaigns protected by LLM: {len(strategy.get('protect_campaigns', []))}")
    print(f"Human escalation: {strategy.get('escalate_to_human')}")
    print(f"Cost: ${strategy.get('cost_usd')}")

    print("\n✅ Portfolio Strategy Agent working correctly!")
