"""
Portfolio Strategy Agent — Round 3 fixes.
CI is now deterministic Gamma-posterior quantiles (scipy).
budget_fraction stored with each decision.
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
from scipy import stats as scipy_stats

load_dotenv()
LOGS_PATH = Path("logs")
LOGS_PATH.mkdir(exist_ok=True)


class PortfolioStrategyAgent:

    def __init__(self, top_n=25):
        self.client = anthropic.Anthropic(
            default_headers={
                "anthropic-workspace-id": os.getenv("ANTHROPIC_WORKSPACE_ID")
            }
        )
        self.model = "claude-sonnet-4-6"
        self.top_n = top_n
        self.decisions = []
        logger.success("Portfolio Strategy Agent initialised (temperature=0, deterministic CI)")

    def _credible_interval(self, conv, cost, pooled_rate, alpha=0.10):
        """
        True 90% credible interval for CPA using Gamma-Poisson posterior.
        Deterministic — no random draws.
        """
        prior_cost = 1.0 / max(pooled_rate, 1e-12)
        a = 1.0 + conv
        b = prior_cost + cost
        # Rate posterior: Gamma(a, 1/b)
        # CPA = 1/rate -> inverse-Gamma quantiles
        rate_hi = scipy_stats.gamma.ppf(1 - alpha/2, a=a, scale=1.0/b)
        rate_lo = scipy_stats.gamma.ppf(alpha/2, a=a, scale=1.0/b)
        cpa_lo = 1.0/rate_hi if rate_hi > 0 else np.inf
        cpa_hi = 1.0/rate_lo if rate_lo > 0 else np.inf
        return round(float(cpa_lo), 4), round(float(cpa_hi), 4)

    def _build_prompt(self, day, view_df, budget, lag_note, budget_fraction=None):
        v = view_df.copy().sort_values("known_conversions", ascending=False)
        pooled_rate = (v["known_conversions"].sum() /
                       max(v["cost"].sum(), 1e-12))

        rows = []
        for _, row in v.head(self.top_n).iterrows():
            conv = int(row["known_conversions"])
            cost = round(float(row["cost"]), 4)
            cpa_val = round(float(row["cpa"]), 4) if pd.notna(row.get("cpa")) and row["cpa"] > 0 else "N/A"
            ctr_val = round(float(row.get("ctr", 0)), 4) if pd.notna(row.get("ctr", 0)) else 0.0

            if conv > 0 and cost > 0:
                ci_lo, ci_hi = self._credible_interval(conv, cost, pooled_rate)
                ci = f"[{ci_lo:.4f}-{ci_hi:.4f}]"
            else:
                ci = "[unknown]"

            rows.append(
                f"CampID={int(row['campaign'])}: "
                f"spend={cost}, conv={conv}, "
                f"CPA={cpa_val}, CTR={ctr_val}, "
                f"CPA_90pct_CI={ci}"
            )

        portfolio_text = "\n".join(rows)
        total_spend = round(float(v["cost"].sum()), 4)
        total_conv = int(v["known_conversions"].sum())
        zero_conv = int((v["known_conversions"] == 0).sum())
        active = int((v["cost"] > 0).sum())
        bf_str = f" (budget fraction: {budget_fraction})" if budget_fraction else ""

        prompt = f"""You are a digital marketing portfolio manager for Day {day}.

PORTFOLIO SUMMARY:
- Total campaigns: {len(v)} ({active} active, {zero_conv} with zero known conversions)
- Total spend last 7 days: {total_spend}
- Total known conversions: {total_conv}
- Available budget today: {round(budget, 4)}{bf_str}
{lag_note}

TOP {self.top_n} CAMPAIGNS (by known conversions):
CampID: spend, conversions, CPA, CTR, 90pct_CPA_credible_interval
{portfolio_text}

Set STRATEGY PARAMETERS for the budget allocator.
Use INTEGER campaign IDs only (e.g. 9100693).

Respond ONLY with this JSON (no markdown):
{{
  "strategy": "aggressive" | "conservative" | "balanced",
  "protect_campaigns": [list of INTEGER campaign IDs],
  "exclude_campaigns": [list of INTEGER campaign IDs],
  "focus_on_conversions": true | false,
  "escalate_to_human": true | false,
  "escalation_reason": "reason or null",
  "confidence": "high" | "medium" | "low",
  "reasoning": "one sentence",
  "lag_acknowledged": true | false
}}"""
        return prompt

    def _parse_id(self, x):
        try:
            return int(str(x).replace("Camp", "").replace("camp", "").strip())
        except (ValueError, TypeError):
            return None

    def get_strategy(self, day, view_df, budget, lag_note="",
                     cache=None, cache_key=None, budget_fraction=None):
        if cache is not None and cache_key and cache_key in cache:
            logger.info(f"Day {day}: using cached strategy")
            cached = dict(cache[cache_key])
            cached["budget_fraction"] = budget_fraction
            return cached

        logger.info(f"Portfolio Strategy Agent — Day {day}")
        prompt = self._build_prompt(day, view_df, budget, lag_note, budget_fraction)

        try:
            start = time.time()
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            duration = round(time.time() - start, 2)
            text = response.content[0].text.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]

            strategy = json.loads(text)
            strategy["protect_campaigns"] = [
                v for v in (self._parse_id(x)
                for x in strategy.get("protect_campaigns", [])) if v is not None
            ]
            strategy["exclude_campaigns"] = [
                v for v in (self._parse_id(x)
                for x in strategy.get("exclude_campaigns", [])) if v is not None
            ]
            strategy["day"] = day
            strategy["budget_fraction"] = budget_fraction
            strategy["duration_seconds"] = duration
            strategy["input_tokens"] = response.usage.input_tokens
            strategy["output_tokens"] = response.usage.output_tokens
            strategy["cost_usd"] = round(
                (response.usage.input_tokens * 3 +
                 response.usage.output_tokens * 15) / 1_000_000, 6
            )
            strategy["fallback"] = False

            if cache is not None and cache_key:
                cache[cache_key] = strategy

            self.decisions.append(strategy)
            logger.success(
                f"Day {day}: {strategy.get('strategy')} | "
                f"protect={len(strategy.get('protect_campaigns',[]))} | "
                f"exclude={len(strategy.get('exclude_campaigns',[]))} | "
                f"escalate={strategy.get('escalate_to_human')} | "
                f"{duration}s | ${strategy['cost_usd']}"
            )
            return strategy

        except json.JSONDecodeError as e:
            logger.warning(f"Day {day} JSON parse failed: {e}")
            return self._fallback(day, budget_fraction)
        except Exception as e:
            logger.error(f"Day {day} strategy failed: {e}")
            return self._fallback(day, budget_fraction)

    def _fallback(self, day, budget_fraction=None):
        strat = {
            "day": day, "budget_fraction": budget_fraction,
            "strategy": "balanced",
            "protect_campaigns": [], "exclude_campaigns": [],
            "focus_on_conversions": True, "escalate_to_human": False,
            "escalation_reason": None, "confidence": "low",
            "reasoning": "Fallback: LLM unavailable",
            "lag_acknowledged": False, "fallback": True, "cost_usd": 0.0
        }
        self.decisions.append(strat)
        return strat

    def save(self, path=None, run_name="v2"):
        if path is None:
            path = LOGS_PATH / f"llm_portfolio_decisions_{run_name}.jsonl"
        with open(path, "w") as f:
            for d in self.decisions:
                f.write(json.dumps(d) + "\n")
        logger.success(f"Saved {len(self.decisions)} strategies to {path}")


class PortfolioAllocator:

    STRATEGY_MULTIPLIERS = {
        "aggressive": 1.3,
        "balanced": 1.0,
        "conservative": 0.7
    }

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

    def allocate(self, view_df, budget, strategy_params, seed=42):
        rng = np.random.default_rng(seed)
        v = view_df.set_index("campaign").copy()
        exclude = set(strategy_params.get("exclude_campaigns", [])) & set(v.index)
        protect = set(strategy_params.get("protect_campaigns", [])) & set(v.index)
        strat = strategy_params.get("strategy", "balanced")
        mult = self.STRATEGY_MULTIPLIERS.get(strat, 1.0)

        pooled = v["known_conversions"].sum() / max(v["cost"].sum(), 1e-12)
        prior_cost = 1.0 / max(pooled, 1e-12)
        a = 1.0 + v["known_conversions"]
        b = prior_cost + v["cost"]
        scores = pd.Series(
            rng.gamma(a.values, 1.0 / b.values), index=v.index
        )
        scores[list(exclude)] = -np.inf
        if protect:
            scores[list(protect)] = scores.max() * 2
        if strategy_params.get("focus_on_conversions", True):
            converting = v[v["known_conversions"] > 0].index
            scores[converting] *= mult

        cap = v["last_day_cost"].copy()
        cap[list(exclude)] = 0.0
        return self._greedy(scores, cap, budget)

    def get_baselines(self, view_df, budget, seed=42):
        rng = np.random.default_rng(seed)
        v = view_df.set_index("campaign").copy()
        cap = v["last_day_cost"]
        allocations = {}

        w = cap.clip(lower=0)
        allocations["logged_mix"] = w / w.sum() * budget if w.sum() > 0 else w

        allocations["uniform"] = self._greedy(
            pd.Series(rng.random(len(cap)), index=cap.index), cap, budget
        )

        median_cpa = v["cpa"].median()
        dead = (v["known_conversions"] == 0) & (v["cost"] > 3.0 * median_cpa)
        rule_scores = -v["cpa"].fillna(np.inf)
        rule_scores[dead] = -1e18
        allocations["rule_based"] = self._greedy(
            rule_scores, cap.where(~dead, 0.0), budget
        )

        rng2 = np.random.default_rng(seed)
        pooled = v["known_conversions"].sum() / max(v["cost"].sum(), 1e-12)
        prior_cost = 1.0 / max(pooled, 1e-12)
        a = 1.0 + v["known_conversions"]
        b = prior_cost + v["cost"]
        thompson_scores = pd.Series(
            rng2.gamma(a.values, 1.0 / b.values), index=v.index
        )
        allocations["thompson_sampling"] = self._greedy(thompson_scores, cap, budget)

        return allocations
