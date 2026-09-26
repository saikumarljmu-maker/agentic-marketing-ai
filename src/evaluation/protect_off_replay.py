"""
Protect-off replay — reproducibility fix.
Replays saved LLM decisions twice using exactly the same data,
days, budgets, seeds and allocator as the main run.
The with-protection arm must reproduce Table 4.4 LLM row exactly.

Usage:
    python -m src.evaluation.protect_off_replay --run v2
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROC    = Path("data/processed")
RESULTS = Path("results")
LOGS    = Path("logs")


def _greedy(scores, capacity, budget):
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


def _llm_alloc(view, budget, strategy, seed, with_protect):
    rng = np.random.default_rng(seed)
    v = view.set_index("campaign")
    exclude = set(strategy.get("exclude_campaigns") or []) & set(v.index)
    protect = set(strategy.get("protect_campaigns") or []) & set(v.index) \
              if with_protect else set()
    strat = strategy.get("strategy", "balanced")
    mult = {"aggressive": 1.3, "balanced": 1.0, "conservative": 0.7}.get(strat, 1.0)
    pooled = v["known_conversions"].sum() / max(v["cost"].sum(), 1e-12)
    prior_cost = 1.0 / max(pooled, 1e-12)
    a = 1.0 + v["known_conversions"]
    b = prior_cost + v["cost"]
    scores = pd.Series(rng.gamma(a.values, 1.0 / b.values), index=v.index)
    scores[list(exclude)] = -np.inf
    if protect:
        scores[list(protect)] = scores.max() * 2
    if strategy.get("focus_on_conversions", True):
        converting = v[v["known_conversions"] > 0].index
        scores[converting] *= mult
    cap = v["last_day_cost"].copy()
    cap[list(exclude)] = 0.0
    return _greedy(scores, cap, budget)


def _score(alloc, truth):
    conv_sum = 0.0
    for camp in alloc.index:
        if camp not in truth.index:
            continue
        ls = float(truth.loc[camp, "logged_cost"])
        lc = float(truth.loc[camp, "conversions"])
        ps = float(alloc.get(camp, 0))
        if ls <= 0:
            continue
        conv_sum += lc * (ps / ls) if ps <= ls else lc
    return conv_sum


def _oracle_conv(view, truth, budget):
    oracle_rate = (truth["conversions"] / truth["logged_cost"].replace(0, np.nan)).fillna(0)
    cap = truth["logged_cost"]
    alloc = pd.Series(0.0, index=cap.index)
    remaining = budget
    for c in oracle_rate.sort_values(ascending=False).index:
        if remaining <= 0:
            break
        take = min(float(cap.get(c, 0)), remaining)
        alloc[c] = take
        remaining -= take
    return _score(alloc, truth)


def run(run_name="v2", n_boot=10_000, seed=42):
    import sys
    sys.path.insert(0, "src")
    from data.criteo_prep import observable_view, realised_outcomes

    dc   = pd.read_parquet(PROC / "daily_cost.parquet")
    conv = pd.read_parquet(PROC / "conversions.parquet")

    dec_path = LOGS / f"llm_portfolio_decisions_{run_name}.jsonl"
    decisions = [json.loads(l) for l in open(dec_path) if l.strip()]
    print(f"Loaded {len(decisions)} decisions from {dec_path}")

    seen = {}
    for d in decisions:
        key = (d["day"], d.get("budget_fraction"))
        if key not in seen:
            seen[key] = d

    rows = []
    for (day, bf), strategy in sorted(seen.items()):
        if bf is None:
            continue
        view = observable_view(dc, conv, decision_day=day, window=7)
        if view.empty:
            continue
        today_logged = dc[dc["day"] == day]["cost"].sum()
        if today_logged <= 0:
            continue
        budget = today_logged * bf
        truth = realised_outcomes(dc, conv, day)
        if truth.empty:
            continue
        oc = _oracle_conv(view, truth, budget)
        if oc <= 0:
            continue
        for seed_i in range(3):
            with_alloc    = _llm_alloc(view, budget, strategy, seed_i, True)
            without_alloc = _llm_alloc(view, budget, strategy, seed_i, False)
            rows.append({
                "day": day,
                "budget_fraction": bf,
                "seed": seed_i,
                "with_pct":    _score(with_alloc,    truth) / oc * 100,
                "without_pct": _score(without_alloc, truth) / oc * 100,
                "oracle_conv": oc,
            })

    df = pd.DataFrame(rows)
    df["diff_pp"] = df["without_pct"] - df["with_pct"]

    summary_rows = []
    for bf, g in df.groupby("budget_fraction"):
        day_g = g.groupby("day")[["with_pct", "without_pct", "diff_pp"]].mean()
        d_vals = day_g["diff_pp"].values
        rng = np.random.default_rng(seed)
        boots = rng.choice(d_vals, (n_boot, len(d_vals)), replace=True).mean(1)
        p = float(2 * min((boots <= 0).mean(), (boots >= 0).mean()))
        p = min(max(p, 1 / n_boot), 1.0)
        summary_rows.append({
            "budget_fraction":     bf,
            "with_protect_pct":    round(day_g["with_pct"].mean(), 2),
            "without_protect_pct": round(day_g["without_pct"].mean(), 2),
            "diff_pp":             round(d_vals.mean(), 2),
            "ci_95_low":           round(float(np.percentile(boots, 2.5)), 2),
            "ci_95_high":          round(float(np.percentile(boots, 97.5)), 2),
            "p_boot":              round(p, 4),
            "significant":         p < 0.05,
        })

    summary = pd.DataFrame(summary_rows).sort_values("budget_fraction")
    print("\n=== PROTECT-OFF REPLAY ===")
    print(summary.to_string(index=False))
    print("\nwith_protect_pct should match Table 4.4 LLM Portfolio row exactly.")
    out = RESULTS / f"protect_off_{run_name}.csv"
    summary.to_csv(out, index=False)
    print(f"Saved to {out}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="v2")
    args = parser.parse_args()
    run(args.run)
