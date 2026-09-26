"""
Statistical analysis of V2 replay — Round 3 fixes.
Adds: LLM vs rule_based, lag bootstrap CI, protection audit, fixed escalation audit.

Usage:
    python -m src.evaluation.analyse_v2 --run v2
    python -m src.evaluation.analyse_v2 --run v2 --ablation v2_nolag
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

PROC = Path("data/processed")
RES = Path("results")
LOGS = Path("logs")


def load(run):
    df = pd.read_parquet(RES / f"replay_results_{run}.parquet")
    dec_path = LOGS / f"llm_portfolio_decisions_{run}.jsonl"
    if dec_path.exists():
        dec = [json.loads(l) for l in open(dec_path) if l.strip()]
        dec_df = pd.DataFrame(dec)
    else:
        dec_df = pd.DataFrame()
    return df, dec_df


def holm(p_values):
    p = np.asarray(p_values)
    order = np.argsort(p)
    adj = np.empty_like(p, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def paired_bootstrap(df, policy_a, policy_b, n_boot=10_000, seed=0):
    rows = []
    for bf, g in df.groupby("budget_fraction"):
        pv = g.groupby(["day", "policy"])["pct_of_oracle"].mean().unstack()
        if policy_a not in pv.columns or policy_b not in pv.columns:
            continue
        d = (pv[policy_a] - pv[policy_b]).dropna().values
        rng = np.random.default_rng(seed)
        boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
        p_two = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
        rows.append({
            "budget_fraction": bf,
            "comparison": f"{policy_a} vs {policy_b}",
            "mean_diff_pp": round(d.mean(), 3),
            "ci_95_low": round(np.percentile(boots, 2.5), 3),
            "ci_95_high": round(np.percentile(boots, 97.5), 3),
            "p_boot": round(min(max(p_two, 1/n_boot), 1.0), 4),
            "cohens_dz": round(d.mean() / d.std(ddof=1), 3) if d.std(ddof=1) > 0 else 0,
            "n_days": len(d),
            "practically_meaningful": abs(d.mean()) >= 1.0
        })
    return rows


def protection_audit(dec_df, daily_cost, conversions, horizon=7):
    """
    Tests the crowding explanation:
    1. What fraction of budget do protected campaigns absorb?
    2. Are protected campaigns actually more efficient than portfolio average?
    """
    if dec_df.empty:
        return {}

    rows = []
    for _, row in dec_df.iterrows():
        day = row.get("day")
        bf = row.get("budget_fraction")
        protect = row.get("protect_campaigns") or []
        if not protect:
            continue

        # Future 7-day outcomes
        days_ahead = range(day, min(day + horizon, 31))
        c = daily_cost[daily_cost["day"].isin(days_ahead)].groupby(
            "campaign")["cost"].sum()
        k = conversions[conversions["impression_day"].isin(days_ahead)].groupby(
            "campaign")["conversions"].sum()

        port_cpa = c.sum() / max(k.sum(), 1e-12)
        cap_sum = sum(float(c.get(int(cid), 0)) for cid in protect)

        correct = 0
        total = 0
        for cid in protect:
            cid = int(cid)
            cost_c = float(c.get(cid, 0))
            if cost_c <= 0:
                continue
            conv_c = float(k.get(cid, 0))
            cpa_c = cost_c / conv_c if conv_c > 0 else np.inf
            total += 1
            if cpa_c < port_cpa:
                correct += 1

        rows.append({
            "day": day,
            "budget_fraction": bf,
            "n_protected": len(protect),
            "protect_capacity": round(cap_sum, 4),
            "protect_accuracy": round(correct / total, 3) if total > 0 else None
        })

    if not rows:
        return {}

    result_df = pd.DataFrame(rows)
    return {
        "mean_protect_accuracy": round(result_df["protect_accuracy"].mean(), 3),
        "mean_n_protected": round(result_df["n_protected"].mean(), 1),
        "mean_protect_capacity": round(result_df["protect_capacity"].mean(), 4),
        "n_observations": len(result_df)
    }


def escalation_audit(dec_df, df):
    """
    Are escalated days harder? Difficulty measured on NON-LLM baselines
    (not circular). Uses thompson_sampling as difficulty proxy.
    """
    if dec_df.empty or "escalate_to_human" not in dec_df.columns:
        return {"escalated_days": 0, "note": "No decisions found"}

    escalated_days = set(
        int(d) for d in dec_df[
            dec_df["escalate_to_human"] == True
        ]["day"].tolist()
    )

    if not escalated_days:
        return {
            "escalated_days": 0,
            "note": "No escalations triggered in this run — escalation governance not observed"
        }

    # Use thompson as difficulty proxy (not LLM itself — avoids circularity)
    ts_df = df[df["policy"] == "thompson_sampling"].copy()
    ts_df["escalated"] = ts_df["day"].isin(escalated_days)

    esc = ts_df[ts_df["escalated"]]["pct_of_oracle"].values
    non_esc = ts_df[~ts_df["escalated"]]["pct_of_oracle"].values

    if len(esc) < 2 or len(non_esc) < 2:
        return {"escalated_days": len(escalated_days),
                "note": "Insufficient data for test"}

    stat, p = stats.mannwhitneyu(esc, non_esc, alternative="less")

    return {
        "escalated_days": int(len(escalated_days)),
        "mean_pct_oracle_escalated_days": round(float(esc.mean()), 2),
        "mean_pct_oracle_non_escalated_days": round(float(non_esc.mean()), 2),
        "mann_whitney_p": round(float(p), 4),
        "escalated_days_harder": bool(p < 0.05),
        "interpretation": (
            "Escalated days were significantly harder"
            if p < 0.05
            else "No significant difference in difficulty on escalated days"
        )
    }


def calibration(dec_df, df):
    """LLM accuracy vs Thompson by confidence level."""
    if dec_df.empty or "confidence" not in dec_df.columns:
        return pd.DataFrame()

    rows = []
    for _, row in dec_df.iterrows():
        day = row.get("day")
        bf = row.get("budget_fraction")
        conf = row.get("confidence", "unknown")
        if bf is None:
            continue

        subset = df[(df["day"] == day) & (df["budget_fraction"] == bf) &
                    (df["policy"] == "llm_portfolio")]
        thompson = df[(df["day"] == day) & (df["budget_fraction"] == bf) &
                      (df["policy"] == "thompson_sampling")]

        if subset.empty or thompson.empty:
            continue

        llm_conv = subset["conversions"].mean()
        ts_conv = thompson["conversions"].mean()
        rows.append({
            "confidence": conf,
            "llm_better": bool(llm_conv > ts_conv),
            "diff": round(llm_conv - ts_conv, 2)
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    return result.groupby("confidence")["llm_better"].agg(
        accuracy="mean", n="count"
    ).round(3)


def lag_ablation(df_lag, df_nolag, n_boot=10_000, seed=42):
    """Lag vs no-lag with paired bootstrap CI."""
    rows = []
    for bf in df_lag["budget_fraction"].unique():
        pv_lag = df_lag[(df_lag["budget_fraction"] == bf) &
                        (df_lag["policy"] == "llm_portfolio")
                        ].groupby("day")["pct_of_oracle"].mean()
        pv_nolag = df_nolag[(df_nolag["budget_fraction"] == bf) &
                            (df_nolag["policy"] == "llm_portfolio")
                            ].groupby("day")["pct_of_oracle"].mean()
        common = pv_lag.index.intersection(pv_nolag.index)
        d = (pv_lag[common] - pv_nolag[common]).values
        if len(d) < 2:
            continue
        rng = np.random.default_rng(seed)
        boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
        p_two = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
        rows.append({
            "budget_fraction": bf,
            "mean_diff_pp": round(d.mean(), 3),
            "ci_95_low": round(np.percentile(boots, 2.5), 3),
            "ci_95_high": round(np.percentile(boots, 97.5), 3),
            "p_boot": round(min(max(p_two, 1/n_boot), 1.0), 4),
            "significant": bool(p_two < 0.05)
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="v2")
    parser.add_argument("--ablation", default=None)
    args = parser.parse_args()

    print(f"\n=== ANALYSIS: {args.run} ===\n")
    df, dec_df = load(args.run)
    out_dir = RES / f"analysis_{args.run}"
    out_dir.mkdir(exist_ok=True)

    print("MEAN % OF HINDSIGHT ORACLE:")
    pivot = df.groupby(["policy", "budget_fraction"])[
        "pct_of_oracle"].mean().round(2).unstack()
    print(pivot.to_string())

    print("\nMEAN CONVERSIONS PER DAY:")
    conv_pivot = df.groupby(["policy", "budget_fraction"])[
        "conversions"].mean().round(2).unstack()
    print(conv_pivot.to_string())

    print("\n=== BOOTSTRAP CONFIDENCE INTERVALS ===")
    comparisons = [
        ("llm_portfolio", "thompson_sampling"),
        ("llm_portfolio", "rule_based"),
        ("llm_portfolio", "logged_mix"),
        ("thompson_sampling", "logged_mix"),
        ("rule_based", "logged_mix"),
    ]
    all_rows = []
    for a, b in comparisons:
        rows = paired_bootstrap(df, a, b)
        all_rows.extend(rows)

    boot_df = pd.DataFrame(all_rows)
    if not boot_df.empty:
        boot_df["p_holm"] = holm(boot_df["p_boot"].values).round(4)
        boot_df["significant_holm"] = boot_df["p_holm"] < 0.05
        print(boot_df[[
            "comparison", "budget_fraction", "mean_diff_pp",
            "ci_95_low", "ci_95_high", "p_holm",
            "significant_holm", "practically_meaningful"
        ]].to_string(index=False))
        boot_df.to_csv(out_dir / "bootstrap.csv", index=False)

    print("\n=== CALIBRATION ===")
    cal = calibration(dec_df, df)
    if not cal.empty:
        print(cal.to_string())
        cal.to_csv(out_dir / "calibration.csv")
    else:
        print("No calibration data (budget_fraction missing from decisions)")

    print("\n=== ESCALATION AUDIT ===")
    esc = escalation_audit(dec_df, df)
    print(json.dumps(esc, indent=2))
    with open(out_dir / "escalation.json", "w") as f:
        json.dump(esc, f, indent=2)

    print("\n=== PROTECTION AUDIT ===")
    if not dec_df.empty:
        dc = pd.read_parquet(PROC / "daily_cost.parquet")
        conv = pd.read_parquet(PROC / "conversions.parquet")
        prot = protection_audit(dec_df, dc, conv)
        print(json.dumps(prot, indent=2))
        with open(out_dir / "protection_audit.json", "w") as f:
            json.dump(prot, f, indent=2)

    if args.ablation:
        print(f"\n=== LAG ABLATION: {args.run} vs {args.ablation} ===")
        df_nolag, _ = load(args.ablation)
        abl = lag_ablation(df, df_nolag)
        print(abl.to_string(index=False))
        abl.to_csv(out_dir / "ablation.csv", index=False)

    print(f"\n✅ Analysis complete — results in {out_dir}")


if __name__ == "__main__":
    main()
