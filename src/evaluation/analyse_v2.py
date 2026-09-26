"""
Statistical analysis of V2 replay results.
Produces: bootstrap CIs, Holm correction, calibration, escalation audit, lag ablation.

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
    """Bootstrap CIs on day-level differences."""
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
            "n_days": len(d)
        })
    return rows


def calibration(dec_df, df):
    """Accuracy by stated confidence level."""
    if dec_df.empty or "confidence" not in dec_df.columns:
        return pd.DataFrame()

    rows = []
    for _, row in dec_df.iterrows():
        day = row.get("day")
        bf = row.get("budget_fraction")
        conf = row.get("confidence", "unknown")

        subset = df[
            (df["day"] == day) &
            (df["budget_fraction"] == bf) &
            (df["policy"] == "llm_portfolio")
        ]
        thompson = df[
            (df["day"] == day) &
            (df["budget_fraction"] == bf) &
            (df["policy"] == "thompson_sampling")
        ]

        if subset.empty or thompson.empty:
            continue

        llm_conv = subset["conversions"].mean()
        ts_conv = thompson["conversions"].mean()
        llm_better = llm_conv > ts_conv

        rows.append({
            "day": day,
            "budget_fraction": bf,
            "confidence": conf,
            "llm_better_than_thompson": llm_better,
            "llm_conv": round(llm_conv, 2),
            "thompson_conv": round(ts_conv, 2),
            "diff": round(llm_conv - ts_conv, 2)
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    summary = result.groupby("confidence")["llm_better_than_thompson"].agg(
        ["mean", "count"]
    ).rename(columns={"mean": "accuracy", "count": "n"})
    return summary


def escalation_audit(dec_df, df):
    """Are escalated days actually harder? Mann-Whitney test."""
    if dec_df.empty or "escalate_to_human" not in dec_df.columns:
        return {}

    escalated_days = set(
        dec_df[dec_df["escalate_to_human"] == True]["day"].tolist()
    )
    if not escalated_days:
        return {"escalated_days": 0, "note": "No escalations found"}

    llm_df = df[df["policy"] == "llm_portfolio"].copy()
    llm_df["escalated"] = llm_df["day"].isin(escalated_days)

    esc = llm_df[llm_df["escalated"]]["pct_of_oracle"].values
    non_esc = llm_df[~llm_df["escalated"]]["pct_of_oracle"].values

    if len(esc) < 2 or len(non_esc) < 2:
        return {"escalated_days": len(escalated_days), "note": "Insufficient data"}

    stat, p = stats.mannwhitneyu(esc, non_esc, alternative="less")

    return {
        "escalated_days": len(escalated_days),
        "mean_pct_oracle_escalated": round(esc.mean(), 2),
        "mean_pct_oracle_not_escalated": round(non_esc.mean(), 2),
        "mann_whitney_p": round(p, 4),
        "escalated_days_harder": p < 0.05,
        "interpretation": (
            "Escalated days were significantly harder (lower % oracle)"
            if p < 0.05
            else "No significant difference — escalation not clearly linked to difficulty"
        )
    }


def lag_ablation(df_lag, df_nolag):
    """Compare lag vs no-lag LLM performance."""
    rows = []
    for bf in df_lag["budget_fraction"].unique():
        lag = df_lag[
            (df_lag["budget_fraction"] == bf) &
            (df_lag["policy"] == "llm_portfolio")
        ]["pct_of_oracle"].mean()
        nolag = df_nolag[
            (df_nolag["budget_fraction"] == bf) &
            (df_nolag["policy"] == "llm_portfolio")
        ]["pct_of_oracle"].mean()
        rows.append({
            "budget_fraction": bf,
            "with_lag_note_pct": round(lag, 2),
            "without_lag_note_pct": round(nolag, 2),
            "difference_pp": round(lag - nolag, 2)
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

    # 1. Summary table
    print("MEAN % OF HINDSIGHT ORACLE:")
    pivot = df.groupby(["policy", "budget_fraction"])[
        "pct_of_oracle"].mean().round(2).unstack()
    print(pivot.to_string())

    print("\nMEAN CONVERSIONS PER DAY:")
    conv_pivot = df.groupby(["policy", "budget_fraction"])[
        "conversions"].mean().round(2).unstack()
    print(conv_pivot.to_string())

    # 2. Bootstrap CIs
    print("\n=== BOOTSTRAP CONFIDENCE INTERVALS ===")
    comparisons = [
        ("llm_portfolio", "thompson_sampling"),
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
        p_vals = boot_df["p_boot"].values
        boot_df["p_holm"] = holm(p_vals).round(4)
        boot_df["significant_holm"] = boot_df["p_holm"] < 0.05
        print(boot_df[[
            "comparison", "budget_fraction", "mean_diff_pp",
            "ci_95_low", "ci_95_high", "p_boot", "p_holm", "significant_holm"
        ]].to_string(index=False))
        boot_df.to_csv(out_dir / "bootstrap.csv", index=False)

    # 3. Calibration
    print("\n=== CONFIDENCE CALIBRATION ===")
    cal = calibration(dec_df, df)
    if not cal.empty:
        print(cal.to_string())
        cal.to_csv(out_dir / "calibration.csv")
    else:
        print("No calibration data available")

    # 4. Escalation audit
    print("\n=== ESCALATION AUDIT ===")
    esc = escalation_audit(dec_df, df)
    print(json.dumps(esc, indent=2))
    with open(out_dir / "escalation.json", "w") as f:
        json.dump(esc, f, indent=2)

    # 5. Lag ablation
    if args.ablation:
        print(f"\n=== LAG ABLATION: {args.run} vs {args.ablation} ===")
        df_nolag, _ = load(args.ablation)
        abl = lag_ablation(df, df_nolag)
        print(abl.to_string(index=False))
        abl.to_csv(out_dir / "ablation.csv", index=False)

    print(f"\n✅ Analysis complete — results in {out_dir}")


if __name__ == "__main__":
    main()
