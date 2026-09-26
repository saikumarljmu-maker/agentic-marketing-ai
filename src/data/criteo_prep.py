"""
Corrected Criteo preparation — v2 fixes applied.
observable_view now includes ctr column.
"""

from pathlib import Path
import numpy as np
import pandas as pd

DAY = 86_400
RAW = Path("data/raw/criteo_attribution_dataset/criteo_attribution_dataset.tsv")
OUT = Path("data/processed")
USECOLS = ["timestamp", "campaign", "click", "cost", "conversion_timestamp",
           "conversion_id", "attribution"]


def prepare(raw_path=RAW, out_dir=OUT, campaigns=None, n_campaigns=None,
            seed=42, chunksize=2_000_000):
    raw_path, out_dir = Path(raw_path), Path(out_dir)
    keep = set(campaigns) if campaigns is not None else None

    if keep is None and n_campaigns:
        print(f"Sampling {n_campaigns} whole campaigns...")
        ids = pd.read_csv(raw_path, sep="\t", usecols=["campaign"])["campaign"].unique()
        rng = np.random.default_rng(seed)
        keep = set(rng.choice(ids, size=min(n_campaigns, len(ids)), replace=False).tolist())
        print(f"Selected {len(keep)} campaigns")

    cost_parts, conv_parts = [], []
    chunk_num = 0

    for chunk in pd.read_csv(raw_path, sep="\t", usecols=USECOLS, chunksize=chunksize):
        chunk_num += 1
        print(f"Processing chunk {chunk_num} ({len(chunk):,} rows)...")
        if keep is not None:
            chunk = chunk[chunk["campaign"].isin(keep)]
        chunk["day"] = (chunk["timestamp"] // DAY).astype(int)

        cost_parts.append(chunk.groupby(["day", "campaign"]).agg(
            impressions=("timestamp", "size"),
            clicks=("click", "sum"),
            cost=("cost", "sum"),
            last_day_cost=("cost", "sum")
        ).reset_index())

        att = chunk[(chunk["attribution"] == 1) & (chunk["conversion_id"] >= 0)].copy()
        if len(att) > 0:
            att["conversion_day"] = (att["conversion_timestamp"] // DAY).astype(int)
            conv_parts.append(pd.DataFrame({
                "conversion_id": att["conversion_id"].values,
                "campaign":      att["campaign"].values,
                "impression_day": att["day"].values,
                "conversion_day": att["conversion_day"].values,
            }))

    print("Aggregating daily cost table...")
    daily_cost = (pd.concat(cost_parts)
                  .groupby(["day", "campaign"], as_index=False)
                  .agg(impressions=("impressions", "sum"),
                       clicks=("clicks", "sum"),
                       cost=("cost", "sum"),
                       last_day_cost=("cost", "last"))
                  .sort_values(["day", "campaign"]))

    print("Aggregating conversions table...")
    conv = pd.concat(conv_parts).drop_duplicates("conversion_id")
    conversions = (conv.groupby(["campaign", "impression_day", "conversion_day"])
                   .size().rename("conversions").reset_index())

    out_dir.mkdir(parents=True, exist_ok=True)
    daily_cost.to_parquet(out_dir / "daily_cost.parquet", index=False)
    conversions.to_parquet(out_dir / "conversions.parquet", index=False)

    days = daily_cost["day"].nunique()
    camps = daily_cost["campaign"].nunique()
    imps = int(daily_cost["impressions"].sum())
    convs = int(conversions["conversions"].sum())
    print(f"\n=== PREPARATION COMPLETE ===")
    print(f"Days:                    {days}")
    print(f"Campaigns:               {camps:,}")
    print(f"Total impressions:       {imps:,}")
    print(f"Distinct conversions:    {convs:,}")
    return daily_cost, conversions


def observable_view(daily_cost, conversions, decision_day, window=7):
    """
    What a manager could see at START of decision_day.
    Now includes ctr column. No future information.
    """
    lo, hi = decision_day - window, decision_day - 1
    c = daily_cost[(daily_cost["day"] >= lo) & (daily_cost["day"] <= hi)]
    cost = c.groupby("campaign").agg(
        cost=("cost", "sum"),
        clicks=("clicks", "sum"),
        impressions=("impressions", "sum"),
        last_day_cost=("last_day_cost", "last")
    )
    k = conversions[
        (conversions["impression_day"] >= lo) &
        (conversions["impression_day"] <= hi) &
        (conversions["conversion_day"] < decision_day)
    ]
    known = k.groupby("campaign")["conversions"].sum().rename("known_conversions")
    view = cost.join(known, how="left").fillna({"known_conversions": 0})
    view["cpa"] = view["cost"] / view["known_conversions"].replace(0, np.nan)
    view["ctr"] = view["clicks"] / view["impressions"].replace(0, np.nan)
    view = view.fillna({"ctr": 0.0})
    return view.reset_index()


def realised_outcomes(daily_cost, conversions, day):
    """Ground truth for evaluation only."""
    cost = daily_cost[daily_cost["day"] == day].set_index("campaign")["cost"]
    conv = (conversions[conversions["impression_day"] == day]
            .groupby("campaign")["conversions"].sum())
    return pd.DataFrame({"logged_cost": cost}).join(conv.rename("conversions")).fillna(0)


if __name__ == "__main__":
    prepare()
