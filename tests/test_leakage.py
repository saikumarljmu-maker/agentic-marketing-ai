"""
Leakage tests for the observable view.
Run: python -m pytest tests -q
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.criteo_prep import observable_view

PROC = Path("data/processed")


def _toy():
    daily_cost = pd.DataFrame({
        "day": [5, 6, 6],
        "campaign": [1, 1, 2],
        "impressions": [10, 10, 10],
        "clicks": [1, 1, 1],
        "cost": [1.0, 1.0, 1.0],
        "last_day_cost": [1.0, 1.0, 1.0]
    })
    conversions = pd.DataFrame({
        "campaign":       [1, 1, 1, 2],
        "impression_day": [5, 6, 6, 6],
        "conversion_day": [6, 6, 7, 9],
        "conversions":    [1, 1, 1, 1]
    })
    return daily_cost, conversions


def test_future_conversions_hidden_toy():
    dc, conv = _toy()
    v = observable_view(dc, conv, decision_day=7).set_index("campaign")
    assert v.loc[1, "known_conversions"] == 2
    assert v.loc[2, "known_conversions"] == 0
    assert np.isnan(v.loc[2, "cpa"])


def test_no_future_cost_toy():
    dc, conv = _toy()
    dc = pd.concat([dc, pd.DataFrame({
        "day": [7], "campaign": [1],
        "impressions": [99], "clicks": [9],
        "cost": [50.0], "last_day_cost": [50.0]
    })])
    v = observable_view(dc, conv, decision_day=7).set_index("campaign")
    assert v.loc[1, "cost"] == 2.0


def test_ctr_column_present():
    dc, conv = _toy()
    v = observable_view(dc, conv, decision_day=7)
    assert "ctr" in v.columns


@pytest.mark.skipif(
    not (PROC / "conversions.parquet").exists(),
    reason="processed data not present"
)
@pytest.mark.parametrize("day", range(7, 24))
def test_real_data_no_leakage(day):
    dc = pd.read_parquet(PROC / "daily_cost.parquet")
    conv = pd.read_parquet(PROC / "conversions.parquet")
    v = observable_view(dc, conv, decision_day=day).set_index("campaign")
    allowed = conv[
        (conv["conversion_day"] < day) &
        (conv["impression_day"] >= day - 7) &
        (conv["impression_day"] <= day - 1)
    ]
    expected = allowed.groupby("campaign")["conversions"].sum()
    got = v["known_conversions"]
    assert got.sum() == expected.sum()
