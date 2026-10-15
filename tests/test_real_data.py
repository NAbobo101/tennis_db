"""Checks against the full vendored dataset (``pytest -m slow``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tennis_predictor.config import Settings, load_settings
from tennis_predictor.data.loader import load_matches
from tennis_predictor.features.dataset import build_dataset

ROOT = Path(__file__).resolve().parents[1]
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (ROOT / "data" / "raw" / "tennis_atp" / "atp_matches_2024.csv").is_file(),
        reason="vendored data not available",
    ),
]


@pytest.fixture(scope="module")
def atp() -> tuple[Settings, pd.DataFrame]:
    settings = load_settings(ROOT / "configs" / "atp.yaml")
    data = settings.data.model_copy(
        update={"raw_dir": ROOT / "data" / "raw", "cache_dir": ROOT / "data" / "interim"}
    )
    settings = settings.model_copy(update={"data": data})
    return settings, load_matches(settings.data, settings.tour)


def test_row_counts(atp: tuple[Settings, pd.DataFrame]) -> None:
    _, matches = atp
    assert 900_000 < len(matches) < 930_000
    main = matches.loc[(matches["tier"] == "main") & matches["year"].between(2023, 2024)]
    assert 5_500 < len(main) < 6_500


def test_no_player_reappears_after_losing_a_knockout(atp: tuple[Settings, pd.DataFrame]) -> None:
    """With the chronological sort, a loser never plays again in the same knockout event."""
    _, matches = atp
    knockout = matches.loc[
        (matches["tier"] == "main")
        & ~matches["round"].isin(["RR", "BR"])
        & (matches["year"] >= 2000)
    ]
    violations = 0
    for _, event in knockout.groupby("tourney_id", sort=False):
        eliminated: set[int] = set()
        for winner, loser in zip(
            event["winner_id"].tolist(), event["loser_id"].tolist(), strict=True
        ):
            violations += winner in eliminated or loser in eliminated
            eliminated.add(loser)
    assert violations <= 10  # a handful of team events (ATP Cup, United Cup, Olympics)


def test_truncation_invariance_on_real_data(atp: tuple[Settings, pd.DataFrame]) -> None:
    settings, matches = atp
    full, _ = build_dataset(matches, settings)
    truncated, _ = build_dataset(matches.loc[matches["year"] <= 2022], settings)
    rows = len(truncated)
    assert rows == int((full.meta["year"] <= 2022).sum())
    pd.testing.assert_frame_equal(full.X.iloc[:rows], truncated.X)
    np.testing.assert_array_equal(full.y[:rows], truncated.y)
