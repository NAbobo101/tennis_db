"""The tests that justify every number in the README: no feature may see the future."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tennis_predictor.config import Settings
from tennis_predictor.features.dataset import Dataset, build_dataset
from tennis_predictor.features.specs import swap_players

RESULT_COLUMNS = ["score", "outcome", "w_games", "l_games", "stats_ok"]


def _same_rows(left: Dataset, right: Dataset, rows: int) -> None:
    pd.testing.assert_frame_equal(left.X.iloc[:rows], right.X.iloc[:rows])
    pd.testing.assert_frame_equal(left.meta.iloc[:rows], right.meta.iloc[:rows])
    np.testing.assert_array_equal(left.y[:rows], right.y[:rows])


def test_features_ignore_the_match_result(
    matches: pd.DataFrame, settings: Settings, dataset: Dataset
) -> None:
    """Swapping winner and loser of one match must not change that match's features."""
    target = len(dataset) // 2
    meta = dataset.meta.iloc[target]
    position = matches.index[
        (matches["tourney_id"] == meta["tourney_id"])
        & (matches["round"] == meta["round"])
        & matches["winner_id"].isin([meta["a_id"], meta["b_id"]])
        & matches["loser_id"].isin([meta["a_id"], meta["b_id"]])
    ][0]
    flipped = matches.copy()
    for column in matches.columns:
        if column.startswith("winner_"):
            other = "loser_" + column.removeprefix("winner_")
            flipped.loc[position, [column, other]] = matches.loc[
                position, [other, column]
            ].to_numpy()
        elif column.startswith("w_"):
            other = "l_" + column.removeprefix("w_")
            flipped.loc[position, [column, other]] = matches.loc[
                position, [other, column]
            ].to_numpy()

    changed, _ = build_dataset(flipped, settings)
    _same_rows(dataset, changed, target)
    row_original = dataset.X.iloc[[target]].reset_index(drop=True)
    row_changed = changed.X.iloc[[target]].reset_index(drop=True)
    if changed.meta.iloc[target]["a_id"] != meta["a_id"]:
        row_changed = swap_players(row_changed)
    pd.testing.assert_frame_equal(row_original, row_changed)
    assert (
        changed.y[target] != dataset.y[target] or changed.meta.iloc[target]["a_id"] != meta["a_id"]
    )


def test_future_matches_do_not_change_past_features(
    matches: pd.DataFrame, settings: Settings, dataset: Dataset
) -> None:
    """Scrambling every result after a cut-off leaves all earlier rows untouched."""
    cutoff_day = int(dataset.meta["day"].quantile(0.6))
    future = matches["day"] > cutoff_day
    rng = np.random.default_rng(1)
    scrambled = matches.copy()
    scrambled.loc[future, "outcome"] = rng.choice(
        ["completed", "retired", "walkover"], size=int(future.sum())
    )
    scrambled.loc[future, "w_games"] = rng.integers(0, 40, size=int(future.sum())).astype("int32")
    stat_columns = [
        c for c in matches.columns if c.startswith(("w_", "l_")) and not c.endswith("games")
    ]
    scrambled.loc[future, stat_columns] = scrambled.loc[future, stat_columns].to_numpy()[::-1]

    changed, _ = build_dataset(scrambled, settings)
    past = int((dataset.meta["day"] <= cutoff_day).sum())
    assert past > 100
    _same_rows(dataset, changed, past)


def test_truncation_invariance(matches: pd.DataFrame, settings: Settings, dataset: Dataset) -> None:
    """Building the dataset from a prefix of history reproduces the same rows exactly."""
    cutoff_day = int(dataset.meta["day"].quantile(0.4))
    truncated, _ = build_dataset(matches.loc[matches["day"] <= cutoff_day], settings)
    assert len(truncated) == int((dataset.meta["day"] <= cutoff_day).sum())
    _same_rows(dataset, truncated, len(truncated))


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_splits_are_chronological(dataset: Dataset, settings: Settings, split: str) -> None:
    years = dataset.subset(split).meta["year"]  # type: ignore[arg-type]
    assert years.map(settings.split.split_of).eq(split).all()
