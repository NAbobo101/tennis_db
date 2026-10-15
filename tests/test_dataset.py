from __future__ import annotations

import numpy as np
import pandas as pd

from tennis_predictor.config import Settings
from tennis_predictor.features.dataset import Dataset, build_dataset, eligible_mask, orientation_bit
from tennis_predictor.features.specs import FEATURE_NAMES
from tennis_predictor.features.tracker import FeatureTracker

RATE_FEATURES = [
    name for name in FEATURE_NAMES if name.endswith(("_diff", "_score", "_log", "_min"))
]


def test_eligibility_rules(matches: pd.DataFrame, settings: Settings) -> None:
    mask = eligible_mask(matches, settings)
    eligible = matches.loc[mask]
    assert set(eligible["tier"]) == {"main"}
    assert set(eligible["outcome"]) == {"completed"}
    assert not eligible["excluded"].any()
    assert {"QF", "RR"} <= set(eligible["round"])  # quarter-finals are not qualifying rounds
    assert not eligible["round"].str.match(r"^Q\d$").any()
    assert set(eligible["tourney_level"]) == set(settings.levels)
    # Team cups and the burn-in season exist in the data but never become rows.
    assert (matches["tourney_level"] == "D").any()
    assert not (eligible["tourney_level"] == "D").any()
    assert (matches["year"] < settings.split.train_start).any()
    assert eligible["year"].min() == settings.split.train_start
    assert (matches["excluded"] & (matches["outcome"] == "completed")).any()


def test_every_played_match_feeds_the_ratings(
    matches: pd.DataFrame, built: tuple[Dataset, FeatureTracker]
) -> None:
    """Lower tiers, team cups, retirements and the burn-in season all update ratings."""
    _, tracker = built
    played = matches.loc[matches["outcome"].isin(["completed", "retired"]) & ~matches["excluded"]]
    counts = pd.concat([played["winner_id"], played["loser_id"]]).value_counts()
    for player_id, count in counts.items():
        assert tracker.players[int(player_id)].elo.matches == count
    assert set(played["tier"]) == {"main", "qual_chall"}


def test_lower_tiers_change_the_features(
    matches: pd.DataFrame, settings: Settings, dataset: Dataset
) -> None:
    main_only, _ = build_dataset(
        matches.loc[matches["tier"] == "main"].reset_index(drop=True), settings
    )
    assert len(main_only) == len(dataset)
    assert not main_only.X.equals(dataset.X)


def test_excluded_events_are_ignored(
    matches: pd.DataFrame, settings: Settings, dataset: Dataset
) -> None:
    without, _ = build_dataset(matches.loc[~matches["excluded"]].reset_index(drop=True), settings)
    pd.testing.assert_frame_equal(without.X, dataset.X)


def test_dataset_shape_and_labels(
    dataset: Dataset, matches: pd.DataFrame, settings: Settings
) -> None:
    assert list(dataset.X.columns) == list(FEATURE_NAMES)
    assert len(dataset) == int(eligible_mask(matches, settings).sum())
    assert set(np.unique(dataset.y)) == {0, 1}
    assert 0.4 < dataset.y.mean() < 0.6  # random but balanced orientation
    assert dataset.X.dtypes.eq("float64").all()


def test_missing_values_and_varied_flags(dataset: Dataset) -> None:
    X = dataset.X
    assert X["height_diff"].isna().any()  # missing and impossible heights
    assert X["log_points_diff"].notna().all()  # unranked players count as 0 points
    nan_allowed = {"age_diff", "height_diff", "log_points_diff"}
    assert X[[c for c in FEATURE_NAMES if c not in nan_allowed]].notna().all().all()
    for flag in ("unranked_diff", "qualifier_diff", "wildcard_diff", "after_retirement_diff"):
        assert set(np.unique(X[flag])) == {-1.0, 0.0, 1.0}, flag
    assert np.isfinite(X[[c for c in RATE_FEATURES if c not in nan_allowed]]).all().all()


def test_retirements_flag_the_next_match(matches: pd.DataFrame, dataset: Dataset) -> None:
    """The player who retired is flagged in their next training row, from either side."""
    retired = matches.loc[matches["outcome"] == "retired"]
    checked = 0
    for position, row in retired.iterrows():
        loser = row["loser_id"]
        later = dataset.meta.index[
            (dataset.meta["day"] > row["day"])
            & ((dataset.meta["a_id"] == loser) | (dataset.meta["b_id"] == loser))
        ]
        if len(later) == 0:
            continue
        next_row = int(later[0])
        played_between = matches.loc[
            (matches.index > position)
            & (matches["day"] < dataset.meta.loc[next_row, "day"])
            & ((matches["winner_id"] == loser) | (matches["loser_id"] == loser))
            & matches["outcome"].isin(["completed", "retired"])
        ]
        if len(played_between):
            continue  # another match in between resets the flag
        sign = 1.0 if dataset.meta.loc[next_row, "a_id"] == loser else -1.0
        assert dataset.X.loc[next_row, "after_retirement_diff"] in (sign, 0.0)
        checked += dataset.X.loc[next_row, "after_retirement_diff"] == sign
    assert checked >= 3


def test_metadata_is_consistent_with_labels(dataset: Dataset, matches: pd.DataFrame) -> None:
    lookup = matches.set_index(["tourney_id", "round", "winner_id", "loser_id"])
    for row, label in zip(dataset.meta.itertuples(), dataset.y, strict=True):
        winner, loser = (row.a_id, row.b_id) if label == 1 else (row.b_id, row.a_id)
        assert (row.tourney_id, row.round, winner, loser) in lookup.index


def test_orientation_is_deterministic() -> None:
    bits = {orientation_bit(42, "2020-580", n, 1, 2) for n in range(64)}
    assert bits == {0, 1}
    assert orientation_bit(42, "2020-580", 7, 1, 2) == orientation_bit(42, "2020-580", 7, 1, 2)


def test_subset(dataset: Dataset) -> None:
    parts = [dataset.subset(split) for split in ("train", "val", "test")]
    assert sum(len(part) for part in parts) == len(dataset)
    for part in parts:
        assert len(part.X) == len(part.y) == len(part.meta)
        assert part.X.index.equals(pd.RangeIndex(len(part)))
