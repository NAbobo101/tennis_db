from __future__ import annotations

import numpy as np
import pandas as pd

from tennis_predictor.config import Settings
from tennis_predictor.features.dataset import Dataset, eligible_mask, orientation_bit
from tennis_predictor.features.specs import FEATURE_NAMES


def test_eligibility_rules(matches: pd.DataFrame, settings: Settings) -> None:
    mask = eligible_mask(matches, settings)
    eligible = matches.loc[mask]
    assert set(eligible["tier"]) == {"main"}
    assert set(eligible["outcome"]) == {"completed"}
    assert not eligible["excluded"].any()
    assert "QF" in set(eligible["round"])  # quarter-finals are not qualifying rounds
    assert not eligible["round"].str.match(r"^Q\d$").any()
    assert set(eligible["tourney_level"]) <= set(settings.levels)
    assert not matches.loc[
        ~mask
        & (matches["tier"] == "main")
        & (matches["outcome"] == "completed")
        & ~matches["excluded"]
    ].shape[0]


def test_dataset_shape_and_labels(
    dataset: Dataset, matches: pd.DataFrame, settings: Settings
) -> None:
    assert list(dataset.X.columns) == list(FEATURE_NAMES)
    assert len(dataset) == int(eligible_mask(matches, settings).sum())
    assert set(np.unique(dataset.y)) == {0, 1}
    assert 0.4 < dataset.y.mean() < 0.6  # random but balanced orientation
    assert dataset.X.dtypes.eq("float64").all()


def test_metadata_is_consistent_with_labels(dataset: Dataset, matches: pd.DataFrame) -> None:
    lookup = matches.set_index(["tourney_id", "round", "winner_id", "loser_id"])
    for row, label in zip(dataset.meta.head(200).itertuples(), dataset.y[:200], strict=True):
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
