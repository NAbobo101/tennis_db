"""The tests that justify every number in the README: no feature may see the future."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tennis_predictor.config import Settings
from tennis_predictor.data.schema import day_to_date
from tennis_predictor.features.dataset import Dataset, build_dataset
from tennis_predictor.features.specs import swap_players


def _same_rows(left: Dataset, right: Dataset, rows: int) -> None:
    pd.testing.assert_frame_equal(left.X.iloc[:rows], right.X.iloc[:rows])
    pd.testing.assert_frame_equal(left.meta.iloc[:rows], right.meta.iloc[:rows])
    np.testing.assert_array_equal(left.y[:rows], right.y[:rows])


def _position(matches: pd.DataFrame, meta: pd.Series) -> int:
    players = [meta["a_id"], meta["b_id"]]
    hits = matches.index[
        (matches["tourney_id"] == meta["tourney_id"])
        & (matches["round"] == meta["round"])
        & matches["winner_id"].isin(players)
        & matches["loser_id"].isin(players)
    ]
    assert len(hits) == 1
    return int(hits[0])


def _flip_result(matches: pd.DataFrame, position: int) -> pd.DataFrame:
    """Swap winner and loser (identities, statistics and games) of one match."""
    flipped = matches.copy()
    for column in matches.columns:
        if column.startswith("winner_"):
            other = "loser_" + column.removeprefix("winner_")
        elif column.startswith("w_"):
            other = "l_" + column.removeprefix("w_")
        else:
            continue
        flipped.loc[position, [column, other]] = matches.loc[position, [other, column]].to_numpy()
    return flipped


def _targets(dataset: Dataset) -> list[int]:
    """Rows spread over the history, including some with past meetings and retirements."""
    X = dataset.X
    spread = list(np.linspace(len(X) // 10, len(X) - 1, 12, dtype=int))
    with_h2h = X.index[X["h2h_matches_log"] > 0][
        :: max(1, int((X["h2h_matches_log"] > 0).sum()) // 8)
    ][:8]
    after_ret = X.index[X["after_retirement_diff"] != 0][:4]
    return sorted({int(i) for i in [*spread, *with_h2h, *after_ret]})


def test_features_ignore_the_match_result(
    matches: pd.DataFrame, settings: Settings, dataset: Dataset
) -> None:
    """Flipping the result of a match changes its label but never its own features."""
    kept_orientation = 0
    for target in _targets(dataset):
        meta = dataset.meta.iloc[target]
        changed, _ = build_dataset(_flip_result(matches, _position(matches, meta)), settings)
        _same_rows(dataset, changed, target)
        original = dataset.X.iloc[[target]].reset_index(drop=True)
        flipped = changed.X.iloc[[target]].reset_index(drop=True)
        if changed.meta.iloc[target]["a_id"] == meta["a_id"]:
            kept_orientation += 1
            assert changed.y[target] == 1 - dataset.y[target]
        else:
            flipped = swap_players(flipped)
        pd.testing.assert_frame_equal(original, flipped)
    # Rows whose orientation survives the flip are the ones able to expose a label leak.
    assert kept_orientation >= 4


def test_rank_and_h2h_match_independent_recomputation(
    matches: pd.DataFrame, settings: Settings, dataset: Dataset
) -> None:
    """Oracle checks of two features against values recomputed from the raw rows."""
    unranked = settings.features.unranked_rank
    a_rank = dataset.meta["a_rank"].astype(float).fillna(unranked).to_numpy()
    b_rank = dataset.meta["b_rank"].astype(float).fillna(unranked).to_numpy()
    np.testing.assert_allclose(dataset.X["log_rank_diff"], np.log(a_rank) - np.log(b_rank))

    played = matches.loc[matches["outcome"].isin(["completed", "retired"]) & ~matches["excluded"]]
    block = matches[["day", "tourney_id", "round_order"]].apply(tuple, axis=1)
    for target in _targets(dataset):
        meta = dataset.meta.iloc[target]
        a, b = meta["a_id"], meta["b_id"]
        position = _position(matches, meta)
        # Everything before the match's (week, event, round) block counts as history.
        block_start = int(block.index[block == block[position]][0])
        earlier = played.loc[played.index < block_start]
        wins_a = int(((earlier["winner_id"] == a) & (earlier["loser_id"] == b)).sum())
        wins_b = int(((earlier["winner_id"] == b) & (earlier["loser_id"] == a)).sum())
        expected = (wins_a - wins_b) / (wins_a + wins_b + settings.features.h2h_prior)
        assert dataset.X.loc[target, "h2h_score"] == pytest.approx(expected)
        assert dataset.X.loc[target, "h2h_matches_log"] == pytest.approx(
            math.log1p(wins_a + wins_b)
        )


def test_truncation_invariance_by_week(
    matches: pd.DataFrame, settings: Settings, dataset: Dataset
) -> None:
    """Building the dataset from a prefix of history reproduces the same rows exactly."""
    cutoff_day = int(dataset.meta["day"].quantile(0.4))
    truncated, _ = build_dataset(matches.loc[matches["day"] <= cutoff_day], settings)
    assert len(truncated) == int((dataset.meta["day"] <= cutoff_day).sum())
    _same_rows(dataset, truncated, len(truncated))


def test_truncation_invariance_inside_events(
    matches: pd.DataFrame, settings: Settings, dataset: Dataset
) -> None:
    """Cutting history in the middle of an event (after qualifying, after the quarter-finals,
    inside a round robin) must not change any earlier row."""
    main = matches.loc[matches["tier"] == "main"]
    cuts = []
    for tourney_id in ("2016-001", "2019-004", "2021-007", "2023-F01"):
        event = main.loc[main["tourney_id"] == tourney_id]
        cuts.append(
            int(matches.index[matches["tourney_id"] == tourney_id][0])
        )  # after earlier weeks
        cuts.append(int(event.index[event["round"].isin(["QF", "RR"])][-1]) + 1)
        cuts.append(int(event.index[event["round"].isin(["R16", "RR"])][2]))  # inside a round
    for cut in cuts:
        truncated, _ = build_dataset(matches.iloc[:cut], settings)
        _same_rows(dataset, truncated, len(truncated))


@pytest.mark.parametrize("tourney_id", ["2016-F01", "2023-F01"])
def test_round_robin_order_is_irrelevant(
    matches: pd.DataFrame, settings: Settings, dataset: Dataset, tourney_id: str
) -> None:
    """Upstream numbers round-robin matches by player, not by date: any order of a stage
    must give the same features for every match of the stage and everything before it."""
    stage = matches.index[
        (matches["tourney_id"] == tourney_id) & (matches["round"] == "RR")
    ].to_numpy()
    order = np.arange(len(matches))
    order[stage] = np.random.default_rng(3).permutation(stage)
    changed, _ = build_dataset(matches.iloc[order].reset_index(drop=True), settings)

    def stage_and_before(data: Dataset) -> pd.DataFrame:
        meta = data.meta
        in_stage = (meta["tourney_id"] == tourney_id) & (meta["round"] == "RR")
        keep = in_stage | (meta["day"] < meta.loc[in_stage, "day"].min())
        key = ["tourney_id", "round", "a_id", "b_id"]
        rows = pd.concat([meta.loc[keep, key], data.X.loc[keep]], axis=1)
        return pd.DataFrame(rows.sort_values(key).reset_index(drop=True))

    original = stage_and_before(dataset)
    assert ((original["tourney_id"] == tourney_id) & (original["round"] == "RR")).sum() == 12
    pd.testing.assert_frame_equal(original, stage_and_before(changed))


def test_splits_are_chronological(dataset: Dataset, settings: Settings) -> None:
    train, val, test = (dataset.subset(split).meta for split in ("train", "val", "test"))
    assert train["day"].max() < val["day"].min()
    assert val["day"].max() < test["day"].min()
    years = {
        split: part["year"] for split, part in (("train", train), ("val", val), ("test", test))
    }
    assert years["train"].between(settings.split.train_start, settings.split.train_end).all()
    assert years["val"].between(*settings.split.val_years).all()
    assert years["test"].between(*settings.split.test_years).all()
    assert (
        dataset.meta["year"] == [day_to_date(int(day)).year for day in dataset.meta["day"]]
    ).all()
