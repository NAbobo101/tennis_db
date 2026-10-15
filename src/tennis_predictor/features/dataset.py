"""Turn the chronological match history into a supervised dataset."""

from __future__ import annotations

import logging
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from tennis_predictor.config import Settings
from tennis_predictor.data.schema import (
    GRAND_SLAM,
    LEVEL_ORDINAL,
    LOSER_STATS,
    QUALIFYING_ROUNDS,
    WINNER_STATS,
    Outcome,
)
from tennis_predictor.features.specs import FEATURE_NAMES
from tennis_predictor.features.tracker import FeatureTracker, MatchContext, MatchResult, PlayerInfo

logger = logging.getLogger(__name__)

Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class Dataset:
    """Feature matrix ``X``, labels ``y`` (1 when player A won) and per-row metadata."""

    X: pd.DataFrame
    y: npt.NDArray[np.int8]
    meta: pd.DataFrame

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, split: Split) -> Dataset:
        """Rows of one chronological split."""
        mask = (self.meta["split"] == split).to_numpy()
        return Dataset(
            X=self.X.loc[mask].reset_index(drop=True),
            y=self.y[mask],
            meta=self.meta.loc[mask].reset_index(drop=True),
        )


def eligible_mask(matches: pd.DataFrame, settings: Settings) -> npt.NDArray[np.bool_]:
    """Rows used for training and evaluation: completed tour-level main-draw matches."""
    years = matches["year"].to_numpy()
    in_window = (years >= settings.split.train_start) & (years <= settings.split.test_years[1])
    return np.asarray(
        (matches["tier"] == "main")
        & matches["tourney_level"].isin(settings.levels)
        & ~matches["round"].isin(QUALIFYING_ROUNDS)
        & (matches["outcome"] == Outcome.COMPLETED.value)
        & ~matches["excluded"]
        & in_window,
        dtype=bool,
    )


def orientation_bit(
    seed: int, tourney_id: str, match_num: int, winner_id: int, loser_id: int
) -> int:
    """Deterministic coin flip deciding whether the winner is shown as player B.

    It depends only on the match itself, so the orientation of a row never changes when
    other rows are added or removed.
    """
    key = f"{seed}|{tourney_id}|{match_num}|{winner_id}|{loser_id}".encode()
    return zlib.crc32(key) & 1


@dataclass(frozen=True, slots=True)
class _PlayerColumns:
    """Typed per-row player columns for one side (winner or loser) of the match table."""

    ids: list[int]
    names: list[str]
    ranks: list[float]
    points: list[float]
    ages: list[float]
    heights: list[float]
    hands: list[str]
    entries: list[str]
    iocs: list[str]

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, side: str) -> _PlayerColumns:
        def text(field: str) -> list[str]:
            return [str(value) for value in frame[f"{side}_{field}"].tolist()]

        def number(field: str) -> list[float]:
            return [float(value) for value in frame[f"{side}_{field}"].to_numpy(dtype=float)]

        return cls(
            ids=[int(value) for value in frame[f"{side}_id"].to_numpy(dtype=np.int64)],
            names=text("name"),
            ranks=number("rank"),
            points=number("rank_points"),
            ages=number("age"),
            heights=number("ht"),
            hands=text("hand"),
            entries=text("entry"),
            iocs=text("ioc"),
        )

    def info(self, i: int) -> PlayerInfo:
        return PlayerInfo(
            player_id=self.ids[i],
            name=self.names[i],
            rank=self.ranks[i],
            rank_points=self.points[i],
            age=self.ages[i],
            height=self.heights[i],
            hand=self.hands[i],
            entry=self.entries[i],
            ioc=self.iocs[i],
        )


def _ints(frame: pd.DataFrame, column: str) -> list[int]:
    return [int(value) for value in frame[column].to_numpy(dtype=np.int64)]


def _texts(frame: pd.DataFrame, column: str) -> list[str]:
    return [str(value) for value in frame[column].tolist()]


def build_dataset(matches: pd.DataFrame, settings: Settings) -> tuple[Dataset, FeatureTracker]:
    """Replay the whole history once, snapshotting features before every eligible match.

    Returns the dataset and the tracker in its final state (used for serving predictions).
    """
    tracker = FeatureTracker(settings.elo, settings.features, settings.tour)
    eligible = eligible_mask(matches, settings).tolist()
    level_ordinal = LEVEL_ORDINAL[settings.tour]

    days, best_ofs = _ints(matches, "day"), _ints(matches, "best_of")
    round_orders, match_nums = _ints(matches, "round_order"), _ints(matches, "match_num")
    w_games, l_games = _ints(matches, "w_games"), _ints(matches, "l_games")
    tourney_ids, levels = _texts(matches, "tourney_id"), _texts(matches, "tourney_level")
    surfaces, outcomes = (
        _texts(matches, "surface"),
        [Outcome(v) for v in _texts(matches, "outcome")],
    )
    excluded = matches["excluded"].to_numpy(dtype=bool).tolist()
    stats_ok = matches["stats_ok"].to_numpy(dtype=bool).tolist()
    winners = _PlayerColumns.from_frame(matches, "winner")
    losers = _PlayerColumns.from_frame(matches, "loser")
    winner_stats = matches[list(WINNER_STATS)].to_numpy(dtype=float).tolist()
    loser_stats = matches[list(LOSER_STATS)].to_numpy(dtype=float).tolist()

    rows: list[list[float]] = []
    labels: list[int] = []
    positions: list[int] = []
    flips: list[int] = []
    for i in range(len(matches)):
        if excluded[i]:
            continue
        ctx = MatchContext(
            day=days[i],
            tourney_id=tourney_ids[i],
            surface=surfaces[i],
            best_of=best_ofs[i],
            level_ordinal=level_ordinal.get(levels[i], 0),
            round_order=round_orders[i],
            is_slam=levels[i] == GRAND_SLAM,
        )
        winner, loser = winners.info(i), losers.info(i)
        if eligible[i]:
            flip = orientation_bit(
                settings.seed, ctx.tourney_id, match_nums[i], winner.player_id, loser.player_id
            )
            a, b = (loser, winner) if flip else (winner, loser)
            rows.append(tracker.snapshot(a, b, ctx))
            labels.append(1 - flip)
            positions.append(i)
            flips.append(flip)
        result = MatchResult(
            outcome=outcomes[i],
            winner_stats=tuple(winner_stats[i]) if stats_ok[i] else None,
            loser_stats=tuple(loser_stats[i]) if stats_ok[i] else None,
            winner_games=w_games[i],
            loser_games=l_games[i],
        )
        tracker.update(winner, loser, ctx, result)

    X = pd.DataFrame(rows, columns=list(FEATURE_NAMES), dtype="float64")
    meta = _metadata(matches, positions, flips, settings.split.split_of)
    logger.info("Built %d rows from %d matches", len(X), len(matches))
    return Dataset(X=X, y=np.asarray(labels, dtype=np.int8), meta=meta), tracker


def _metadata(
    matches: pd.DataFrame,
    positions: list[int],
    flips: list[int],
    split_of: Callable[[int], str | None],
) -> pd.DataFrame:
    source = matches.iloc[positions].reset_index(drop=True)
    flipped = np.asarray(flips, dtype=bool)
    meta = pd.DataFrame(
        {
            "day": source["day"].to_numpy(),
            "year": source["year"].to_numpy(),
            "tourney_id": source["tourney_id"].to_numpy(),
            "tourney_name": source["tourney_name"].to_numpy(),
            "tourney_level": source["tourney_level"].to_numpy(),
            "surface": source["surface"].to_numpy(),
            "best_of": source["best_of"].to_numpy(),
            "round": source["round"].to_numpy(),
        }
    )
    for side, own, other in (("a", "winner", "loser"), ("b", "loser", "winner")):
        for field in ("id", "name", "rank"):
            meta[f"{side}_{field}"] = np.where(
                flipped, source[f"{other}_{field}"].to_numpy(), source[f"{own}_{field}"].to_numpy()
            )
    meta["split"] = [split_of(int(year)) for year in meta["year"]]
    return meta
