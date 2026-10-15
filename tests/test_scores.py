from __future__ import annotations

import pandas as pd
import pytest

from tennis_predictor.data.schema import Outcome, canonical_surface
from tennis_predictor.data.scores import (
    classify_outcome,
    count_games,
    normalize_score,
    parse_scores,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  6-4  6-3 ", "6-4 6-3"),
        ("6-4&nbsp;6-3", "6-4 6-3"),
        ("6-1 4-3 RET+H64", "6-1 4-3 RET"),
        (float("nan"), ""),
        (None, ""),
    ],
)
def test_normalize_score(raw: object, expected: str) -> None:
    assert normalize_score(raw) == expected


@pytest.mark.parametrize(
    ("score", "outcome"),
    [
        ("6-4 6-3", Outcome.COMPLETED),
        ("7-6(5) 6-7(3) 7-6(10)", Outcome.COMPLETED),
        ("6-3 3-6 [10-7]", Outcome.COMPLETED),
        ("6-1-6-2", Outcome.COMPLETED),  # typo, but a played match
        ("7-Oct", Outcome.COMPLETED),  # Excel-mangled, but a played match
        ("6-4 2-1 RET", Outcome.RETIRED),
        ("2-1 ret", Outcome.RETIRED),
        ("6-4 6-4 rtd.", Outcome.RETIRED),
        ("RET", Outcome.RETIRED),
        ("W/O", Outcome.WALKOVER),
        (" W/O", Outcome.WALKOVER),
        ("W.0", Outcome.WALKOVER),
        ("Walkover", Outcome.WALKOVER),
        ("6-3 1-0 DEF", Outcome.DEFAULT),
        ("Def.", Outcome.DEFAULT),
        ("6-4 3-6 Played and unfinished", Outcome.ABANDONED),
        ("3-6 8-10 ABD", Outcome.ABANDONED),
        ("5-4 ab", Outcome.ABANDONED),
        ("UNK", Outcome.UNKNOWN),
        ("BYE", Outcome.UNKNOWN),
        ("", Outcome.UNKNOWN),
    ],
)
def test_classify_outcome(score: str, outcome: Outcome) -> None:
    assert classify_outcome(score) is outcome


@pytest.mark.parametrize(
    ("score", "games"),
    [
        ("6-4 6-3", (12, 7)),
        ("7-6(5) 4-6 6-4", (17, 16)),
        ("6-3 3-6 [10-7]", (10, 9)),
        ("6-4 2-1 RET", (8, 5)),
        ("70-68", (70, 68)),
        ("7-Oct", None),
    ],
)
def test_count_games(score: str, games: tuple[int, int] | None) -> None:
    assert count_games(score) == games


def test_parse_scores_vectorized() -> None:
    parsed = parse_scores(pd.Series(["6-4 6-3", "W/O", None, "6-4 6-3"], index=[10, 11, 12, 13]))
    assert list(parsed.index) == [10, 11, 12, 13]
    assert parsed["outcome"].tolist() == ["completed", "walkover", "unknown", "completed"]
    assert parsed["w_games"].tolist() == [12, 0, 0, 12]


@pytest.mark.parametrize(
    ("raw", "surface"),
    [
        ("Hard", "Hard"),
        ("Carpet", "Hard"),
        ("clay", "Clay"),
        ("Grass", "Grass"),
        (None, "Unknown"),
        ("Ice", "Unknown"),
    ],
)
def test_canonical_surface(raw: object, surface: str) -> None:
    assert canonical_surface(raw) == surface
