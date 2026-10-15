"""Constants describing the Sackmann match files."""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import Final, NamedTuple

STAT_FIELDS: Final = (
    "ace",
    "df",
    "svpt",
    "1stIn",
    "1stWon",
    "2ndWon",
    "SvGms",
    "bpSaved",
    "bpFaced",
)
WINNER_STATS: Final = tuple(f"w_{field}" for field in STAT_FIELDS)
LOSER_STATS: Final = tuple(f"l_{field}" for field in STAT_FIELDS)

PLAYER_FIELDS: Final = ("id", "name", "hand", "ht", "age", "entry", "rank", "rank_points", "ioc")

#: Columns read from every match file (the remaining ones are never used).
MATCH_COLUMNS: Final = (
    "tourney_id",
    "tourney_name",
    "surface",
    "tourney_level",
    "tourney_date",
    "match_num",
    "round",
    "best_of",
    "score",
    *(f"winner_{field}" for field in PLAYER_FIELDS),
    *(f"loser_{field}" for field in PLAYER_FIELDS),
    *WINNER_STATS,
    *LOSER_STATS,
)

NUMERIC_COLUMNS: Final = (
    "match_num",
    "best_of",
    "winner_ht",
    "winner_age",
    "winner_rank",
    "winner_rank_points",
    "loser_ht",
    "loser_age",
    "loser_rank",
    "loser_rank_points",
    *WINNER_STATS,
    *LOSER_STATS,
)

#: Chronological order of rounds inside one tournament. Round robin sits between the
#: last knockout round of the group stage and the semi-finals (Tour Finals, United Cup).
ROUND_ORDER: Final = {
    "Q1": 0,
    "Q2": 1,
    "Q3": 2,
    "Q4": 3,
    "Q5": 4,
    "ER": 5,
    "R128": 6,
    "R64": 7,
    "R32": 8,
    "R16": 9,
    "RR": 10,
    "QF": 11,
    "SF": 12,
    "BR": 13,
    "F": 14,
}
#: Never test qualifying with ``startswith("Q")``: that also matches quarter-finals.
QUALIFYING_ROUNDS: Final = frozenset({"Q1", "Q2", "Q3", "Q4", "Q5"})

SURFACES: Final = ("Hard", "Clay", "Grass")
UNKNOWN_SURFACE: Final = "Unknown"
GRAND_SLAM: Final = "G"

#: Ordinal importance of each tournament level, used as a context feature.
LEVEL_ORDINAL: Final[dict[str, dict[str, int]]] = {
    "atp": {"G": 4, "F": 3, "M": 2, "O": 2, "A": 1},
    "wta": {"G": 4, "F": 3, "PM": 2, "O": 2, "P": 1, "I": 0, "W": 0},
}


class FileFamily(NamedTuple):
    """Location of one family of yearly match files."""

    repo: str
    pattern: str


FILE_FAMILIES: Final[dict[tuple[str, str], FileFamily]] = {
    ("atp", "main"): FileFamily("tennis_atp", "atp_matches_{year}.csv"),
    ("atp", "qual_chall"): FileFamily("tennis_atp", "atp_matches_qual_chall_{year}.csv"),
    ("atp", "futures"): FileFamily("tennis_atp", "atp_matches_futures_{year}.csv"),
    ("wta", "main"): FileFamily("tennis_wta", "wta_matches_{year}.csv"),
    ("wta", "qual_itf"): FileFamily("tennis_wta", "wta_matches_qual_itf_{year}.csv"),
}


class Outcome(StrEnum):
    """How a match ended, parsed from the ``score`` column."""

    COMPLETED = "completed"
    RETIRED = "retired"
    WALKOVER = "walkover"
    DEFAULT = "default"
    ABANDONED = "abandoned"
    UNKNOWN = "unknown"


#: Outcomes that count as a played match for ratings, form and head-to-head.
PLAYED_OUTCOMES: Final = frozenset({Outcome.COMPLETED, Outcome.RETIRED})


#: Day numbers count days since this date, so every match in the data has a positive day.
EPOCH: Final = dt.date(1900, 1, 1)


def date_to_day(date: dt.date) -> int:
    """Convert a calendar date to the internal day number."""
    return (date - EPOCH).days


def day_to_date(day: int) -> dt.date:
    """Convert an internal day number back to a calendar date."""
    return EPOCH + dt.timedelta(days=day)


def canonical_surface(raw: object) -> str:
    """Map a raw surface to Hard, Clay, Grass or Unknown (Carpet is merged into Hard)."""
    if not isinstance(raw, str):
        return UNKNOWN_SURFACE
    value = raw.strip().capitalize()
    if value == "Carpet":
        return "Hard"
    return value if value in SURFACES else UNKNOWN_SURFACE
